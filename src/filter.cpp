#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "cblas.h"
#include "funcs.hpp"
#include "quads.hpp"
#include "polynomials.hpp"
#include "filter.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#include "filter_kernels.h"
#endif


void Filter::setup(InputStruct *input, FRSolver &solver)
{
  /* Assign local pointers */
  this->input = input;
  order = solver.order;
  geo = &solver.geo;
  eles = solver.eles;
  faces = solver.faces;
  this->solver = &solver;
  
  /* Setup sensor */
  setup_vandermonde_matrices();
  setup_concentration_matrix();
  setup_oppS();
  setup_threshold();
  U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  sensor.assign({eles->nEles});

  
  /* Setup filter */
  DeltaHat.resize(input->filt_maxLevels);
  oppF_1D.resize(input->filt_maxLevels);
  oppF_spts.resize(input->filt_maxLevels);
  oppF_fpts.resize(input->filt_maxLevels);
  for (unsigned int level = 0; level < input->filt_maxLevels; level++)
  {
    setup_DeltaHat(level);
    setup_oppF_1D(level);
    setup_oppF(level);
  }

  /* Copy data to GPU */
#ifdef _GPU
  U_spts_d = U_spts;
  oppS_d = oppS;
  KS_d = KS;
  sensor_d = sensor;

  oppF_spts_d.resize(input->filt_maxLevels);
  oppF_fpts_d.resize(input->filt_maxLevels);
  for (unsigned int level = 0; level < input->filt_maxLevels; level++)
  {
    oppF_spts_d[level] = oppF_spts[level];
    oppF_fpts_d[level] = oppF_fpts[level];
  }
#endif
}


void Filter::setup_vandermonde_matrices()
{
  /* Vandermonde matrix for the orthonormal basis */
  Vander.assign({eles->nSpts1D, order + 1});
  for (unsigned int j = 0; j <= order; j++)
  {
    double normC = std::sqrt(2.0 / (2.0 * j + 1.0));
    for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
      Vander(spt, j) = Legendre(j, eles->loc_spts_1D[spt]) / normC;
  }
  
  /* Calculate inverse of Vandermonde matrix */
  VanderInv.assign({eles->nSpts1D, order + 1});
  Vander.calc_LU();
  mdvector<double> eye({eles->nSpts1D, order + 1});
  for (unsigned int j = 0; j <= order; j++)
    eye(j,j) = 1.0;
  Vander.solve(VanderInv, eye);
  
  /* Vandermonde matrix for the derivatives of the orthonormal basis functions */
  Vander_d1.assign({eles->nSpts1D, order + 1});
  for (unsigned int j = 0; j <= order; j++)
  {
    double normC = std::sqrt(2.0 / (2.0 * j + 1.0));
    for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
      Vander_d1(spt, j) = Legendre_d1(j, eles->loc_spts_1D[spt]) / normC;
  }
}


void Filter::setup_concentration_matrix()
{
  Conc.assign({eles->nSpts1D, order + 1});
  oppS_1D.assign({eles->nSpts1D, order + 1});
  
  if (order == 0) 
  {
    Conc(0,0) = 0.0;
    return;
  }
  
  for (unsigned int j = 0; j <= order; j++)
  {
    for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
    {
      double x = eles->loc_spts_1D[spt];
      Conc(spt, j) = (M_PI / order) *std::sqrt(1.0 - x*x) *Vander_d1(spt, j);
    }
  }
}


void Filter::setup_threshold()
{
  // Normalization tolerance
  normalTol = 0.1;
  
  // Local arrays
  mdvector<double> u_canon, KS_canon, KS_step, KS_ramp;
  u_canon.assign({eles->nSpts1D, 2});
  KS_canon.assign({eles->nSpts1D, 2});
  KS_step.assign({eles->nSpts1D});
  KS_ramp.assign({eles->nSpts1D});
  
  // Centered step in parent domain
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
    u_canon(spt, 0) = step(eles->loc_spts_1D[spt]);
    
  // Ramp in parent domain
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
    u_canon(spt, 1) = step(eles->loc_spts_1D[spt]) *eles->loc_spts_1D[spt];
  
  // Evaluate filtered kernel
  auto &A = oppS_1D(0, 0);
  auto &B = u_canon(0, 0);
  auto &C = KS_canon(0, 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2, eles->nSpts1D, 1.0, &A, oppS_1D.ldim(), &B, u_canon.ldim(), 0.0, &C, KS_canon.ldim());
  
  // Apply non-linear enhancement
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    KS_step(spt) = order * (KS_canon(spt, 0) * KS_canon(spt, 0));
    KS_ramp(spt) = order * (KS_canon(spt, 1) * KS_canon(spt, 1));
  }
  
  // Calculate threshold
  threshJ = (1.0 - input->sen_Jfac) *KS_ramp.max_val() + input->sen_Jfac *KS_step.max_val();
    
  // Print results
  if (input->rank == 0) 
  {
    std::cout << " Sensor threshold = " << threshJ << std::endl;
  }
}


void Filter::setup_oppS()
{
  // 1D Sensor matrix
  auto &A = Conc(0, 0);
  auto &B = VanderInv(0, 0);
  auto &C = oppS_1D(0, 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, eles->nSpts1D, eles->nSpts1D, 1.0, &A, Conc.ldim(), &B, VanderInv.ldim(), 0.0, &C, oppS_1D.ldim());
  
  // Sensor operator
  if (input->nDims == 2) // Quads
  {
    oppS.assign({2 * eles->nSpts, eles->nSpts}, 0.0);
    KS.assign({2 * eles->nSpts, eles->nEles, eles->nVars});
    
    // xi lines
    for (unsigned int k = 0; k < eles->nSpts1D; k++)
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
        for (unsigned int i = 0; i < eles->nSpts1D; i++)
          oppS(i + k*eles->nSpts1D, j + k*eles->nSpts1D) = oppS_1D(i,j);
      
    // eta lines
    for (unsigned int k = 0; k < eles->nSpts1D; k++)
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
        for (unsigned int i = 0; i < eles->nSpts1D; i++)
          oppS(eles->nSpts + i + k*eles->nSpts1D, j*eles->nSpts1D + k) = oppS_1D(i,j);   
  } 
  else // Hexes   
  {
    ThrowException("Sensor on hexes hasn't been implemented yet.");
  }
}


void Filter::apply_sensor()
{
#ifdef _CPU
  //Initialize sensor
  sensor.fill(0.0); 
  
  // Copy data to local structure
  U_spts = eles->U_spts;
  
  // Normalize data
  if (input->sen_norm)
  {
    for (unsigned int var = 0; var < eles->nVars; var++)
    {
      // Find element maximum and minimum
#pragma omp parallel for 
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
      double uMax = U_spts(0,ele,var), uMin = U_spts(0,ele,var);
#pragma omp parallel for reduction(max:uMax) reduction(min:uMin)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          uMax = std::max(uMax, U_spts(spt,ele,var));
          uMin = std::min(uMin, U_spts(spt,ele,var));
        }
        
        if (uMax - uMin > normalTol)
        {
#pragma omp parallel for 
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
            U_spts(spt,ele,var) = (U_spts(spt,ele,var) - uMin) / (uMax - uMin);
        }
      }
    }
  }
  
  // Calculate KS
  auto &A = oppS(0,0);
  auto &B = U_spts(0, 0, 0);
  auto &C = KS(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    2 * eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, oppS.ldim(), &B, U_spts.ldim(), 0.0, &C, KS.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
    2 * eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, oppS.ldim(), &B, U_spts.ldim(), 0.0, &C, KS.ldim());
#endif
    
    // Apply non-liqnear enhancement and store sensor values
  for (unsigned int var = 0; var < eles->nVars; var++)
  {
#pragma omp parallel for
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      double sen = 0.0;
#pragma omp parallel for reduction(max:sen)
      for (unsigned int row = 0; row < 2*eles->nSpts; row++)
      {
        KS(row, ele, var) = order * (KS(row, ele, var) * KS(row, ele, var));
        sen = std::max(sen, KS(row, ele, var));
      }
      sensor(ele) = std::max(sensor(ele), sen);
    }
  }
#endif

#ifdef _GPU
  
  // Copy data to local structure
  device_copy(U_spts_d, eles->U_spts_d, U_spts_d.max_size());
  
  // Normalize data
  if (input->sen_norm)
  {
    normalize_data_wrapper(U_spts_d, normalTol, eles->nSpts, eles->nEles, eles->nVars);
  }
  
  // Calculate KS
  cublasDGEMM_wrapper(2 * eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0,
    oppS_d.data(), 2 * eles->nSpts, U_spts_d.data(), eles->nSpts, 0.0,
    KS_d.data(), 2 * eles->nSpts);
  
  // Apply non-linear enhancement and store sensor values
  compute_max_sensor_wrapper(KS_d, sensor_d, order, max_sensor_d, eles->nSpts, eles->nEles, eles->nVars);

#endif
}
    
  
void Filter::setup_DeltaHat(unsigned int level)
{ 
  // Determine filter width in parent space
  double DH;
  if (input->dt_type) // CFL based time step
    DH = input->filt_gamma * std::sqrt(input->CFL) * 2.0 / std::pow(order + 1.0, 0.25);
  else // Exogenously fixed time step
    DH = input->filt_gamma * std::sqrt(0.25) * 2.0 / std::pow(order + 1.0, 0.25);
  
  // Level scaling
  DH *= (level + 1);
    
  // Check for kernel positivity
  double DeltaHatMax = 2.0 / (order + 1.0);
  if (DH <= 0  || DH > DeltaHatMax)
  {
    if (input->rank == 0) std::cout << " WARNING: Negative filter kernel at level " << level << ". > Reduce gamma or maxLevels. " << std::endl;
  }
  
  // Assign
  DeltaHat[level] = DH;
}


void Filter::setup_oppF_1D(unsigned int level)
{   
  // Assign filter matrix
  oppF_1D[level].assign({eles->nSpts1D, eles->nSpts1D + 2});
  double DH = DeltaHat[level];

  // Loop over solution points - rows
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    // Internal weights - first P+1 columns
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
    {
      // Set integration limits
      double xiL = std::max(-1.0, eles->loc_spts_1D[spt] - 0.5*DH);
      double xiR = std::min(1.0, eles->loc_spts_1D[spt] + 0.5*DH);

      // Evaluate integral through quadrature
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
      {
        double xi = xiL + (xiR - xiL) * (eles->loc_spts_1D[j] + 1.0) / 2.0;
        double fun = 0.5 * (xiR-xiL)* Lagrange(eles->loc_spts_1D, i, xi) / DH;
        oppF_1D[level](spt, i) += fun * eles->weights_spts(j);
      }
    }
    
    // Boundary weights
    oppF_1D[level](spt, eles->nSpts1D) = std::max(0.0, 0.5*DH -1.0 -eles->loc_spts_1D[spt]) / DH;
    oppF_1D[level](spt, eles->nSpts1D + 1) = std::max(0.0, eles->loc_spts_1D[spt] +0.5*DH -1) / DH;
  }
  
  // Evaluate error measure for matrix entries
  mdvector<double> unity({eles->nSpts1D});
  double tol = 0;
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    for (unsigned int i = 0; i < eles->nSpts1D + 2; i++)
      unity(spt) += oppF_1D[level](spt, i);
    tol += std::abs(unity(spt) - 1.0);
  }
  if (tol > 1e-10 && input->rank == 0) std::cout << " WARNING: Filter matrix elements are not converged! Tolerance = " << tol << std::endl;
}


void Filter::setup_oppF(unsigned int level)
{   
  // Filter operator
  if (input->nDims == 2) // Quads
  {
    mdvector<double> F_spts({2 * eles->nSpts, eles->nSpts}, 0.0);
    mdvector<double> F_fpts({2 * eles->nSpts, eles->nFpts}, 0.0);
    
    // xi lines
    for (unsigned int k = 0; k < eles->nSpts1D; k++)
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
        for (unsigned int i = 0; i < eles->nSpts1D; i++)
          F_spts(i + k*eles->nSpts1D, j + k*eles->nSpts1D) = oppF_1D[level](i,j);
      
    // eta lines
    for (unsigned int k = 0; k < eles->nSpts1D; k++)
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
        for (unsigned int i = 0; i < eles->nSpts1D; i++)
          F_spts(eles->nSpts + i + k*eles->nSpts1D, j*eles->nSpts1D + k) = oppF_1D[level](i,j);
        
    // Bottom edge
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      for (unsigned int i = 0; i < eles->nSpts1D; i++)
        F_fpts(eles->nSpts + j*eles->nSpts1D + i, j) = oppF_1D[level](i, eles->nSpts1D);
    
    // Right edge
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      for (unsigned int i = 0; i < eles->nSpts1D; i++)
        F_fpts(j*eles->nSpts1D + i, eles->nSpts1D + j) = oppF_1D[level](i, eles->nSpts1D + 1);
    
    // Top edge
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      for (unsigned int i = 0; i < eles->nSpts1D; i++)
        F_fpts(eles->nSpts + j*eles->nSpts1D + i, 2*eles->nSpts1D + (eles->nSpts1D - j - 1)) = oppF_1D[level](i, eles->nSpts1D + 1);
    
    // Left edge
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      for (unsigned int i = 0; i < eles->nSpts1D; i++)
        F_fpts(j*eles->nSpts1D + i, 3*eles->nSpts1D + (eles->nSpts1D - j - 1)) = oppF_1D[level](i, eles->nSpts1D);
      
    // Averaging
    mdvector<double> half({eles->nSpts, 2 * eles->nSpts}, 0.0);
    for (unsigned int spt = 0; spt < eles->nSpts; spt++)
    {
      half(spt, spt) = 0.5;
      half(spt, eles->nSpts + (spt / eles->nSpts1D) + (spt % eles->nSpts1D) *eles->nSpts1D) = 0.5;
    }
      
    // Composition
    oppF_spts[level].assign({eles->nSpts, eles->nSpts}, 0.0);
    auto &A = half(0, 0);
    auto &B = F_spts(0, 0);
    auto &C = oppF_spts[level](0, 0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
      eles->nSpts, eles->nSpts, 2 * eles->nSpts, 1.0, &A, half.ldim(), &B, F_spts.ldim(), 0.0, &C, oppF_spts[level].ldim());
    
    oppF_fpts[level].assign({eles->nSpts, eles->nFpts}, 0.0);
    auto &D = F_fpts(0, 0);
    auto &E = oppF_fpts[level](0, 0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
      eles->nSpts, eles->nFpts, 2 * eles->nSpts, 1.0, &A, half.ldim(), &D, F_fpts.ldim(), 0.0, &E, oppF_fpts[level].ldim() );
      
  } 
  else // Hexes   
  {
    ThrowException("Filter on hexes hasn't been implemented yet.");
  }
}


unsigned int Filter::apply_filter(unsigned int level)
{
  
  // Check if filtering is required
#ifdef _CPU
  if (sensor.max_val() < threshJ) return 0;
#endif
#ifdef _GPU
  if (max_sensor_d < threshJ) return 0;
#endif
  
  // Transfer information to flux points
  eles->extrapolate_U(0, eles->nEles); 
  solver->U_to_faces(0, eles->nEles); 
#ifdef _MPI 
  faces->send_U_data();
  faces->recv_U_data();
  faces->compute_common_U(geo->nGfpts_int + geo->nGfpts_bnd, geo->nGfpts);
#else
  faces->compute_common_U(0, geo->nGfpts); 
#endif 
  solver->U_from_faces(0, eles->nEles); 
 
#ifdef _CPU
  // Loop over variables
  // Contribution from solution points
  auto &A = oppF_spts[level](0,0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = U_spts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, oppF_spts[level].ldim(), &B, eles->U_spts.ldim(), 0.0, &C, U_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, oppF_spts[level].ldim(), &B, eles->U_spts.ldim(), 0.0, &C, U_spts.ldim());
#endif
  
  // Contribution from flux points
  auto &D = oppF_fpts[level](0,0);
  auto &E = eles->U_fpts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts, eles->nEles * eles->nVars, eles->nFpts, 1.0, &D, oppF_fpts[level].ldim(), &E, eles->U_fpts.ldim(), 1.0, &C, U_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts, eles->nEles * eles->nVars, eles->nFpts, 1.0, &D, oppF_fpts[level].ldim(), &E, eles->U_fpts.ldim(), 1.0, &C, U_spts.ldim());
#endif
   
  // Copy back filtered values
#pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    // Check for sensor value
    if (sensor(ele) < threshJ) continue;

#pragma omp parallel for collapse(2)    
    for (unsigned int var = 0; var < eles->nVars; var++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        eles->U_spts(spt,ele,var) = U_spts(spt,ele,var);
  }
#endif

#ifdef _GPU
  // Loop over variables
  // Contribution from solution points
  cublasDGEMM_wrapper(eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0,
    oppF_spts_d[level].data(), eles->nSpts, eles->U_spts_d.data(), eles->nSpts, 0.0,
    U_spts_d.data(), eles->nSpts);

  // Contribution from flux points
  cublasDGEMM_wrapper(eles->nSpts, eles->nEles * eles->nVars, eles->nFpts, 1.0,
    oppF_fpts_d[level].data(), eles->nSpts, eles->U_fpts_d.data(), eles->nFpts, 1.0,
    U_spts_d.data(), eles->nSpts);
   
  // Copy back filtered values
  copy_filtered_solution_wrapper(U_spts_d, eles->U_spts_d, sensor_d, threshJ, eles->nSpts, eles->nEles, eles->nVars);

#endif
  
  return 1;
}
