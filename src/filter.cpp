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
  setup_threshold();
  setup_reshapeOp(); 
	sensor.assign({eles->nEles});
  
  /* Setup filter */
  setup_DeltaHat();
  setup_Fop();
  setup_appendOp();
	
#ifdef _GPU
	sensor_d = sensor;
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
	
	/* LU factorization for the Vandermonde matrix */
  Vander.calc_LU();
	
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
  // Local arrays
  mdvector<double> u_canon, uh_canon, KS, KS_step, KS_ramp;
  u_canon.assign({eles->nSpts1D, 2});
  uh_canon.assign({eles->nSpts1D, 2});
  KS.assign({eles->nSpts1D, 2});
  KS_step.assign({eles->nSpts1D});
  KS_ramp.assign({eles->nSpts1D});
  
	// Centered step in parent domain
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
    u_canon(spt, 0) = step(eles->loc_spts_1D[spt]);
    
  // Ramp in parent domain
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
    u_canon(spt, 1) = step(eles->loc_spts_1D[spt]) *eles->loc_spts_1D[spt];
  
  // Evaluate modal coefficients 
  Vander.solve(uh_canon, u_canon);

  // Evaluate filtered kernel
  auto &A = Conc(0, 0);
  auto &B = uh_canon(0, 0);
  auto &C = KS(0, 0);
  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2, eles->nSpts1D, 1.0, &A, eles->nSpts1D, &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
  
  // Apply non-linear enhancement
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    KS_step(spt) = order * (KS(spt, 0) * KS(spt, 0));
    KS_ramp(spt) = order * (KS(spt, 1) * KS(spt, 1));
  }
  
  // Calculate threshold
  threshJ = (1.0 - input->sen_Jfac) *KS_ramp.max_val() + input->sen_Jfac *KS_step.max_val();
    
  // Print results
  if (input->rank == 0) 
  {
    std::cout << " Sensor threshold values:   ";
    std::cout << " Step = " << KS_step.max_val() << "  Ramp = " << KS_ramp.max_val() << "  Weighted = " << threshJ << std::endl;
  }
}


void Filter::setup_reshapeOp()
{
  if (input->nDims == 2) // Quads
  {
    reshapeOp.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    
    unsigned int cnt = 0;
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      for (unsigned int i = 0; i < eles->nSpts1D; i++)
        reshapeOp(i,j) = cnt++;
      
    cnt = 0;
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
      for (unsigned int j = eles->nSpts1D; j < 2 * eles->nSpts1D; j++)
        reshapeOp(i,j) = cnt++; 
  } 
  else // Hexes   
  {
    if (input->rank == 0) std::cout << "Sensor on hexes havn't been implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }
}


double Filter::apply_sensor(unsigned int ele, unsigned int var)
{
  // Local arrays 
  mdvector<double> u, u_lines, uh_lines, KS_lines;
  u.assign({eles->nSpts});
  if (input->nDims == 2) // Quads
  {
    u_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    uh_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    KS_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
  }
  else // Hexes   
  {
    if (input->rank == 0) std::cout << "Sensor on hexes havn't been implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }  

  // Copy data to local memory
  for (unsigned int spt = 0; spt < eles->nSpts; spt++)
    u(spt) = eles->U_spts(spt, ele, var);

  // Normalize data 
  if (input->sen_norm)
  {
    double uMax = u.max_val();
    double uMin = u.min_val();
    for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      u(spt) = (u(spt) - uMin) / (uMax - uMin + 1e-10);
  }
  
  // Reshape solution
#pragma omp parallel for collapse(2)
  for (unsigned int j = 0; j < 2 * eles->nSpts1D; j++)
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
      u_lines(i,j) = u(reshapeOp(i,j));

  // Evaluate modal coefficients
  Vander.solve(uh_lines, u_lines);
       
  // Evaluate filtered kernel
  auto &A = Conc(0, 0);
  auto &B = uh_lines(0, 0);
  auto &C = KS_lines(0, 0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2*eles->nSpts1D, eles->nSpts1D, 1.0, &A, eles->nSpts1D, 
    &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2*eles->nSpts1D, eles->nSpts1D, 1.0, &A, eles->nSpts1D, 
    &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
#endif
    
  // Apply non-linear enhancement
#pragma omp parallel for collapse(2)    
  for (unsigned int j = 0; j < 2 * eles->nSpts1D; j++)
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
      KS_lines(i, j) = order * (KS_lines(i, j) * KS_lines(i, j));
    
  // Store sensor values
  return KS_lines.max_val(); 
}


void Filter::apply_sensor()
{
  //Initialize sensor
  sensor.fill(0.0); 
  
#pragma omp parallel for 
  // Loop over elements
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    double sen = 0;
#pragma omp parallel for reduction(max : sen)
    // Loop over conservative variables
    for (unsigned int var = 0; var < eles->nVars; var++)
      sen = std::max(sen, apply_sensor(ele, var));
    sensor(ele) = sen;
  }    
}
    
  
void Filter::setup_DeltaHat()
{ 
  // Determine filter width in parent space
  if (input->dt_type) // CFL based time step
    DeltaHat = input->filt_gamma * std::sqrt(input->CFL) * 2.0 / std::pow(order + 1.0, 0.25);
  else // Exogenously fixed time step
    DeltaHat = input->filt_gamma * std::sqrt(0.25) * 2.0 / std::pow(order + 1.0, 0.25);
    
  // Check for kernel positivity
  double DeltaHatMax = 2.0 / (order + 1.0);
  if (DeltaHat <= 0  || DeltaHat > DeltaHatMax)
  {
    if (input->rank == 0) std::cout << " \n WARNING: Negative filter kernel! Gamma should be small and positive. \n" << std::endl;
      // exit(EXIT_FAILURE);
  }
}


void Filter::setup_Fop()
{   
  // Assign filter matrix
  Fop.assign({eles->nSpts1D, eles->nSpts1D + 2});

  // Loop over solution points - rows
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    // Internal weights - first P+1 columns
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
    {
      // Set integration limits
      double xiL = std::max(-1.0, eles->loc_spts_1D[spt] - 0.5*DeltaHat);
      double xiR = std::min(1.0, eles->loc_spts_1D[spt] + 0.5*DeltaHat);

      // Evaluate integral through quadrature
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
      {
        double xi = xiL + (xiR - xiL) * (eles->loc_spts_1D[j] + 1.0) / 2.0;
        double fun = 0.5 * (xiR-xiL)* Lagrange(eles->loc_spts_1D, i, xi) / DeltaHat;
        Fop(spt, i) += fun * eles->weights_spts(j);
      }
    }
    
    // Boundary weights
    Fop(spt, eles->nSpts1D) = std::max(0.0, 0.5*DeltaHat -1.0 -eles->loc_spts_1D[spt]) / DeltaHat;
    Fop(spt, eles->nSpts1D + 1) = std::max(0.0, eles->loc_spts_1D[spt] +0.5*DeltaHat -1) / DeltaHat;
  }
  
  // Evaluate error measure for matrix entries
  mdvector<double> unity({eles->nSpts1D});
  double tol = 0;
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    for (unsigned int i = 0; i < eles->nSpts1D + 2; i++)
      unity(spt) += Fop(spt, i);
    tol += std::abs(unity(spt) - 1.0);
  }
  if (input->rank == 0) std::cout << "  L1 norm of tolerance for 1D filter matrix = " << tol << std::endl;
}


void Filter::setup_appendOp()
{
  if (input->nDims == 2) // Quads
  {
    appendOp.assign({2, 2 * eles->nSpts1D});
    unsigned int cnt = 0;
    
    // Bottom edge
    for (unsigned int j = eles->nSpts1D; j < 2 * eles->nSpts1D; j++)
      appendOp(0,j) = cnt++;
    
    // Right edge
    for (unsigned int j = 0; j < eles->nSpts1D; j++)
      appendOp(1,j) = cnt++;
      
    // Top edge
    for (unsigned int j = 2 * eles->nSpts1D - 1; j >= eles->nSpts1D; j--)
      appendOp(1,j) = cnt++;
    
    // Left edge
    for (int j = eles->nSpts1D - 1; j >= 0; j--)
      appendOp(0,j) = cnt++;
  } 
  else // Hexes   
  {
    if (input->rank == 0) std::cout << "Filter on hexes havn't been implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }  
}


void Filter::apply_filter(unsigned int ele, unsigned int var)
{
  // Check for sensor value
  if (sensor(ele) < threshJ) return;
  
  // Local arrays with accommodation for flux points
  mdvector<double> u_lines_spt, u_lines;
  if (input->nDims == 2) // Quads
  {
    u_lines_spt.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    u_lines.assign({eles->nSpts1D + 2, 2 * eles->nSpts1D});
  }
  else // Hexes   
  {
    if (input->rank == 0) std::cout << "Filter on hexes havn't been implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }  

  // Reshape solution and append flux points
#pragma omp parallel for collapse(2)
  for (unsigned int j = 0; j < 2 * eles->nSpts1D; j++)
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
      u_lines(i,j) = eles->U_spts(reshapeOp(i,j), ele, var);

#pragma omp parallel for collapse(2)      
  for (unsigned int j = 0; j < 2 * eles->nSpts1D; j++)
    for (unsigned int i = 0; i < 2; i++)
      u_lines(i + eles->nSpts1D, j) = eles->U_fpts(appendOp(i,j), ele, var);
    
  // Evaluate filtered solution
  auto &A = Fop(0, 0);
  auto &B = u_lines(0, 0);
  auto &C = u_lines_spt(0, 0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2*eles->nSpts1D, (eles->nSpts1D + 2), 1.0, &A, eles->nSpts1D, 
    &B, (eles->nSpts1D + 2), 0.0, &C, eles->nSpts1D);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2*eles->nSpts1D, (eles->nSpts1D + 2), 1.0, &A, eles->nSpts1D, 
    &B, (eles->nSpts1D + 2), 0.0, &C, eles->nSpts1D);
#endif
    
  // Overwrite solution with averaged values
#pragma omp parallel for collapse(2)    
  for (unsigned int j = 0; j < eles->nSpts1D; j++)
    for (unsigned int i = 0; i < eles->nSpts1D; i++)
      eles->U_spts(reshapeOp(i,j), ele, var) = 0.5 * (u_lines_spt(i,j) + u_lines_spt(j, i + eles->nSpts1D));
}


void Filter::apply_filter()
{
  // Check if filtering is required
  if (sensor.max_val() < threshJ) return;
  
  // Transfer information to flux points
  eles->extrapolate_U(); // Extrapolate solution to flux points
  solver->U_to_faces(); // Copy flux point data from element local to face local storage
#ifdef _MPI // Compute common interface solution at flux points
  faces->send_U_data();
  faces->recv_U_data();
  faces->compute_common_U(geo->nGfpts_int + geo->nGfpts_bnd, geo->nGfpts);
#else
  faces->compute_common_U(0, geo->nGfpts); 
#endif 
  solver->U_from_faces(); // Copy solution data at flux points from face local to element local storage  
   
  // Filter each element
#pragma omp parallel for collapse(2)
  // Loop over elements
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    // Loop over conservative variables
    for (unsigned int var = 0; var < eles->nVars; var++)
      apply_filter(ele, var);
  }    
}
