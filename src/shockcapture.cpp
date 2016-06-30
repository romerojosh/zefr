#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "cblas.h"
#include "funcs.hpp"
#include "quads.hpp"
#include "polynomials.hpp"
#include "shockcapture.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#include "shockcapture_kernels.h"
#endif

using namespace std;

void ShockCapture::setup(InputStruct *input, FRSolver &solver)
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
  setup_expfilter_matrix();
  U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  U_filt.assign({eles->nSpts, eles->nEles, eles->nVars});
  sensor.assign({eles->nEles});
  sensor_bool.assign({eles->nEles});

    /* Copy data to GPU */
#ifdef _GPU
  U_spts_d = U_spts;
  U_filt_d = U_filt;
  oppS_d = oppS;
  KS_d = KS;
  sensor_d = sensor;
  sensor_bool_d = sensor_bool;
  Vander2DInv_tr_d = Vander2DInv_tr;
  Vander2D_tr_d = Vander2D_tr;
  filt_d = filt;
  filt2_d = filt2;
#endif
}


void ShockCapture::setup_vandermonde_matrices()
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

  /* 2D Vandermonde matrix on quads */
  mdvector<double> loc({2});
  Vander2D.assign({eles->nSpts, (order + 1)*(order + 1)});

  #pragma omp parallel for 
  for (int i=0;i<eles->nSpts;i++)
  {
    loc(0) = eles->loc_spts(i,0);
    loc(1) = eles->loc_spts(i,1);

    for (int j=0;j<(order + 1)*(order + 1);j++)
      Vander2D(i,j) = Legendre2D(j,loc,order);
  }
 
  /* Calculate inverse of the 2D Vandermonde matrix */
  Vander2DInv.assign({eles->nSpts, (order + 1)*(order + 1)});
  mdvector<double> eye2D({eles->nSpts, eles->nSpts});
  Vander2D.calc_LU();
  for (unsigned int j = 0; j < eles->nSpts; j++)
    eye2D(j,j) = 1.0;
  Vander2D.solve(Vander2DInv,eye2D);

  /* Get truncated inverse 2D Vandermonde matrix */
  if(order >= 1)
  {
    Vander2D_tr.assign({eles->nSpts, eles->nDims + 1});
    for (uint i = 0; i < eles->nSpts; i++)
    {
      for (uint j = 0; j < eles->nDims + 1; j++)
        Vander2D_tr(i,j) = Vander2D(i,j);
    }
  
    Vander2DInv_tr.assign({eles->nDims + 1, eles->nSpts});
    for (uint i= 0; i < eles->nDims + 1; i++)
    {
      for (uint j = 0; j < eles->nSpts; j++)
        Vander2DInv_tr(i,j) = Vander2DInv(i,j);
    }
  }
  else
  {
    Vander2D_tr = Vander2D;
    Vander2DInv_tr = Vander2DInv;
  }
}


void ShockCapture::setup_concentration_matrix()
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

double ShockCapture::calc_expfilter_coeffs(int in_mode, int type)
{
// Evaluate exponential filter
  double sigma, eta;
  sigma = 1;

  // For shock capturing
  double alpha = input->alpha;
  double s = input->filtexp;

  // For aliasing/Washing away slow high freq. waves
  double alpha2 = 1;
  double s2 = 4;

  int n_dof=(order+1)*(order+1);

  // Hardcoded for 2D, threshold no. of modes below which there is no effect of filter
  double eta_c = 2.0/n_dof;         

  if(in_mode<n_dof)
  {
    int i,j,k;
    int mode;

    mode = 0;
    for (k=0;k<=2*order;k++)
    {
      for (j=0;j<=k;j++)
      {
        i = k-j;
        if(i<=order && j<=order)
        {
          if(mode==in_mode) // found the correct mode
          {
            eta = (double)(i+j)/n_dof;

            if(type == 1)
            {
              sigma = exp(-alpha*pow(eta,s));

              //if(eta <= eta_c)
                //sigma = 1;
              //else
                //sigma = exp(-alpha*pow( (eta - eta_c)/(1 - eta_c), s ));
            }
            else
              sigma = exp(-alpha2*pow(eta,s2));
          }

          mode++;
        }
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating exponential filter ...." << endl;
  }

  return sigma;
}

void ShockCapture::setup_expfilter_matrix()
{
  /* Product of the diagonal sigma matrix and Vandermonde */
  mdvector<double> temp({eles->nSpts, (order+1)*(order+1)});
  mdvector<double> temp2({eles->nSpts, (order+1)*(order+1)});
  filt.assign({eles->nSpts, (order+1)*(order+1)});
  filt2.assign({eles->nSpts, (order+1)*(order+1)});

  double sigma, sigma2;
  for (uint i=0; i < eles->nSpts; i++)
  {
    sigma = calc_expfilter_coeffs(i,1);
    sigma2 = calc_expfilter_coeffs(i,2);
    for (uint j=0; j<(order + 1)*(order + 1); j++)
    {
      temp(i,j) = sigma*Vander2DInv(i,j);
      temp2(i,j) = sigma2*Vander2DInv(i,j);
    }
  }

  /* Writing my own straightforward multiplication routine */
  for (uint i=0; i < eles->nSpts; i++)
    for (uint j=0; j<(order + 1)*(order + 1); j++)
      for(uint k=0; k<(order + 1)*(order + 1); k++)
      {
        filt(i,j) += Vander2D(i,k)*temp(k,j);
        filt2(i,j) += Vander2D(i,k)*temp2(k,j);
      }
         
}

void ShockCapture::setup_threshold()
{
  // Normalization tolerance
  normalTol = 0.1;

  // Non-linear enhancement exponent
  double Q = input->nonlin_exp;
  
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
    eles->nSpts1D, 2, eles->nSpts1D, 1.0, &A, eles->nSpts1D, &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
  
  // Apply non-linear enhancement
  double epsilon = log(order)/order;
  for (unsigned int spt = 0; spt < eles->nSpts1D; spt++)
  {
    KS_step(spt) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS_canon(spt, 0)), Q);
    KS_ramp(spt) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS_canon(spt, 1)), Q);
  }
  
  // Calculate threshold
  threshJ = (1.0 - input->sen_Jfac) *KS_ramp.max_val() + input->sen_Jfac *KS_step.max_val();
    
  // Print results
  if (input->rank == 0) 
  {
    std::cout << " Sensor threshold = " << threshJ << std::endl;
  }
}


void ShockCapture::setup_oppS()
{
  // 1D Sensor matrix
  auto &A = Conc(0, 0);
  auto &B = VanderInv(0, 0);
  auto &C = oppS_1D(0, 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, eles->nSpts1D, eles->nSpts1D, 1.0, &A, eles->nSpts1D, &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
  
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
    if (input->rank == 0) std::cout << "Sensor on hexes hasn't been implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }
}


void ShockCapture::apply_sensor()
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
  for (unsigned int var = 0; var < eles->nVars; var++)
  {
    auto &A = oppS(0,0);
    auto &B = U_spts(0, 0, var);
    auto &C = KS(0, 0, var);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
      2 * eles->nSpts, eles->nEles, eles->nSpts, 1.0, &A, 2 * eles->nSpts, &B, eles->nSpts, 0.0, &C, 2 * eles->nSpts);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      2 * eles->nSpts, eles->nEles, eles->nSpts, 1.0, &A, 2 * eles->nSpts, &B, eles->nSpts, 0.0, &C, 2 * eles->nSpts);
#endif
  }
    
  // Apply non-liqnear enhancement and store sensor values
  double Q = input->nonlin_exp;
  double epsilon = log(order)/order;
  for (unsigned int var = 0; var < eles->nVars; var++)
  {    
    
#pragma omp parallel for
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      double sen = 0.0;
#pragma omp parallel for reduction(max:sen)
      for (unsigned int row = 0; row < 2*eles->nSpts; row++)
      {
        KS(row, ele, var) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS(row, ele, var)), Q);
        sen = std::max(sen, KS(row, ele, var));
      }
      sensor(ele) = std::max(sensor(ele), sen);
      sensor_bool(ele) = sensor(ele) > threshJ;
    }
  }
#endif

#ifdef _GPU
  
  // Copy data to local structure
  device_copy(U_spts_d, eles->U_spts_d, U_spts_d.get_nvals());
  
  // Normalize data
  if (input->sen_norm)
  {
    normalize_data_wrapper(U_spts_d, normalTol, eles->nSpts, eles->nEles, eles->nVars);
  }
  
  // Calculate KS
  for (unsigned int var = 0; var < eles->nVars; var++)
  {
    cublasDGEMM_wrapper(2 * eles->nSpts, eles->nEles, eles->nSpts, 1.0,
      oppS_d.data(), 2 * eles->nSpts, U_spts_d.data() + var * eles->nEles * eles->nSpts, eles->nSpts, 0.0,
      KS_d.data() + var * eles->nEles * 2 * eles->nSpts, 2 * eles->nSpts);
  }
    
  // Apply non-linear enhancement and store sensor values
  compute_max_sensor_wrapper(KS_d, sensor_d, order, max_sensor_d, sensor_bool_d, threshJ, eles->nSpts, eles->nEles, eles->nVars, input->nonlin_exp);

#endif
}

void ShockCapture::compute_Umodal()
{
#ifdef _CPU  
  auto &A = Vander2DInv_tr(0,0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->Umodal(0, 0, 0);

  #ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nDims+1, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nDims+1, &B, eles->nSpts, 0.0, &C, eles->nDims+1);
  #else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nDims+1, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nDims+1, &B, eles->nSpts, 0.0, &C, eles->nDims+1);
  #endif
//cout<<eles->Umodal(0,23,0)<<eles->Umodal(1,23,0)<<eles->Umodal(2,23,0)<<endl;
#endif

#ifdef _GPU
  cublasDGEMM_wrapper(eles->nDims+1, eles->nEles * eles->nVars, eles->nSpts, 1.0, Vander2DInv_tr_d.data(), eles->nDims+1, 
    eles->U_spts_d.data(), eles->nSpts, 0.0, eles->Umodal_d.data(), eles->nDims+1);  
#endif 
}

void ShockCapture::compute_Unodal()
{
#ifdef _CPU  
  auto &A = Vander2D_tr(0,0);
  auto &B = eles->Umodal(0, 0, 0);
  auto &C = U_spts(0, 0, 0);

  #ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nDims+1, 1.0, &A, eles->nSpts, &B, eles->nDims + 1, 0.0, &C, eles->nSpts);
  #else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nDims+1, 1.0, &A, eles->nSpts, &B, eles->nDims + 1, 0.0, &C, eles->nSpts);
  #endif

  // Copy back to eles->U_Spts only when sensor is greater than threshold
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
  cublasDGEMM_wrapper(eles->nSpts, eles->nEles * eles->nVars, eles->nDims + 1, 1.0, Vander2D_tr_d.data(), eles->nSpts, 
    eles->Umodal_d.data(), eles->nDims + 1, 0.0, U_spts_d.data(), eles->nSpts);

  // Copy back to eles->U_Spts only when sensor is greater than threshold
  copy_filtered_solution_wrapper(U_spts_d, eles->U_spts_d, sensor_d, threshJ, eles->nSpts, eles->nEles, eles->nVars, 1);  
#endif 
}

void ShockCapture::apply_expfilter()
{
#ifdef _CPU  
  auto &A = filt(0,0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = U_filt(0, 0, 0);

  #ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->eles->nSpts);
  #else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->eles->nSpts);
  #endif

  // Copy back to eles->U_Spts only when sensor is greater than threshold
  #pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    // Check for sensor value
    if (sensor(ele) < threshJ) continue;

    #pragma omp parallel for collapse(2)    
    for (unsigned int var = 0; var < eles->nVars; var++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        eles->U_spts(spt,ele,var) = U_filt(spt,ele,var);
  }

#endif

#ifdef _GPU
  cublasDGEMM_wrapper(eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, filt_d.data(), eles->nSpts, 
    eles->U_spts_d.data(), eles->nSpts, 0.0, U_filt_d.data(), eles->nSpts); 

  // Copy back to eles->U_Spts only when sensor is greater than threshold
  copy_filtered_solution_wrapper(U_filt_d, eles->U_spts_d, sensor_d, threshJ, eles->nSpts, eles->nEles, eles->nVars, 1);  
#endif   
}

void ShockCapture::apply_expfilter_type2()
{
#ifdef _CPU  
  auto &A = filt2(0,0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = U_filt(0, 0, 0);

  #ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->eles->nSpts);
  #else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles * eles->nVars, 
      eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->eles->nSpts);
  #endif

  // Copy back to eles->U_Spts only when sensor is greater than threshold
  #pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    // Check for sensor value
    if (sensor(ele) > threshJ) continue;

    #pragma omp parallel for collapse(2)    
    for (unsigned int var = 0; var < eles->nVars; var++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        eles->U_spts(spt,ele,var) = U_filt(spt,ele,var);
  }

#endif

#ifdef _GPU
  cublasDGEMM_wrapper(eles->nSpts, eles->nEles * eles->nVars, eles->nSpts, 1.0, filt2_d.data(), eles->nSpts, 
    eles->U_spts_d.data(), eles->nSpts, 0.0, U_filt_d.data(), eles->nSpts); 

  // Copy back to eles->U_Spts only when sensor is greater than threshold
  copy_filtered_solution_wrapper(U_filt_d, eles->U_spts_d, sensor_d, threshJ, eles->nSpts, eles->nEles, eles->nVars, 2);  
#endif   
}

void ShockCapture::bring_to_square(uint ele, uint var, double Ulow, double Uhigh)
{
  assert(Ulow <= 0.0 && Uhigh >= 0.0);
  double Ubound = min(-Ulow, Uhigh);
  double x = eles->Umodal(1, ele, var) + eles->Umodal(2, ele, var);
  double y = eles->Umodal(1, ele, var) - eles->Umodal(2, ele, var);

  double dx = std::max<double>({-Ubound - x, 0, x - Ubound});
  double dy = std::max<double>({-Ubound - y, 0, y - Ubound});
  x = x + dx;
  y = y + dy;
  eles->Umodal(1, ele, var) = 0.5*(x + y);
  eles->Umodal(2, ele, var) = 0.5*(x - y);
}

/* Limiter - currently written for 2D */
void ShockCapture::limiter()
{
#ifdef _CPU
  // Determine whether we need limiting or not
  //if (sensor.max_val() < threshJ) return;
  int nbr;
  double Ucurr, Umin, Umax; // Refers to averages over elements
  for(uint ele = 0; ele < eles->nEles; ele++)
  {
    if(sensor(ele) > threshJ){
      for(uint var = 0; var < eles->nVars; var++)
      {
        Ucurr = 0.5*eles->Umodal(0, ele, var);
        Umin = Ucurr; Umax = Ucurr;
        for(uint face = 0; face < eles->nFaces; face++)
        {
          nbr = geo->ele_adj(face, ele);
          if(nbr != -1)
          {
            Umax = 0.5*eles->Umodal(0, nbr, var) > Umax ? 0.5*eles->Umodal(0, nbr, var) : Umax;
            Umin = 0.5*eles->Umodal(0, nbr, var) < Umin ? 0.5*eles->Umodal(0, nbr, var) : Umin;
          }
        }
        bring_to_square(ele, var, Umin - Ucurr, Umax - Ucurr);       
      }
    }
  }
#endif

#ifdef _GPU
  limiter_wrapper(eles->nEles, eles->nFaces, eles->nVars, threshJ, geo->ele_adj_d, sensor_d, eles->Umodal_d);
#endif
}

// void compute_primitive()
// {
// #ifdef _GPU
//   compute_primitive_wrapper(eles->nSpts, eles->nEles, eles->nVars, eles->U_spts_d, U_prim_d);
// #endif  
// }
