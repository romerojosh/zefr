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

  /* multi-D Vandermonde matrix on quads */
  mdvector<double> loc({eles->nDims});
  VanderND.assign({eles->nSpts, eles->nSpts});

  #pragma omp parallel for 
  for (int i=0;i<eles->nSpts;i++)
  {
    for(int dim=0; dim<eles->nDims; dim++)
      loc(dim) = eles->loc_spts(i,dim);

    for (int j=0;j<eles->nSpts;j++)
      VanderND(i,j) = LegendreND(j,loc,order,eles->nDims);
  }
 
  /* Calculate inverse of the 2D Vandermonde matrix */
  VanderNDInv.assign({eles->nSpts, eles->nSpts});
  mdvector<double> eyeND({eles->nSpts, eles->nSpts});
  VanderND.calc_LU();
  for (unsigned int j = 0; j < eles->nSpts; j++)
    eyeND(j,j) = 1.0;
  VanderND.solve(VanderNDInv,eyeND);
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

  // For aliasing/Washing away slow high freq. waves for convergence
  double alpha2 = input->alpha2;
  double s2 = input->filtexp2;

  if(eles->nDims == 2)
  {
    int n_dof=(order+1)*(order+1);

    // Threshold no. of modes below which there is no effect of filter
    double eta_c = 2.0/n_dof;         

    if(in_mode >= n_dof)
    {
      cout << "ERROR: Invalid mode when evaluating exponential filter ...." << endl;
      exit(EXIT_FAILURE);
    }

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

  else if(eles->nDims == 3)
  {
    unsigned int n_dof=(order+1)*(order+1)*(order+1);

    // Threshold no. of modes below which there is no effect of filter
    double eta_c = 3.0/n_dof;         

    if(in_mode >= n_dof)
    {
      cout << "ERROR: Invalid mode when evaluating exponential filter ...." << endl;
      exit(EXIT_FAILURE);
    }

      unsigned int i,j,k,l;
      unsigned int mode;
      mode = 0;
      #pragma omp parallel for
      for(l=0;l<=3*order;l++)
      {
        for(k=0;k<=l;k++)
        {
          for(j=0;j<=l-k;j++)
          {
            i = l-k-j;
            if(i<=order && j<=order && k <=order)
            {
              if(mode==in_mode) // found the correct mode
              {
                eta = (double)(i+j+k)/n_dof;

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
    }

    else
    {
      cout << "ERROR: Legendre basis not implemented for higher than 3 dimensions" << endl;
    }    
  return sigma;
}

void ShockCapture::setup_expfilter_matrix()
{
  /* Product of the diagonal sigma matrix and Vandermonde */
  mdvector<double> temp({eles->nSpts, eles->nSpts});
  mdvector<double> temp2({eles->nSpts, eles->nSpts});
  filt.assign({eles->nSpts, eles->nSpts});
  filt2.assign({eles->nSpts, eles->nSpts});

  double sigma, sigma2;
  for (uint i=0; i < eles->nSpts; i++)
  {
    sigma = calc_expfilter_coeffs(i,1);
    sigma2 = calc_expfilter_coeffs(i,2);
    for (uint j=0; j<eles->nSpts; j++)
    {
      temp(i,j) = sigma*VanderNDInv(i,j);
      temp2(i,j) = sigma2*VanderNDInv(i,j);
    }
  }

  /* Writing my own straightforward multiplication routine */
  /* TODO: Use CBLAS */
  for (uint i=0; i < eles->nSpts; i++)
    for (uint j=0; j<eles->nSpts; j++)
      for(uint k=0; k<eles->nSpts; k++)
      {
        filt(i,j) += VanderND(i,k)*temp(k,j);
        filt2(i,j) += VanderND(i,k)*temp2(k,j);
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
  
  oppS.assign({eles->nDims * eles->nSpts, eles->nSpts}, 0.0);
  KS.assign({eles->nDims * eles->nSpts, eles->nEles, eles->nVars});
  // Sensor operator
  if (input->nDims == 2) // Quads
  {    
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
  else if (input->nDims == 3) // Hexes
  {
    int nSpts2D = eles->nSpts1D*eles->nSpts1D;
    
    // xi lines
    for (unsigned int k = 0; k < nSpts2D; k++)
      for (unsigned int j = 0; j < eles->nSpts1D; j++)
        for (unsigned int i = 0; i < eles->nSpts1D; i++)
          oppS(i + k*eles->nSpts1D, j + k*eles->nSpts1D) = oppS_1D(i,j);
      
    // eta lines
    for(unsigned int l = 0; l < eles->nSpts1D; l++)
      for (unsigned int k = 0; k < eles->nSpts1D; k++)
        for (unsigned int j = 0; j < eles->nSpts1D; j++)
          for (unsigned int i = 0; i < eles->nSpts1D; i++)
            oppS(eles->nSpts + i + k*eles->nSpts1D + l*nSpts2D, j*eles->nSpts1D + k + l*nSpts2D) = oppS_1D(i,j);

    // zeta lines
    for(unsigned int l = 0; l < eles->nSpts1D; l++)
      for (unsigned int k = 0; k < eles->nSpts1D; k++)
        for (unsigned int j = 0; j < eles->nSpts1D; j++)
          for (unsigned int i = 0; i < eles->nSpts1D; i++)
            oppS(2*eles->nSpts + i + k*eles->nSpts1D + l*nSpts2D, j*nSpts2D + k + l*eles->nSpts1D) = oppS_1D(i,j);   
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
      eles->nDims * eles->nSpts, eles->nEles, eles->nSpts, 1.0, &A, eles->nDims * eles->nSpts, 
      &B, eles->nSpts, 0.0, &C, eles->nDims * eles->nSpts);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      eles->nDims * eles->nSpts, eles->nEles, eles->nSpts, 1.0, &A, eles->nDims * eles->nSpts, 
      &B, eles->nSpts, 0.0, &C, eles->nDims * eles->nSpts);
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
      for (unsigned int row = 0; row < eles->nDims*eles->nSpts; row++)
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
    cublasDGEMM_wrapper(eles->nDims * eles->nSpts, eles->nEles, eles->nSpts, 1.0,
      oppS_d.data(), eles->nDims * eles->nSpts, U_spts_d.data() + var * eles->nEles * eles->nSpts, eles->nSpts, 0.0,
      KS_d.data() + var * eles->nEles * eles->nDims * eles->nSpts, eles->nDims * eles->nSpts);
  }
    
  // Apply non-linear enhancement and store sensor values
  compute_max_sensor_wrapper(KS_d, sensor_d, order, max_sensor_d, sensor_bool_d, threshJ, eles->nSpts, 
    eles->nEles, eles->nVars, eles->nDims, input->nonlin_exp);

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
