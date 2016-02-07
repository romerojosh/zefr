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
  geo = &solver.geo;
  order = solver.order;
  eles = solver.eles;
  faces = solver.faces;
  
	/* Setup sensor */
	if (input->rank == 0) std::cout << "Setting up sensor..." << std::endl;
  setup_vandermonde_matrices();
	setup_concentration_matrix();
  setup_threshold();
  setup_reshapeOp(); 
	sensor.assign({eles->nEles});
  u.assign({eles->nSpts});
	
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
  
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2, eles->nSpts1D, 1.0, &A, eles->nSpts1D, &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
    eles->nSpts1D, 2, eles->nSpts1D, 1.0, &A, eles->nSpts1D, &B, eles->nSpts1D, 0.0, &C, eles->nSpts1D);
#endif
  
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
    std::cout << " Sensor threshold values: " << std::endl;
    std::cout << " Step: " << KS_step.max_val() << " \t Ramp: " << KS_ramp.max_val() << " \t Weighted: " << threshJ << std::endl;
  }
}


void Filter::setup_reshapeOp()
{
  if (input->nDims == 2) // Quads
  {
    reshapeOp.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    u_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    uh_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    KS_lines.assign({eles->nSpts1D, 2 * eles->nSpts1D});
    
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
    if (input->rank == 0) std::cout << "Sensor on hexes not yet implemented." << std::endl;
    exit(EXIT_FAILURE);
  }  
}


void Filter::apply_sensor_ele(unsigned int ele)
{
  // Loop over conservative variables
#pragma omp parallel for 
  for (unsigned int var = 0; var < eles->nVars; var++)
  {
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
    sensor(ele) = std::max(KS_lines.max_val(), sensor(ele));
  }
}


void Filter::apply_sensor()
{
  sensor.fill(0.0); 
#pragma omp parallel for 
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
      apply_sensor_ele(ele);
}
