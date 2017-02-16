/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "elements_kernels.h"
#include "flux.hpp"
#include "input.hpp"
#include "mdvector_gpu.h"

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

static const unsigned int MAX_GRID_DIM = 65535;

__device__
double determinant(double* mat, unsigned int M)
{
  double Det = 0;

  switch(M)
  {
    case 0:
      break;

    case 1:
      Det = mat[0];
      break;

    case 2:
      Det = mat[0]*mat[M+1] - mat[1]*mat[M*1];
      break;

    default:
    {
      unsigned int N = M;
      // Use minor-matrix recursion

      int sign = -1;
      double *Minor = new double[(M-1)*(M-1)];

      for (int row = 0; row < M; row++)
      {
        sign *= -1;
        // Setup the minor matrix (expanding along first column)
        int i0 = 0;
        for (int i = 0; i < M; i++)
        {
          if (i == row) continue;
          for (int j = 1; j < N; j++)
          {
            Minor[(M-1)*i0+j-1] = mat[M*i+j];
          }
          i0++;
        }
        // Add in the minor's determinant
        Det += sign*determinant(Minor,M-1)*mat[M*row+0];
      }

      delete[] Minor;

      break;
    }
  }
  return Det;
}

__device__
void adjoint(double *mat, double *adj, int M)
{
  unsigned int N = M;

  int signRow = -1;
  double *Minor = new double[(M-1)*(M-1)];

  for (int row = 0; row < M; row++)
  {
    signRow *= -1;
    int sign = -1*signRow;
    for (int col = 0; col < N; col++)
    {
      sign *= -1;
      // Setup the minor matrix (expanding along row, col)
      int i0 = 0;
      for (int i = 0; i < M; i++)
      {
        if (i == row) continue;
        int j0 = 0;
        for (int j = 0; j < N; j++)
        {
          if (j == col) continue;
          Minor[(M-1)*i0+j0] = mat[M*i+j];
          j0++;
        }
        i0++;
      }
      // Recall: adjoint is TRANSPOSE of cofactor matrix
      adj[M*col+row] = sign*determinant(Minor,M-1);
    }
  }

  delete[] Minor;
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void compute_F(mdvector_gpu<double> F_spts, 
    const mdvector_gpu<double> U_spts, mdvector_gpu<double> dU_spts, const mdvector_gpu<double> inv_jaco_spts,
    const mdvector_gpu<double> jaco_det_spts, unsigned int nSpts, unsigned int nEles, 
    const mdvector_gpu<double> AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous, bool grad_via_div,
    bool overset = false, const int* iblank = NULL, bool motion = false)
{

  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles) 
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double U[nVars];
  double tdU[nVars][nDims];
  double dU[nVars][nDims] = {{0.0}};
  double F[nVars][nDims];
  double inv_jaco[nDims][nDims];

  /* Get state variables and reference space gradients */
  for (unsigned int var = 0; var < nVars; var++)
  {
    U[var] = U_spts(spt, ele, var);

    if(viscous) 
    {
      for(unsigned int dim = 0; dim < nDims; dim++)
      {
        tdU[var][dim] = dU_spts(spt, ele, var, dim);
      }
    }
  }

  /* Get metric terms */
  for (int dim1 = 0; dim1 < nDims; dim1++)
    for (int dim2 = 0; dim2 < nDims; dim2++)
      inv_jaco[dim1][dim2] = inv_jaco_spts(spt, ele, dim1, dim2);

  if (viscous)
  {
    /* Transform gradient to physical space */
    double inv_jaco_det = 1.0 / jaco_det_spts(spt,ele);

    for (unsigned int var = 0; var < nVars; var++)
    {
      for (int dim1 = 0; dim1 < nDims; dim1++)
      {
        if (!grad_via_div)
        {
          for (int dim2 = 0; dim2 < nDims; dim2++)
          {
            dU[var][dim1] += (tdU[var][dim2] * inv_jaco[dim2][dim1]);
          }

          dU[var][dim1] *= inv_jaco_det;

        }
        else
        {
          dU[var][dim1] = tdU[var][dim1] * inv_jaco_det;
        }

        /* Write physical gradient to global memory */
        dU_spts(spt, ele, var, dim1) = dU[var][dim1];

      }
    }
  }

  /* Compute fluxes */
  if (equation == AdvDiff)
  {
    double A[nDims];
    for(unsigned int dim = 0; dim < nDims; dim++)
      A[dim] = AdvDiff_A(dim);

    compute_Fconv_AdvDiff<nVars, nDims>(U, F, A);
    if(viscous) 
      compute_Fvisc_AdvDiff_add<nVars, nDims>(dU, F, AdvDiff_D);

  }
  else if (equation == EulerNS)
  {
    double P;
    compute_Fconv_EulerNS<nVars, nDims>(U, F, P, gamma);
    if(viscous) 
      compute_Fvisc_EulerNS_add<nVars, nDims>(U, dU, F, gamma, prandtl, mu_in,
        rt, c_sth, fix_vis);
  }

  if (!motion)
  {
    /* Transform flux to reference space */
    double tF[nVars][nDims] = {{0.0}};;

    for (unsigned int var = 0; var < nVars; var++)
    {
      for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
      {
        for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
        {
          tF[var][dim1] += F[var][dim2] * inv_jaco[dim1][dim2];
        }
      }
    }

    /* Write out transformed fluxes */
    for (unsigned int var = 0; var < nVars; var++)
    {
      for(unsigned int dim = 0; dim < nDims; dim++)
      {
        F_spts(spt, ele, var, dim) = tF[var][dim];
      }
    }
  }
  else
  {
    /* Write out physical fluxes */
    for (unsigned int var = 0; var < nVars; var++)
    {
      for(unsigned int dim = 0; dim < nDims; dim++)
      {
        F_spts(spt, ele, var, dim) = F[var][dim];
      }
    }
  }

}

void compute_F_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, mdvector_gpu<double> &inv_jaco_spts, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nEles, unsigned int nDims, 
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous,
    bool grad_via_div, bool overset, int* iblank, bool motion)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * nEles + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_F<1, 2, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, grad_via_div, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<1, 3, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, grad_via_div, overset, iblank, motion);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_F<4, 2, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, grad_via_div, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<5, 3, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, grad_via_div, overset, iblank, motion);
  }

}

template<unsigned int nVars, unsigned int nDims>
__global__
void compute_unit_advF(mdvector_gpu<double> F_spts, mdvector_gpu<double> U_spts, mdvector_gpu<double> inv_jaco_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int dim)
{

  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  double U[nVars];
  double inv_jaco[nDims];

  /* Get state variables */
  for (unsigned int var = 0; var < nVars; var++)
  {
    U[var] = U_spts(spt, ele, var);
  }

  /* Get required metric terms */
  for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
  {
      inv_jaco[dim1] = inv_jaco_spts(spt, ele, dim1, dim);
  }

  /* Compute transformed unit advection flux along provided dimension */
  for (unsigned int var = 0; var < nVars; var++)
  {
    for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
    {
        F_spts(spt, ele, var, dim1) = U[var] * inv_jaco[dim1];
    }
  }
}

void compute_unit_advF_wrapper(mdvector_gpu<double>& F_spts, mdvector_gpu<double>& U_spts, mdvector_gpu<double>& inv_jaco_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, unsigned int equation, unsigned int dim)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * nEles + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_unit_advF<1, 2><<<blocks, threads>>>(F_spts, U_spts, inv_jaco_spts, nSpts, nEles, dim);
    else if (nDims == 3)
      compute_unit_advF<1, 3><<<blocks, threads>>>(F_spts, U_spts, inv_jaco_spts, nSpts, nEles, dim);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_unit_advF<4, 2><<<blocks, threads>>>(F_spts, U_spts, inv_jaco_spts, nSpts, nEles, dim);
    else if (nDims == 3)
      compute_unit_advF<5, 3><<<blocks, threads>>>(F_spts, U_spts, inv_jaco_spts, nSpts, nEles, dim);
  }
}

template <unsigned int nVars>
__global__
void transform_gradF_quad(mdvector_gpu<double> divF_spts,
    mdvector_gpu<double> dF_spts, mdvector_gpu<double> jaco_spts,
    mdvector_gpu<double> grid_vel_spts, mdvector_gpu<double> dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    bool overset = false, int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset and iblank[ele] != 1)
    return;

  /* Get metric terms */
  double jaco[2][2];
  jaco[0][0] = jaco_spts(spt, ele, 0, 0);
  jaco[0][1] = jaco_spts(spt, ele, 0, 1);
  jaco[1][0] = jaco_spts(spt, ele, 1, 0);
  jaco[1][1] = jaco_spts(spt, ele, 1, 1);

  double A = grid_vel_spts(spt,ele,1) * jaco[0][1] - grid_vel_spts(spt,ele,0) * jaco[1][1];
  double B = grid_vel_spts(spt,ele,0) * jaco[1][0] - grid_vel_spts(spt,ele,1) * jaco[0][0];

  for (unsigned int n = 0; n < nVars; n++)
  {
    divF_spts(spt,ele,n,stage)  =  dF_spts(spt,ele,n,0,0)*jaco[1][1] - dF_spts(spt,ele,n,1,0)*jaco[0][1] + dU_spts(spt,ele,n,0)*A;
    divF_spts(spt,ele,n,stage) += -dF_spts(spt,ele,n,0,1)*jaco[1][0] + dF_spts(spt,ele,n,1,1)*jaco[0][0] + dU_spts(spt,ele,n,1)*B;
  }
}

void transform_gradF_quad_wrapper(mdvector_gpu<double> &divF_spts,
    mdvector_gpu<double> &dF_spts, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &grid_vel_spts, mdvector_gpu<double> &dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    unsigned int equation, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
    transform_gradF_quad<1><<<blocks, threads>>>(divF_spts, dF_spts, jaco_spts, grid_vel_spts, dU_spts, nSpts, nEles, stage, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    transform_gradF_quad<4><<<blocks, threads>>>(divF_spts, dF_spts, jaco_spts, grid_vel_spts, dU_spts, nSpts, nEles, stage, overset, iblank);
  }
}

template <unsigned int nVars>
__global__
void transform_gradF_hexa(mdvector_gpu<double> divF_spts,
    mdvector_gpu<double> dF_spts, mdvector_gpu<double> jaco_spts,
    mdvector_gpu<double> grid_vel_spts, mdvector_gpu<double> dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    bool overset = false, int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset and iblank[ele] != 1)
    return;

  /* Get metric terms */
  double jaco[16], S[16];
  jaco[4*0+0] = jaco_spts(spt, ele, 0, 0);
  jaco[4*0+1] = jaco_spts(spt, ele, 0, 1);
  jaco[4*0+2] = jaco_spts(spt, ele, 0, 2);
  jaco[4*1+0] = jaco_spts(spt, ele, 1, 0);
  jaco[4*1+1] = jaco_spts(spt, ele, 1, 1);
  jaco[4*1+2] = jaco_spts(spt, ele, 1, 2);
  jaco[4*2+0] = jaco_spts(spt, ele, 2, 0);
  jaco[4*2+1] = jaco_spts(spt, ele, 2, 1);
  jaco[4*2+2] = jaco_spts(spt, ele, 2, 2);
  jaco[4*3+0] = 0.;
  jaco[4*3+1] = 0.;
  jaco[4*3+2] = 0.;

  jaco[4*0+3] = grid_vel_spts(spt, ele, 0);
  jaco[4*1+3] = grid_vel_spts(spt, ele, 1);
  jaco[4*2+3] = grid_vel_spts(spt, ele, 2);
  jaco[4*3+3] = 1;
//  double jaco[4][4], S[4][4];
//  jaco[0][0] = jaco_spts(spt, ele, 0, 0);
//  jaco[0][1] = jaco_spts(spt, ele, 0, 1);
//  jaco[0][2] = jaco_spts(spt, ele, 0, 2);
//  jaco[1][0] = jaco_spts(spt, ele, 1, 0);
//  jaco[1][1] = jaco_spts(spt, ele, 1, 1);
//  jaco[1][2] = jaco_spts(spt, ele, 1, 2);
//  jaco[2][0] = jaco_spts(spt, ele, 2, 0);
//  jaco[2][1] = jaco_spts(spt, ele, 2, 1);
//  jaco[2][2] = jaco_spts(spt, ele, 2, 2);
//  jaco[3][0] = 0.;
//  jaco[3][1] = 0.;
//  jaco[3][2] = 0.;

//  jaco[4*0+3] = grid_vel_spts(spt, ele, 0);
//  jaco[4*1+3] = grid_vel_spts(spt, ele, 1);
//  jaco[4*2+3] = grid_vel_spts(spt, ele, 2);
//  jaco[4*3+3] = 1;

  adjoint(jaco, S, 4);
//  double det = jaco

  for (unsigned int n = 0; n < nVars; n++)
    divF_spts(spt, ele, n, stage) = 0;

  for (unsigned int dim1 = 0; dim1 < 3; dim1++)
  {
    for (unsigned int dim2 = 0; dim2 < 3; dim2++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        divF_spts(spt, ele, n, stage) += dF_spts(spt, ele, n, dim1, dim2) * S[4*dim2+dim1];
      }
    }
  }

  for (unsigned int dim = 0; dim < 3; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      divF_spts(spt, ele, n, stage) += dU_spts(spt, ele, n, dim) * S[4*dim+3];
    }
  }
}

void transform_gradF_hexa_wrapper(mdvector_gpu<double> &divF_spts,
    mdvector_gpu<double> &dF_spts, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &grid_vel_spts, mdvector_gpu<double> &dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    unsigned int equation, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
    transform_gradF_hexa<1><<<blocks, threads>>>(divF_spts, dF_spts, jaco_spts, grid_vel_spts, dU_spts, nSpts, nEles, stage, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    transform_gradF_hexa<5><<<blocks, threads>>>(divF_spts, dF_spts, jaco_spts, grid_vel_spts, dU_spts, nSpts, nEles, stage, overset, iblank);
  }
}

__global__
void normal_flux(mdvector_gpu<double> tempF, mdvector_gpu<double> dFn,
    mdvector_gpu<double> norm, mdvector_gpu<double> dA,
    mdvector_gpu<int> fpt2gfpt, mdvector_gpu<char> fpt2slot, unsigned int nFpts,
    unsigned int nEles, unsigned int dim, unsigned int var)
{
  const unsigned int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (ele >= nEles)
    return;

  int gfpt = fpt2gfpt(fpt,ele);
  int slot = fpt2slot(fpt,ele);

  if (gfpt < 0)
    return;

  double fac = (slot == 1) ? -1 : 1; // factor to negate normal if "right" element (slot = 1)
  dFn(fpt,ele,var) -= tempF(fpt,ele) * fac * norm(gfpt,dim) * dA(gfpt);
}

void extrapolate_Fn_wrapper(mdvector_gpu<double> &oppE,
    mdvector_gpu<double> &F_spts, mdvector_gpu<double> &tempF_fpts,
    mdvector_gpu<double> &dFn_fpts, mdvector_gpu<double> &norm,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt,
    mdvector_gpu<char> &fpt2slot, unsigned int nSpts, unsigned int nFpts,
    unsigned int nEles, unsigned int nDims, unsigned int nVars, bool motion)
{
  int threads = 128;
  int blocks = ((nFpts * nEles) + threads - 1)/ threads;

  int M = nFpts;
  int N = nEles;
  int K = nSpts;

  if (motion)
  {
    for (int dim = 0; dim < nDims; dim++)
    {
      for (int var = 0; var < nVars; var++)
      {
        auto A = oppE.data();
        auto B = F_spts.data() + nSpts * nEles * (var + nVars * dim);
        auto C = tempF_fpts.data();

        cublasDGEMM_wrapper(M, N, K, 1., A, oppE.ldim(), B, F_spts.ldim(), 0.,
                            C, tempF_fpts.ldim(), 0);

        sync_stream(0);
        check_error();

        normal_flux<<<blocks, threads>>>(tempF_fpts, dFn_fpts, norm, dA,
            fpt2gfpt, fpt2slot, nFpts, nEles, dim, var);

        sync_stream(0);
        check_error();
      }
    }
  }
  else
  {
    for (int dim = 0; dim < nDims; dim++)
    {
      for (int var = 0; var < nVars; var++)
      {
        auto A = oppE.data();
        auto B = F_spts.data() + nSpts * nEles * (var + nVars * dim);
        auto C = dFn_fpts.data();

        cublasDGEMM_wrapper(M, N, K, 1., A, oppE.ldim(), B, F_spts.ldim(), -1.,
            C, dFn_fpts.ldim());
      }
    }
  }
}

__global__
void compute_Uavg(mdvector_gpu<double> U_spts, 
    mdvector_gpu<double> Uavg, mdvector_gpu<double> jaco_det_spts, 
    mdvector_gpu<double> weights_spts, mdvector_gpu<double> vol, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, int order)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  /* Compute average solution using quadrature */
  for (unsigned int n = 0; n < nVars; n++)
  {
    double sum = 0.0;

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      sum += weights_spts(spt) * jaco_det_spts(spt, ele) * U_spts(spt, ele, n);
    }

    Uavg(ele, n) = sum / vol(ele); 

  }

}

void compute_Uavg_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &Uavg, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &weights_spts, mdvector_gpu<double> &vol, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, int order)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  compute_Uavg<<<blocks, threads>>>(U_spts, Uavg, jaco_det_spts, weights_spts, vol, nSpts, nEles, nVars, nDims, order);
}

__global__
void poly_squeeze(mdvector_gpu<double> U_spts, 
    mdvector_gpu<double> U_fpts, mdvector_gpu<double> Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  double V[3]; 

  /* For each element, check for negative density at solution and flux points */
  double tol = 1e-10;

  bool negRho = false;
  double minRho = U_spts(0, ele, 0);

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    if (U_spts(spt, ele, 0) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_spts(spt, ele, 0));
    }
  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    if (U_fpts(fpt, ele, 0) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_fpts(fpt, ele, 0));
    }
  }

  /* If negative density found, squeeze density */
  if (negRho)
  {
    double theta = (Uavg(ele, 0) - tol) / (Uavg(ele , 0) - minRho); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, 0) = theta * U_spts(spt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);

    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      U_fpts(fpt, ele, 0) = theta * U_fpts(fpt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);
    
  }

  /* For each element, check for entropy loss */
  double minTau = 1.0;

  /* Get minimum tau value */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double rho = U_spts(spt, ele, 0);
    double momF = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
      momF += U_spts(spt, ele, dim + 1) * U_spts(spt, ele, dim + 1);

    momF /= U_spts(spt, ele, 0);
    double P = (gamma - 1.0) * (U_spts(spt, ele, nDims + 1) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    double rho = U_fpts(fpt, ele, 0);
    double momF = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
      momF += U_fpts(fpt, ele, dim + 1) * U_fpts(fpt, ele, dim + 1);

    momF /= U_fpts(fpt, ele, 0);
    double P = (gamma - 1.0) * (U_fpts(fpt, ele, nDims + 1) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }

  /* If minTau is negative, squeeze solution */
  if (minTau < 0)
  {
    double rho = Uavg(ele, 0);
    double Vsq = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      V[dim] = Uavg(ele, dim + 1) / rho;
      Vsq += V[dim] * V[dim];
    }

    double e = Uavg(ele, nDims + 1);
    double P = (gamma - 1.0) * (e - 0.5 * rho * Vsq);

    double eps = minTau / (minTau - P + exps0 * pow(rho, gamma));

//      if (P < input->exps0 * std::pow(rho, input->gamma))
//        std::cout << "Constraint violated. Lower CFL?" << std::endl;

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        U_spts(spt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_spts(spt, ele, n);
      }
    }

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        U_fpts(fpt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_fpts(fpt, ele, n);
      }
    }
  }

}

void poly_squeeze_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  poly_squeeze<<<blocks, threads>>>(U_spts, U_fpts, Uavg, gamma, exps0, nSpts, nFpts,
      nEles, nVars, nDims);
}

__global__
void copy_coords_ele(mdvector_gpu<double> nodes, mdvector_gpu<double> g_nodes,
    mdvector_gpu<int> ele2node, unsigned int nEles, unsigned int nNodes)
{
  int node = (blockDim.x * blockIdx.x + threadIdx.x) % nNodes;
  int ele =  (blockDim.x * blockIdx.x + threadIdx.x) / nNodes;
  int dim = blockIdx.y;

  if (ele >= nEles)
    return;

  nodes(node, ele, dim) = g_nodes(dim, ele2node(node, ele));
}

__global__
void copy_coords_face(mdvector_gpu<double> coord, mdvector_gpu<double> e_coord,
    mdvector_gpu<int> fpt2gfpt, unsigned int nEles, unsigned int nFpts)
{
  int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  int ele =  (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;
  int dim = blockIdx.y;

  if (ele >= nEles)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  if (gfpt < 0) return;

  coord(gfpt, dim) = e_coord(fpt, ele, dim);
}

void update_coords_wrapper(mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &g_nodes,  mdvector_gpu<double> &shape_spts,
    mdvector_gpu<double> &shape_fpts, mdvector_gpu<double> &coord_spts,
    mdvector_gpu<double> &coord_fpts, mdvector_gpu<double> &coord_faces,
    mdvector_gpu<int> &ele2node, mdvector_gpu<int> &fpt2gfpt, unsigned int nSpts,
    unsigned int nFpts, unsigned int nNodes, unsigned int nEles,
    unsigned int nDims)
{
  int threads = 128;
  dim3 blocksE((nEles * nNodes + threads - 1) / threads, nDims);

  copy_coords_ele<<<blocksE,threads>>>(nodes, g_nodes, ele2node, nEles, nNodes);

  double *B = nodes.data();

  double *As = shape_spts.data();
  double *Cs = coord_spts.data();

  cublasDGEMM_transA_wrapper(nSpts, nEles * nDims, nNodes, 1.0, As,
      shape_spts.ldim(), B, nodes.ldim(), 0.0, Cs, coord_spts.ldim());

  double *Af = shape_fpts.data();
  double *Cf = coord_fpts.data();

  cublasDGEMM_transA_wrapper(nFpts, nEles * nDims, nNodes, 1.0, Af,
      shape_fpts.ldim(), B, nodes.ldim(), 0.0, Cf, coord_fpts.ldim());

  dim3 blocksF((nEles * nFpts + threads - 1) / threads, nDims);

  copy_coords_face<<<blocksF,threads>>>(coord_faces, coord_fpts, fpt2gfpt, nEles, nFpts);
}

template<unsigned int nDims>
__global__
void update_h_ref(mdvector_gpu<double> h_ref, mdvector_gpu<double> coord_fpts,
    unsigned int nEles, unsigned int nFpts, unsigned int nPts1D)
{
  int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (ele >= nEles)
    return;

  if (nDims == 2)
  {
    /* Some indexing to pair up opposing flux points in 2D (on Quad) */
    unsigned int idx = fpt % nPts1D;
    unsigned int fpt1 = fpt;
    unsigned int fpt2 = (fpt / nPts1D + 3) * nPts1D - idx - 1;

    double dx = coord_fpts(fpt1,ele,0) - coord_fpts(fpt2,ele,0);
    double dy = coord_fpts(fpt1,ele,1) - coord_fpts(fpt2,ele,1);
    double dist = std::sqrt(dx*dx + dy*dy);

    h_ref(fpt1, ele) = dist;
    h_ref(fpt2, ele) = dist;
  }
  else
  {
    /// TODO
  }
}

void update_h_ref_wrapper(mdvector_gpu<double> &h_ref,
    mdvector_gpu<double> &coord_fpts, unsigned int nEles, unsigned int nFpts,
    unsigned int nPts1D, unsigned int nDims)
{
  int threads = 128;
  int blocks = (nEles * nFpts + threads - 1) / threads;

  if (nDims == 2)
  {
    update_h_ref<2><<<blocks,threads>>>(h_ref, coord_fpts, nEles, nFpts, nPts1D);
  }
  else
  {
    update_h_ref<3><<<blocks,threads>>>(h_ref, coord_fpts, nEles, nFpts, nPts1D);
  }
}

__global__
void inverse_transform_quad(mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, double *jaco_det, int nEles, int nPts)
{
  int pt = (blockDim.x * blockIdx.x + threadIdx.x) % nPts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nPts;

  if (ele >= nEles)
    return;

  // Determinant of transformation matrix
  if (jaco_det != NULL)
    jaco_det[pt+nPts*ele] = jaco(pt,ele,0,0)*jaco(pt,ele,1,1)-jaco(pt,ele,0,1)*jaco(pt,ele,1,0);

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(pt,ele,0,0) = jaco(pt,ele,1,1);  inv_jaco(pt,ele,0,1) =-jaco(pt,ele,0,1);
  inv_jaco(pt,ele,1,0) =-jaco(pt,ele,1,0);  inv_jaco(pt,ele,1,1) = jaco(pt,ele,0,0);
}

__global__
void inverse_transform_hexa(mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, double* jaco_det, int nEles, int nPts)
{
  int pt = (blockDim.x * blockIdx.x + threadIdx.x) % nPts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nPts;

  if (ele >= nEles)
    return;

  double xr = jaco(pt,ele,0,0);  double xs = jaco(pt,ele,0,1);  double xt = jaco(pt,ele,0,2);
  double yr = jaco(pt,ele,1,0);  double ys = jaco(pt,ele,1,1);  double yt = jaco(pt,ele,1,2);
  double zr = jaco(pt,ele,2,0);  double zs = jaco(pt,ele,2,1);  double zt = jaco(pt,ele,2,2);

  // Determinant of transformation matrix (not always needed)
  if (jaco_det != NULL)
    jaco_det[pt+nPts*ele] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(pt,ele,0,0) = ys*zt - yt*zs;  inv_jaco(pt,ele,0,1) = xt*zs - xs*zt;  inv_jaco(pt,ele,0,2) = xs*yt - xt*ys;
  inv_jaco(pt,ele,1,0) = yt*zr - yr*zt;  inv_jaco(pt,ele,1,1) = xr*zt - xt*zr;  inv_jaco(pt,ele,1,2) = xt*yr - xr*yt;
  inv_jaco(pt,ele,2,0) = yr*zs - ys*zr;  inv_jaco(pt,ele,2,1) = xs*zr - xr*zs;  inv_jaco(pt,ele,2,2) = xr*ys - xs*yr;
}

void calc_transforms_wrapper(mdvector_gpu<double> &nodes, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &jaco_fpts, mdvector_gpu<double> &inv_jaco_spts,
    mdvector_gpu<double> &inv_jaco_fpts, mdvector_gpu<double> &jaco_det_spts,
    mdvector_gpu<double> &dshape_spts, mdvector_gpu<double> &dshape_fpts,
    int nSpts, int nFpts, int nNodes, int nEles, int nDims)
{
  // Calculate forward transform (reference -> physical)
  int ms = nSpts;
  int mf = nFpts;
  int k = nNodes;
  int n = nEles * nDims;

  double* B = nodes.data();

  for (int dim = 0; dim < nDims; dim++)
  {
    double *As = dshape_spts.data() + nNodes * nSpts * dim;
    double *Af = dshape_fpts.data() + nNodes * nFpts * dim;
    double *Cs = jaco_spts.data() + nSpts * nEles * nDims * dim;
    double *Cf = jaco_fpts.data() + nFpts * nEles * nDims * dim;

    cublasDGEMM_transA_wrapper(ms, n, k, 1.0, As, k, B, k, 0.0, Cs, ms);
    cublasDGEMM_transA_wrapper(mf, n, k, 1.0, Af, k, B, k, 0.0, Cf, mf);
  }

  // Calculate inverse transform (physical -> reference) at spts, fpts
  int threads = 128;

  int blocksS = (nSpts * nEles + threads - 1) / threads;
  int blocksF = (nFpts * nEles + threads - 1) / threads;

  if (nDims == 2)
  {
    inverse_transform_quad<<<blocksS,threads>>>(jaco_spts,inv_jaco_spts,
        jaco_det_spts.data(),nEles,nSpts);

    inverse_transform_quad<<<blocksF,threads>>>(jaco_fpts,inv_jaco_fpts,
        NULL,nEles,nFpts);
  }
  else
  {
    inverse_transform_hexa<<<blocksS,threads>>>(jaco_spts,inv_jaco_spts,
        jaco_det_spts.data(),nEles,nSpts);

    inverse_transform_hexa<<<blocksF,threads>>>(jaco_fpts,inv_jaco_fpts,
        NULL,nEles,nFpts);
  }
}

__global__
void calc_normals(mdvector_gpu<double> norm, mdvector_gpu<double> dA,
    mdvector_gpu<double> inv_jaco, mdvector_gpu<double> tnorm,
    mdvector_gpu<int> fpt2gfpt, mdvector_gpu<char> fpt2slot, int nFpts, int nEles, int nDims)
{
  int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (ele >= nEles)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  if (gfpt < 0)
    return;

  int slot = fpt2slot(fpt,ele);

  double DA = 0.0;
  if (slot == 0)
  {
    for (int dim1 = 0; dim1 < nDims; dim1++)
    {
      norm(gfpt,dim1) = 0.0;
      for (int dim2 = 0; dim2 < nDims; dim2++)
      {
        norm(gfpt,dim1) += inv_jaco(fpt,ele,dim2,dim1) * tnorm(fpt,dim2);
      }

      DA += norm(gfpt,dim1) * norm(gfpt,dim1);
    }

    DA = sqrt(DA);

    for (int dim = 0; dim < nDims; dim++)
    {
      norm(gfpt,dim) /= DA;
    }

    dA(gfpt) = DA;
  }
}

void calc_normals_wrapper(mdvector_gpu<double> &norm, mdvector_gpu<double> &dA,
    mdvector_gpu<double> &inv_jaco, mdvector_gpu<double> &tnorm,
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<char> &fpt2slot, int nFpts,
    int nEles, int nDims)
{
  int threads = 128;
  int blocks = (nFpts * nEles + threads - 1) / threads;

  calc_normals<<<blocks,threads>>>(norm,dA,inv_jaco,tnorm,fpt2gfpt,fpt2slot,
      nFpts,nEles,nDims);
}

__global__
void pack_donor_u(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_donors,
    int* donorIDs, int nDonors, unsigned int nSpts, unsigned int nVars)
{
  const unsigned int var = blockIdx.y;
  //const unsigned int spt  = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  //const unsigned int donor= (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;
  const unsigned int spt  = threadIdx.x;
  const unsigned int donor= blockIdx.x;

  if (spt >= nSpts || donor >= nDonors)
    return;

  const unsigned int ele = donorIDs[donor];
  U_donors(spt, donor, var) = U_spts(spt, ele, var);
}

void pack_donor_u_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars)
{
  int threads = nSpts;
  dim3 blocks(nDonors, nVars);

  pack_donor_u<<<blocks, threads>>>(U_spts, U_donors, donorIDs, nDonors, nSpts, nVars);
}

__global__
void pack_donor_grad(mdvector_gpu<double> dU_spts,
    mdvector_gpu<double> dU_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars)
{
  const unsigned int var = blockIdx.y % nVars;
  const unsigned int dim = blockIdx.y / nVars;
  const unsigned int spt   = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int donor = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || donor >= nDonors)
    return;

  const unsigned int ele = donorIDs[donor];
  dU_donors(spt, donor, var, dim) = dU_spts(spt, ele, var, dim);
}

void pack_donor_grad_wrapper(mdvector_gpu<double> &dU_spts,
    mdvector_gpu<double> &dU_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars, unsigned int nDims)
{
  int threads = 128;
  int nblock_x = (nDonors * nSpts + threads - 1) / threads;
  dim3 blocks( nblock_x, nVars*nDims);

  pack_donor_grad<<<blocks, threads>>>(dU_spts, dU_donors, donorIDs, nDonors,
      nSpts, nVars);
}
