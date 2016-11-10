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
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous,
    unsigned int startEle, unsigned int endEle, bool overset = false, const int* iblank = NULL,
    bool motion = false)
{

  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
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

  if (viscous)
  {
    /* Transform gradient to physical space */
    double inv_jaco_det = 1.0 / jaco_det_spts(spt,ele);
    double inv_jaco[nDims][nDims];

    for (int dim1 = 0; dim1 < nDims; dim1++)
      for (int dim2 = 0; dim2 < nDims; dim2++)
        inv_jaco[dim1][dim2] = inv_jaco_spts(spt, ele, dim1, dim2);

    for (unsigned int var = 0; var < nVars; var++)
    {
      for (int dim1 = 0; dim1 < nDims; dim1++)
      {
        for (int dim2 = 0; dim2 < nDims; dim2++)
        {
          dU[var][dim1] += (tdU[var][dim2] * inv_jaco[dim2][dim1]);
        }

        dU[var][dim1] *= inv_jaco_det;

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
  else if (equation == Burgers)
  {
    compute_Fconv_Burgers<nVars, nDims>(U, F);
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
    for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
    {
      for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
      {
        inv_jaco[dim1][dim2] = inv_jaco_spts(spt, ele, dim1, dim2);
      }
    }

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
    unsigned int startEle, unsigned int endEle, bool overset, int* iblank, bool motion)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_F<1, 2, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<1, 3, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
  }
  else if (equation == Burgers)
  {
    if (nDims == 2)
      compute_F<1, 2, Burgers><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<1, 3, Burgers><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_F<4, 2, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<5, 3, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, startEle, endEle, overset, iblank, motion);
  }

}

template <unsigned int nDims>
__global__
void compute_dFdUconv_spts_AdvDiff(mdvector_gpu<double> dFdU_spts, 
    unsigned int nSpts, unsigned int nEles, mdvector_gpu<double> AdvDiff_A)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dFdU_spts(spt, ele, 0, 0, dim) = AdvDiff_A(dim);
  }
}

void compute_dFdUconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &dFdU_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, mdvector_gpu<double> &AdvDiff_A)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * nEles + threads - 1)/threads;

  if (nDims == 2)
  {
    compute_dFdUconv_spts_AdvDiff<2><<<blocks, threads>>>(dFdU_spts, nSpts, nEles, AdvDiff_A);
  }
  else
  {
    compute_dFdUconv_spts_AdvDiff<3><<<blocks, threads>>>(dFdU_spts, nSpts, nEles, AdvDiff_A);
  }
}

template <unsigned int nDims>
__global__
void compute_dFdUconv_spts_Burgers(mdvector_gpu<double> dFdU_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts, unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dFdU_spts(spt, ele, 0, 0, dim) = U_spts(spt, ele, 0);
  }
}

void compute_dFdUconv_spts_Burgers_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * nEles + threads - 1)/threads;

  if (nDims == 2)
  {
    compute_dFdUconv_spts_Burgers<2><<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles);
  }
  else
  {
    compute_dFdUconv_spts_Burgers<3><<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles);
  }
}

__global__
void compute_dFdUconv_spts_2D_EulerNS(mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> U_spts, 
    unsigned int nSpts, unsigned int nEles, double gam)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Primitive Variables */
  double rho = U_spts(spt, ele, 0);
  double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
  double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
  double e = U_spts(spt, ele, 3);

  /* Set convective dFdU values in the x-direction */
  dFdU_spts(spt, ele, 0, 0, 0) = 0;
  dFdU_spts(spt, ele, 1, 0, 0) = 0.5 * ((gam-3.0) * u*u + (gam-1.0) * v*v);
  dFdU_spts(spt, ele, 2, 0, 0) = -u * v;
  dFdU_spts(spt, ele, 3, 0, 0) = -gam * e * u / rho + (gam-1.0) * u * (u*u + v*v);

  dFdU_spts(spt, ele, 0, 1, 0) = 1;
  dFdU_spts(spt, ele, 1, 1, 0) = (3.0-gam) * u;
  dFdU_spts(spt, ele, 2, 1, 0) = v;
  dFdU_spts(spt, ele, 3, 1, 0) = gam * e / rho + 0.5 * (1.0-gam) * (3.0*u*u + v*v);

  dFdU_spts(spt, ele, 0, 2, 0) = 0;
  dFdU_spts(spt, ele, 1, 2, 0) = (1.0-gam) * v;
  dFdU_spts(spt, ele, 2, 2, 0) = u;
  dFdU_spts(spt, ele, 3, 2, 0) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 3, 0) = 0;
  dFdU_spts(spt, ele, 1, 3, 0) = (gam-1.0);
  dFdU_spts(spt, ele, 2, 3, 0) = 0;
  dFdU_spts(spt, ele, 3, 3, 0) = gam * u;

  /* Set convective dFdU values in the y-direction */
  dFdU_spts(spt, ele, 0, 0, 1) = 0;
  dFdU_spts(spt, ele, 1, 0, 1) = -u * v;
  dFdU_spts(spt, ele, 2, 0, 1) = 0.5 * ((gam-1.0) * u*u + (gam-3.0) * v*v);
  dFdU_spts(spt, ele, 3, 0, 1) = -gam * e * v / rho + (gam-1.0) * v * (u*u + v*v);

  dFdU_spts(spt, ele, 0, 1, 1) = 0;
  dFdU_spts(spt, ele, 1, 1, 1) = v;
  dFdU_spts(spt, ele, 2, 1, 1) = (1.0-gam) * u;
  dFdU_spts(spt, ele, 3, 1, 1) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 2, 1) = 1;
  dFdU_spts(spt, ele, 1, 2, 1) = u;
  dFdU_spts(spt, ele, 2, 2, 1) = (3.0-gam) * v;
  dFdU_spts(spt, ele, 3, 2, 1) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + 3.0*v*v);

  dFdU_spts(spt, ele, 0, 3, 1) = 0;
  dFdU_spts(spt, ele, 1, 3, 1) = 0;
  dFdU_spts(spt, ele, 2, 3, 1) = (gam-1.0);
  dFdU_spts(spt, ele, 3, 3, 1) = gam * v;
}

__global__
void compute_dFdUconv_spts_3D_EulerNS(mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> U_spts, 
    unsigned int nSpts, unsigned int nEles, double gam)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Primitive Variables */
  double rho = U_spts(spt, ele, 0);
  double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
  double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
  double w = U_spts(spt, ele, 3) / U_spts(spt, ele, 0);
  double e = U_spts(spt, ele, 4);

  /* Set convective dFdU values in the x-direction */
  dFdU_spts(spt, ele, 0, 0, 0) = 0;
  dFdU_spts(spt, ele, 1, 0, 0) = 0.5 * ((gam-3.0) * u*u + (gam-1.0) * (v*v + w*w));
  dFdU_spts(spt, ele, 2, 0, 0) = -u * v;
  dFdU_spts(spt, ele, 3, 0, 0) = -u * w;
  dFdU_spts(spt, ele, 4, 0, 0) = -gam * e * u / rho + (gam-1.0) * u * (u*u + v*v + w*w);

  dFdU_spts(spt, ele, 0, 1, 0) = 1;
  dFdU_spts(spt, ele, 1, 1, 0) = (3.0-gam) * u;
  dFdU_spts(spt, ele, 2, 1, 0) = v;
  dFdU_spts(spt, ele, 3, 1, 0) = w;
  dFdU_spts(spt, ele, 4, 1, 0) = gam * e / rho + 0.5 * (1.0-gam) * (3.0*u*u + v*v + w*w);

  dFdU_spts(spt, ele, 0, 2, 0) = 0;
  dFdU_spts(spt, ele, 1, 2, 0) = (1.0-gam) * v;
  dFdU_spts(spt, ele, 2, 2, 0) = u;
  dFdU_spts(spt, ele, 3, 2, 0) = 0;
  dFdU_spts(spt, ele, 4, 2, 0) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 3, 0) = 0;
  dFdU_spts(spt, ele, 1, 3, 0) = (1.0-gam) * w;
  dFdU_spts(spt, ele, 2, 3, 0) = 0;
  dFdU_spts(spt, ele, 3, 3, 0) = u;
  dFdU_spts(spt, ele, 4, 3, 0) = (1.0-gam) * u * w;

  dFdU_spts(spt, ele, 0, 4, 0) = 0;
  dFdU_spts(spt, ele, 1, 4, 0) = (gam-1.0);
  dFdU_spts(spt, ele, 2, 4, 0) = 0;
  dFdU_spts(spt, ele, 3, 4, 0) = 0;
  dFdU_spts(spt, ele, 4, 4, 0) = gam * u;

  /* Set convective dFdU values in the y-direction */
  dFdU_spts(spt, ele, 0, 0, 1) = 0;
  dFdU_spts(spt, ele, 1, 0, 1) = -u * v;
  dFdU_spts(spt, ele, 2, 0, 1) = 0.5 * ((gam-1.0) * (u*u + w*w) + (gam-3.0) * v*v);
  dFdU_spts(spt, ele, 3, 0, 1) = -v * w;
  dFdU_spts(spt, ele, 4, 0, 1) = -gam * e * v / rho + (gam-1.0) * v * (u*u + v*v + w*w);

  dFdU_spts(spt, ele, 0, 1, 1) = 0;
  dFdU_spts(spt, ele, 1, 1, 1) = v;
  dFdU_spts(spt, ele, 2, 1, 1) = (1.0-gam) * u;
  dFdU_spts(spt, ele, 3, 1, 1) = 0;
  dFdU_spts(spt, ele, 4, 1, 1) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 2, 1) = 1;
  dFdU_spts(spt, ele, 1, 2, 1) = u;
  dFdU_spts(spt, ele, 2, 2, 1) = (3.0-gam) * v;
  dFdU_spts(spt, ele, 3, 2, 1) = w;
  dFdU_spts(spt, ele, 4, 2, 1) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + 3.0*v*v + w*w);

  dFdU_spts(spt, ele, 0, 3, 1) = 0;
  dFdU_spts(spt, ele, 1, 3, 1) = 0;
  dFdU_spts(spt, ele, 2, 3, 1) = (1.0-gam) * w;
  dFdU_spts(spt, ele, 3, 3, 1) = v;
  dFdU_spts(spt, ele, 4, 3, 1) = (1.0-gam) * v * w;

  dFdU_spts(spt, ele, 0, 4, 1) = 0;
  dFdU_spts(spt, ele, 1, 4, 1) = 0;
  dFdU_spts(spt, ele, 2, 4, 1) = (gam-1.0);
  dFdU_spts(spt, ele, 3, 4, 1) = 0;
  dFdU_spts(spt, ele, 4, 4, 1) = gam * v;

  /* Set convective dFdU values in the z-direction */
  dFdU_spts(spt, ele, 0, 0, 2) = 0;
  dFdU_spts(spt, ele, 1, 0, 2) = -u * w;
  dFdU_spts(spt, ele, 2, 0, 2) = -v * w;
  dFdU_spts(spt, ele, 3, 0, 2) = 0.5 * ((gam-1.0) * (u*u + v*v) + (gam-3.0) * w*w);
  dFdU_spts(spt, ele, 4, 0, 2) = -gam * e * w / rho + (gam-1.0) * w * (u*u + v*v + w*w);

  dFdU_spts(spt, ele, 0, 1, 2) = 0;
  dFdU_spts(spt, ele, 1, 1, 2) = w;
  dFdU_spts(spt, ele, 2, 1, 2) = 0;
  dFdU_spts(spt, ele, 3, 1, 2) = (1.0-gam) * u;
  dFdU_spts(spt, ele, 4, 1, 2) = (1.0-gam) * u * w;

  dFdU_spts(spt, ele, 0, 2, 2) = 0;
  dFdU_spts(spt, ele, 1, 2, 2) = 0;
  dFdU_spts(spt, ele, 2, 2, 2) = w;
  dFdU_spts(spt, ele, 3, 2, 2) = (1.0-gam) * v;
  dFdU_spts(spt, ele, 4, 2, 2) = (1.0-gam) * v * w;

  dFdU_spts(spt, ele, 0, 3, 2) = 1;
  dFdU_spts(spt, ele, 1, 3, 2) = u;
  dFdU_spts(spt, ele, 2, 3, 2) = v;
  dFdU_spts(spt, ele, 3, 3, 2) = (3.0-gam) * w;
  dFdU_spts(spt, ele, 4, 3, 2) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + v*v + 3.0*w*w);

  dFdU_spts(spt, ele, 0, 4, 2) = 0;
  dFdU_spts(spt, ele, 1, 4, 2) = 0;
  dFdU_spts(spt, ele, 2, 4, 2) = 0;
  dFdU_spts(spt, ele, 3, 4, 2) = (gam-1.0);
  dFdU_spts(spt, ele, 4, 4, 2) = gam * w;
}

void compute_dFdUconv_spts_EulerNS_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * nEles + threads - 1)/threads;

  if (nDims == 2)
  {
    compute_dFdUconv_spts_2D_EulerNS<<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles, gamma);
  }
  else
  {
    compute_dFdUconv_spts_3D_EulerNS<<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles, gamma);
  }
}

__global__
void add_scaled_oppD(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD, 
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims, unsigned int startEle, unsigned int endEle)
{
  const unsigned int tidx = (blockIdx.x * blockDim.x  + threadIdx.x);

  for (unsigned int l = (blockIdx.y * blockDim.y  + threadIdx.y); l < (endEle - startEle) * nVars; l += gridDim.y * blockDim.y)
  {
    unsigned int ele = l / nVars + startEle;
    unsigned int idx = l / nVars;
    unsigned int nj = l % nVars;

    for (unsigned int k = tidx; k < nSpts * nVars; k += blockDim.x)
    {
      unsigned int i = k % nSpts;
      unsigned int ni = k / nSpts;

      for (unsigned int j = 0; j < nSpts; j++)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          LHS(i, ni, j, nj, idx) += oppD(i, j, dim) * C(j, ele, ni, nj, dim);
        }
      }
    }

    __syncthreads(); /* To avoid divergence */
  }

}

void add_scaled_oppD_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims, unsigned int startEle, unsigned int endEle)
{
  dim3 threads(36, 6);

  const unsigned int blocksX = 1;
  const unsigned int blocksY = std::min((nVars * (endEle - startEle) + threads.y - 1) / threads.y, (unsigned int) MAX_GRID_DIM);
  dim3 blocks(blocksX, blocksY);

  add_scaled_oppD<<<blocks, threads>>>(LHS, oppD, C, nSpts, nVars, nEles, nDims, startEle, endEle);
}

__global__
void add_scaled_oppDiv(mdvector_gpu<double> LHS_tempSF, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles)
{
  const unsigned int tidx = (blockIdx.x * blockDim.x  + threadIdx.x);

  for (unsigned int l = (blockIdx.y * blockDim.y  + threadIdx.y); l < nEles * nVars; l += gridDim.y * blockDim.y)
  {
    unsigned int ele = l / nVars;
    unsigned int nj = l % nVars;

    for (unsigned int k = tidx; k < nSpts * nVars; k += blockDim.x)
    {
      unsigned int i = k % nSpts;
      unsigned int ni = k / nSpts;

      for (unsigned int j = 0; j < nFpts; j++)
      {
        LHS_tempSF(i, ni, j, nj, ele) = oppDiv_fpts(i, j) * C(j, ele, ni, nj, 0);
      }
    }

    __syncthreads(); /* To avoid divergence */
  }

}

void add_scaled_oppDiv_wrapper(mdvector_gpu<double> &LHS_tempSF, mdvector_gpu<double> &oppDiv_fpts, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles)
{
  dim3 threads(36, 6);

  const unsigned int blocksX = 1;
  const unsigned int blocksY = std::min((nVars * nEles + threads.y - 1) / threads.y, (unsigned int) MAX_GRID_DIM);
  dim3 blocks(blocksX, blocksY);

  add_scaled_oppDiv<<<blocks, threads>>>(LHS_tempSF, oppDiv_fpts, C, nSpts, nFpts, nVars, nEles);
}

__global__
void add_scaled_oppDiv_times_oppE(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, mdvector_gpu<double> oppE,
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles, unsigned int startEle, unsigned int endEle)
{
  const unsigned int tidx = (blockIdx.x * blockDim.x  + threadIdx.x);

  for (unsigned int p = (blockIdx.y * blockDim.y  + threadIdx.y); p < (endEle - startEle) * nVars; p += gridDim.y * blockDim.y)
  {
    unsigned int ele = p / nVars + startEle;
    unsigned int idx = p / nVars;
    unsigned int nj = p % nVars;

    for (unsigned int q = tidx; q < nSpts * nVars; q += blockDim.x)
    {
      unsigned int i = q % nSpts;
      unsigned int ni = q / nSpts;

      for (unsigned int j = 0; j < nSpts; j++)
      {
        double sum = 0.0;
        for (unsigned int k = 0; k < nFpts; k++)
        {
          sum += oppDiv_fpts(i, k) * C(k, ele, ni, nj, 0) * oppE(k, j);
        }

        LHS(i, ni, j, nj, idx) = sum;

      }
    }

    __syncthreads(); /* To avoid divergence */
  }

}

void add_scaled_oppDiv_times_oppE_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, mdvector_gpu<double> oppE,
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles, unsigned int startEle, unsigned int endEle)
{
  dim3 threads(36, 6);

  const unsigned int blocksX = 1;
  const unsigned int blocksY = std::min((nVars * (endEle - startEle) + threads.y - 1) / threads.y, (unsigned int) MAX_GRID_DIM);
  dim3 blocks(blocksX, blocksY);

  add_scaled_oppDiv_times_oppE<<<blocks, threads>>>(LHS, oppDiv_fpts, oppE, C, nSpts, nFpts, nVars, nEles, startEle, endEle);
}

__global__
void finalize_LHS(mdvector_gpu<double> LHS, mdvector_gpu<double> dt, 
    mdvector_gpu<double> jaco_det_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int dt_type, unsigned int startEle, unsigned int endEle)
{
  const unsigned int tidx = (blockIdx.x * blockDim.x  + threadIdx.x);

  for (unsigned int l = (blockIdx.y * blockDim.y  + threadIdx.y); l < (endEle - startEle) * nVars; l += gridDim.y * blockDim.y)
  {
    unsigned int ele = l / nVars + startEle;
    unsigned int idx = l / nVars;
    unsigned int nj = l % nVars;

    double dt_ = dt(ele);

    if (dt_type != 2)
      dt_ = dt(0);

    for (unsigned int k = tidx; k < nSpts * nVars; k += blockDim.x)
    {
      unsigned int i = k % nSpts;
      unsigned int ni = k / nSpts;

      double jaco_det = jaco_det_spts(i, ele);

      for (unsigned int j = 0; j < nSpts; j++)
      {
        LHS(i, ni, j, nj, idx) = dt_ * LHS(i, ni, j, nj, idx) / jaco_det_spts(i, ele) + (double) (i == j && nj == ni);
      }
    }

    __syncthreads(); /* To avoid divergence */
  }

}

void finalize_LHS_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int dt_type, unsigned int startEle, unsigned int endEle)
{
  dim3 threads(36, 6);

  const unsigned int blocksX = 1;
  const unsigned int blocksY = std::min((nVars * (endEle - startEle) + threads.y - 1) / threads.y, (unsigned int) MAX_GRID_DIM);
  dim3 blocks(blocksX, blocksY);

  finalize_LHS<<<blocks, threads>>>(LHS, dt, jaco_det_spts, nSpts, nVars, nEles, dt_type, startEle, endEle);
}

template <unsigned int nVars>
__global__
void transform_dFdU_quad(mdvector_gpu<double> dFdU_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get metric terms */
  double jaco[2][2];
  jaco[0][0] = jaco_spts(spt, ele, 0, 0);
  jaco[0][1] = jaco_spts(spt, ele, 0, 1);
  jaco[1][0] = jaco_spts(spt, ele, 1, 0);
  jaco[1][1] = jaco_spts(spt, ele, 1, 1);

  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      double dFdUtemp = dFdU_spts(spt, ele, ni, nj, 0);

      dFdU_spts(spt, ele, ni, nj, 0) = dFdU_spts(spt, ele, ni, nj, 0) * jaco[1][1] -
                                       dFdU_spts(spt, ele, ni, nj, 1) * jaco[0][1];
      dFdU_spts(spt, ele, ni, nj, 1) = dFdU_spts(spt, ele, ni, nj, 1) * jaco[0][0] -
                                       dFdUtemp * jaco[1][0];
    }
  }
}

void transform_dFdU_quad_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_dFdU_quad<1><<<blocks, threads>>>(dFdU_spts, jaco_spts, nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_dFdU_quad<4><<<blocks, threads>>>(dFdU_spts, jaco_spts, nSpts, nEles);
  }
}

template <unsigned int nVars>
__global__
void transform_dFdU_hexa(mdvector_gpu<double> dFdU_spts, 
    mdvector_gpu<double> inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get metric terms */
  double inv_jaco[3][3];
  inv_jaco[0][0] = inv_jaco_spts(spt, ele, 0, 0);
  inv_jaco[0][1] = inv_jaco_spts(spt, ele, 0, 1);
  inv_jaco[0][2] = inv_jaco_spts(spt, ele, 0, 2);
  inv_jaco[1][0] = inv_jaco_spts(spt, ele, 1, 0);
  inv_jaco[1][1] = inv_jaco_spts(spt, ele, 1, 1);
  inv_jaco[1][2] = inv_jaco_spts(spt, ele, 1, 2);
  inv_jaco[2][0] = inv_jaco_spts(spt, ele, 2, 0);
  inv_jaco[2][1] = inv_jaco_spts(spt, ele, 2, 1);
  inv_jaco[2][2] = inv_jaco_spts(spt, ele, 2, 2);

  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      double dFdUtemp0 = dFdU_spts(spt, ele, ni, nj, 0);
      double dFdUtemp1 = dFdU_spts(spt, ele, ni, nj, 1);

      dFdU_spts(spt, ele, ni, nj, 0) = dFdU_spts(spt, ele, ni, nj, 0) * inv_jaco[0][0] +
                                       dFdU_spts(spt, ele, ni, nj, 1) * inv_jaco[0][1] +
                                       dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco[0][2];

      dFdU_spts(spt, ele, ni, nj, 1) = dFdUtemp0 * inv_jaco[1][0] +
                                       dFdU_spts(spt, ele, ni, nj, 1) * inv_jaco[1][1] +
                                       dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco[1][2];  
                                
      dFdU_spts(spt, ele, ni, nj, 2) = dFdUtemp0 * inv_jaco[2][0]+ 
                                       dFdUtemp1 * inv_jaco[2][1] +  
                                       dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco[2][2]; 
    }
  }
}

void transform_dFdU_hexa_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_dFdU_hexa<1><<<blocks, threads>>>(dFdU_spts, inv_jaco_spts, nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_dFdU_hexa<5><<<blocks, threads>>>(dFdU_spts, inv_jaco_spts, nSpts, nEles);
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

  if (equation == AdvDiff || equation == Burgers)
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

  if (equation == AdvDiff || equation == Burgers)
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
    mdvector_gpu<int> fpt2gfpt, mdvector_gpu<int> fpt2slot, unsigned int nFpts,
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

  dFn(fpt,ele,var) -= tempF(fpt,ele) * norm(gfpt,dim,slot) * dA(gfpt);
}

void extrapolate_Fn_wrapper(mdvector_gpu<double> &oppE,
    mdvector_gpu<double> &F_spts, mdvector_gpu<double> &tempF_fpts,
    mdvector_gpu<double> &dFn_fpts, mdvector_gpu<double> &norm,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt,
    mdvector_gpu<int> &fpt2slot, unsigned int nSpts, unsigned int nFpts,
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
      /* Get quadrature weight */
      double weight; 
      if (nDims == 2)
      {
        unsigned int i = spt % (order + 1);
        unsigned int j = spt / (order + 1);
        weight = weights_spts(i) * weights_spts(j);
      }

      if (nDims == 3)
      {
        unsigned int i = spt % (order + 1);
        unsigned int j = (spt / (order + 1)) % (order + 1);
        unsigned int k = spt / ((order + 1) * (order + 1));
        weight = weights_spts(i) * weights_spts(j) * weights_spts(k);
      }

      sum += weight * jaco_det_spts(spt, ele) * U_spts(spt, ele, n);
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
    mdvector_gpu<int> fpt2gfpt, mdvector_gpu<int> fpt2slot, int nFpts, int nEles, int nDims)
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
  for (int dim1 = 0; dim1 < nDims; dim1++)
  {
    norm(gfpt,dim1,slot) = 0.0;
    for (int dim2 = 0; dim2 < nDims; dim2++)
    {
      norm(gfpt,dim1,slot) += inv_jaco(fpt,ele,dim2,dim1) * tnorm(fpt,dim2);
    }

    DA += norm(gfpt,dim1,slot) * norm(gfpt,dim1,slot);
  }

  DA = sqrt(DA);

  for (int dim = 0; dim < nDims; dim++)
  {
    norm(gfpt,dim,slot) /= DA;
  }

  if (slot == 0)
    dA(gfpt) = DA;
}

void calc_normals_wrapper(mdvector_gpu<double> &norm, mdvector_gpu<double> &dA,
    mdvector_gpu<double> &inv_jaco, mdvector_gpu<double> &tnorm,
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2slot, int nFpts,
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
