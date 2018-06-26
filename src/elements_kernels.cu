#include "aux_kernels.h"
#include "elements_kernels.h"
#include "flux.hpp"
#include "input.hpp"
#include "mdvector_gpu.h"

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

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
void device_adjoint(double *mat, double *adj, int M)
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

__device__
void device_adjoint_4x4(double *mat, double *adj)
{
  double a11 = mat[0],  a12 = mat[1],  a13 = mat[2],  a14 = mat[3];
  double a21 = mat[4],  a22 = mat[5],  a23 = mat[6],  a24 = mat[7];
  double a31 = mat[8],  a32 = mat[9],  a33 = mat[10], a34 = mat[11];
  double a41 = mat[12], a42 = mat[13], a43 = mat[14], a44 = mat[15];

  adj[0] = -a24*a33*a42 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 + a22*a33*a44;
  adj[1] =  a14*a33*a42 - a13*a34*a42 - a14*a32*a43 + a12*a34*a43 + a13*a32*a44 - a12*a33*a44;
  adj[2] = -a14*a23*a42 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 + a12*a23*a44;
  adj[3] =  a14*a23*a32 - a13*a24*a32 - a14*a22*a33 + a12*a24*a33 + a13*a22*a34 - a12*a23*a34;

  adj[4] =  a24*a33*a41 - a23*a34*a41 - a24*a31*a43 + a21*a34*a43 + a23*a31*a44 - a21*a33*a44;
  adj[5] = -a14*a33*a41 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 + a11*a33*a44;
  adj[6] =  a14*a23*a41 - a13*a24*a41 - a14*a21*a43 + a11*a24*a43 + a13*a21*a44 - a11*a23*a44;
  adj[7] = -a14*a23*a31 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 + a11*a23*a34;

  adj[8] = -a24*a32*a41 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 + a21*a32*a44;
  adj[9] =  a14*a32*a41 - a12*a34*a41 - a14*a31*a42 + a11*a34*a42 + a12*a31*a44 - a11*a32*a44;
  adj[10]= -a14*a22*a41 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 + a11*a22*a44;
  adj[11]=  a14*a22*a31 - a12*a24*a31 - a14*a21*a32 + a11*a24*a32 + a12*a21*a34 - a11*a22*a34;

  adj[12]=  a23*a32*a41 - a22*a33*a41 - a23*a31*a42 + a21*a33*a42 + a22*a31*a43 - a21*a32*a43;
  adj[13]= -a13*a32*a41 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 + a11*a32*a43;
  adj[14]=  a13*a22*a41 - a12*a23*a41 - a13*a21*a42 + a11*a23*a42 + a12*a21*a43 - a11*a22*a43;
  adj[15]= -a13*a22*a31 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33;
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void compute_F(mdvector_gpu<double> F_spts, 
    const mdvector_gpu<double> U_spts, mdvector_gpu<double> dU_spts, const mdvector_gpu<double> Vg_spts,
    const mdvector_gpu<double> inv_jaco_spts, const mdvector_gpu<double> jaco_det_spts, unsigned int nSpts, unsigned int nEles,
    const mdvector_gpu<double> AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous,
    bool overset = false, const int* iblank = NULL, bool motion = false)
{

  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts) 
    return;

  if (overset && iblank[ele] != 1)
      return;

  double U[nVars];
  double tdU[nVars][nDims];
  double dU[nVars][nDims];
  double F[nVars][nDims];
  double inv_jaco[nDims][nDims];
  double tF[nVars][nDims];
  double Vg[nDims] = {0.0};

  /* Get state variables and reference space gradients */
  for (unsigned int var = 0; var < nVars; var++)
    U[var] = U_spts(spt, var, ele);

  if (viscous)
  {
    for(unsigned int dim = 0; dim < nDims; dim++)
      for (unsigned int var = 0; var < nVars; var++)
        tdU[var][dim] = dU_spts(dim, spt, var, ele);
  }

  if (motion)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      Vg[dim] = Vg_spts(spt, dim, ele);
  }

  /* Get metric terms */
  for (int dim1 = 0; dim1 < nDims; dim1++)
    for (int dim2 = 0; dim2 < nDims; dim2++)
      inv_jaco[dim1][dim2] = inv_jaco_spts(dim1, spt, dim2, ele);

  if (viscous)
  {
    /* Transform gradient to physical space */
    double inv_jaco_det = 1.0 / jaco_det_spts(spt,ele);

    for (unsigned int var = 0; var < nVars; var++)
    {
      for (int dim1 = 0; dim1 < nDims; dim1++)
      {

        dU[var][dim1] = (tdU[var][0] * inv_jaco[0][dim1]);
        for (int dim2 = 1; dim2 < nDims; dim2++)
          dU[var][dim1] += (tdU[var][dim2] * inv_jaco[dim2][dim1]);

        dU[var][dim1] *= inv_jaco_det;

        /* Write physical gradient to global memory */
        dU_spts(dim1, spt, var, ele) = dU[var][dim1];

      }
    }
  }

  /* Compute fluxes */
  if (equation == AdvDiff)
  {
    double A[nDims];
    for(unsigned int dim = 0; dim < nDims; dim++)
      A[dim] = AdvDiff_A(dim);

    compute_Fconv_AdvDiff<nVars, nDims>(U, F, A, Vg);
    if(viscous)
      compute_Fvisc_AdvDiff_add<nVars, nDims>(dU, F, AdvDiff_D);

  }
  else if (equation == EulerNS)
  {
    double P;
    compute_Fconv_EulerNS<nVars, nDims>(U, F, Vg, P, gamma);
    if(viscous)
      compute_Fvisc_EulerNS_add<nVars, nDims>(U, dU, F, gamma, prandtl, mu_in,
                                              rt, c_sth, fix_vis);
  }

  /* Transform flux to reference space */
  for (unsigned int var = 0; var < nVars; var++)
  {
    for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
    {
      tF[var][dim1] = F[var][0] * inv_jaco[dim1][0];

      for (unsigned int dim2 = 1; dim2 < nDims; dim2++)
      {
        tF[var][dim1] += F[var][dim2] * inv_jaco[dim1][dim2];
      }
    }
  }

  /* Write out transformed fluxes */
  for(unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      F_spts(dim, spt, var, ele) = tF[var][dim];
    }
  }
}

void compute_F_wrapper(mdvector_gpu<double> &F_spts,
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, mdvector_gpu<double> &grid_vel_spts,
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nDims,
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous, bool overset,
    int* iblank, bool motion)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_F<1, 2, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, grid_vel_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<1, 3, AdvDiff><<<blocks, threads>>>(F_spts, U_spts, dU_spts, grid_vel_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, overset, iblank, motion);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_F<4, 2, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, grid_vel_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, overset, iblank, motion);
    else if (nDims == 3)
      compute_F<5, 3, EulerNS><<<blocks, threads>>>(F_spts, U_spts, dU_spts, grid_vel_spts, inv_jaco_spts, jaco_det_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu_in, c_sth, rt, fix_vis, viscous, overset, iblank, motion);
  }

}

template<unsigned int nDims>
__global__
void compute_KPF_Jac(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD_spts1D, 
    mdvector_gpu<double> oppDivE_spts1D, mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> dFcdU, 
    unsigned int nSpts1D, unsigned int nVars, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int elevarjvari = tidx; elevarjvari < nEles*nVars*nVars; elevarjvari += gridDim.x * blockDim.x)
  {
    const unsigned int ele = elevarjvari / (nVars*nVars);
    const unsigned int varjvari = elevarjvari % (nVars*nVars);
    const unsigned int varj = varjvari / nVars;
    const unsigned int vari = varjvari % nVars;

    /* Note: dFcdU follows idx_fpts */
    // TODO: Construct reverse data structure (i.e. 3D: (fi, fj) -> fpt)
    if (nDims == 2)
    {
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
      {
        const double dFcdUB = dFcdU(fi, vari, varj, ele);
        const double dFcdUR = dFcdU(nSpts1D + fi, vari, varj, ele);
        const double dFcdUT = dFcdU(3*nSpts1D-fi-1, vari, varj, ele);
        const double dFcdUL = dFcdU(4*nSpts1D-fi-1, vari, varj, ele);
        for (unsigned int si = 0; si < nSpts1D; si++)
          for (unsigned int sj = 0; sj < nSpts1D; sj++)
          {
            /* xi-direction */
            unsigned int spti = fi * nSpts1D + si;
            unsigned int sptj = fi * nSpts1D + sj;
            LHS(ele, varj, sptj, vari, spti) += oppD_spts1D(si, sj) * dFdU_spts(0, sptj, vari, varj, ele) +
              oppDivE_spts1D(0, si, sj) * dFcdUL + oppDivE_spts1D(1, si, sj) * dFcdUR;

            /* eta-direction */
            spti = si * nSpts1D + fi;
            sptj = sj * nSpts1D + fi;
            LHS(ele, varj, sptj, vari, spti) += oppD_spts1D(si, sj) * dFdU_spts(1, sptj, vari, varj, ele) +
              oppDivE_spts1D(0, si, sj) * dFcdUB + oppDivE_spts1D(1, si, sj) * dFcdUT;
          }
      }
    }

    else if (nDims == 3)
    {
      const unsigned int nSpts2D = nSpts1D * nSpts1D;
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
        for (unsigned int fj = 0; fj < nSpts1D; fj++)
        {
          const double dFcdUB  = dFcdU(fi*nSpts1D + fj, vari, varj, ele);
          const double dFcdUT  = dFcdU(nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, varj, ele);
          const double dFcdUL  = dFcdU(2*nSpts2D + fi*nSpts1D + fj, vari, varj, ele);
          const double dFcdUR  = dFcdU(3*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, varj, ele);
          const double dFcdUF  = dFcdU(4*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, varj, ele);
          const double dFcdUBk = dFcdU(5*nSpts2D + fi*nSpts1D + fj, vari, varj, ele);

          for (unsigned int si = 0; si < nSpts1D; si++)
            for (unsigned int sj = 0; sj < nSpts1D; sj++)
            {
              /* xi-direction */
              unsigned int spti = fi*nSpts2D + fj*nSpts1D + si;
              unsigned int sptj = fi*nSpts2D + fj*nSpts1D + sj;
              LHS(ele, varj, sptj, vari, spti) += oppD_spts1D(si, sj) * dFdU_spts(0, sptj, vari, varj, ele) +
                oppDivE_spts1D(0, si, sj) * dFcdUL + oppDivE_spts1D(1, si, sj) * dFcdUR;

              /* eta-direction */
              spti = fi*nSpts2D + si*nSpts1D + fj;
              sptj = fi*nSpts2D + sj*nSpts1D + fj;
              LHS(ele, varj, sptj, vari, spti) += oppD_spts1D(si, sj) * dFdU_spts(1, sptj, vari, varj, ele) +
                oppDivE_spts1D(0, si, sj) * dFcdUF + oppDivE_spts1D(1, si, sj) * dFcdUBk;

              /* zeta-direction */
              spti = si*nSpts2D + fi*nSpts1D + fj;
              sptj = sj*nSpts2D + fi*nSpts1D + fj;
              LHS(ele, varj, sptj, vari, spti) += oppD_spts1D(si, sj) * dFdU_spts(2, sptj, vari, varj, ele) +
                oppDivE_spts1D(0, si, sj) * dFcdUB + oppDivE_spts1D(1, si, sj) * dFcdUT;
            }
        }
    }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_KPF_Jac_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD_spts1D, 
    mdvector_gpu<double> oppDivE_spts1D, mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> dFcdU, 
    unsigned int nSpts1D, unsigned int nVars, unsigned int nEles, unsigned int nDims)
{
  dim3 threads(192);
  dim3 blocks(std::min((nEles*nVars*nVars + threads.x - 1) / threads.x, MAX_GRID_DIM));

  if (nDims == 2)
    compute_KPF_Jac<2><<<blocks, threads>>>(LHS, oppD_spts1D, oppDivE_spts1D, dFdU_spts, dFcdU, 
        nSpts1D, nVars, nEles);
  else if (nDims == 3)
    compute_KPF_Jac<3><<<blocks, threads>>>(LHS, oppD_spts1D, oppDivE_spts1D, dFdU_spts, dFcdU, 
        nSpts1D, nVars, nEles);
}

template<unsigned int nDims>
__global__
void compute_KPF_Jac_grad(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD_spts1D, 
    mdvector_gpu<double> oppDivE_spts1D, mdvector_gpu<double> oppDE_spts1D, mdvector_gpu<double> dUcdU, 
    mdvector_gpu<double> dFddU_spts, mdvector_gpu<double> dFcddU, mdvector_gpu<double> inv_jaco_spts, 
    mdvector_gpu<double> jaco_det_spts, unsigned int nSpts1D, unsigned int nVars, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;

  for (unsigned int elevarjvari = tidx; elevarjvari < nEles*nVars*nVars; elevarjvari += gridDim.x * blockDim.x)
  {
    const unsigned int ele = elevarjvari / (nVars*nVars);
    const unsigned int varjvari = elevarjvari % (nVars*nVars);
    const unsigned int varj = varjvari / nVars;
    const unsigned int vari = varjvari % nVars;

    /* Note: dFcddU follows idx_fpts */
    // TODO: Construct reverse data structure (i.e. 3D: (fi, fj) -> fpt)
    if (nDims == 2)
    {
      /* Multiply like terms */
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
        for (unsigned int si = 0; si < nSpts1D; si++)
          for (unsigned int sj = 0; sj < nSpts1D; sj++)
          {
            /* xi-direction */
            double val = 0.0;
            for (unsigned int dimj = 0; dimj < nDims; dimj++)
              for (unsigned int vark = 0; vark < nVars; vark++)
              {
                const double dFcddUR = dFcddU(0, dimj, nSpts1D + fi, vari, vark, ele);
                const double dFcddUL = dFcddU(0, dimj, 4*nSpts1D-fi-1, vari, vark, ele);

                const bool diag = (vark == varj); 
                const double dUcdUR = dUcdU(nSpts1D + fi, vark, varj, ele);
                const double dUcdUL = dUcdU(4*nSpts1D-fi-1, vark, varj, ele);

                for (unsigned int sk = 0; sk < nSpts1D; sk++)
                {
                  const unsigned int sptk = fi * nSpts1D + sk;
                  val += (oppD_spts1D(si, sk) * dFddU_spts(0, dimj, sptk, vari, vark, ele) +
                    oppDivE_spts1D(0, si, sk) * dFcddUL + oppDivE_spts1D(1, si, sk) * dFcddUR) *
                    inv_jaco_spts(0, sptk, dimj, ele) / jaco_det_spts(sptk, ele) *
                    (diag*oppD_spts1D(sk, sj) + oppDE_spts1D(0, sk, sj) * dUcdUL + oppDE_spts1D(1, sk, sj) * dUcdUR);
                }
              }
            unsigned int spti = fi * nSpts1D + si;
            unsigned int sptj = fi * nSpts1D + sj;
            LHS(ele, varj, sptj, vari, spti) += val;

            /* eta-direction */
            val = 0.0;
            for (unsigned int dimj = 0; dimj < nDims; dimj++)
              for (unsigned int vark = 0; vark < nVars; vark++)
              {
                const double dFcddUB = dFcddU(0, dimj, fi, vari, vark, ele);
                const double dFcddUT = dFcddU(0, dimj, 3*nSpts1D-fi-1, vari, vark, ele);

                const bool diag = (vark == varj); 
                const double dUcdUB = dUcdU(fi, vark, varj, ele);
                const double dUcdUT = dUcdU(3*nSpts1D-fi-1, vark, varj, ele);

                for (unsigned int sk = 0; sk < nSpts1D; sk++)
                {
                  const unsigned int sptk = sk * nSpts1D + fi;
                  val += (oppD_spts1D(si, sk) * dFddU_spts(1, dimj, sptk, vari, vark, ele) +
                    oppDivE_spts1D(0, si, sk) * dFcddUB + oppDivE_spts1D(1, si, sk) * dFcddUT) *
                    inv_jaco_spts(1, sptk, dimj, ele) / jaco_det_spts(sptk, ele) *
                    (diag*oppD_spts1D(sk, sj) + oppDE_spts1D(0, sk, sj) * dUcdUB + oppDE_spts1D(1, sk, sj) * dUcdUT);
                }
              }
            spti = si * nSpts1D + fi;
            sptj = sj * nSpts1D + fi;
            LHS(ele, varj, sptj, vari, spti) += val;
          }

      /* Multiply cross terms */
      for (unsigned int etai = 0; etai < nSpts1D; etai++)
        for (unsigned int etaj = 0; etaj < nSpts1D; etaj++)
          for (unsigned int xii = 0; xii < nSpts1D; xii++)
            for (unsigned int xij = 0; xij < nSpts1D; xij++)
            {
              /* Cross term: xi*eta + eta*xi */
              double val = 0.0;
              const unsigned int sptij = etai * nSpts1D + xij;
              const unsigned int sptji = etaj * nSpts1D + xii;
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                for (unsigned int vark = 0; vark < nVars; vark++)
                {
                  const double dFcddUB = dFcddU(0, dimj, xii, vari, vark, ele);
                  const double dFcddUR = dFcddU(0, dimj, nSpts1D + etai, vari, vark, ele);
                  const double dFcddUT = dFcddU(0, dimj, 3*nSpts1D-xii-1, vari, vark, ele);
                  const double dFcddUL = dFcddU(0, dimj, 4*nSpts1D-etai-1, vari, vark, ele);

                  const bool diag = (vark == varj); 
                  const double dUcdUB = dUcdU(xij, vark, varj, ele);
                  const double dUcdUR = dUcdU(nSpts1D + etaj, vark, varj, ele);
                  const double dUcdUT = dUcdU(3*nSpts1D-xij-1, vark, varj, ele);
                  const double dUcdUL = dUcdU(4*nSpts1D-etaj-1, vark, varj, ele);

                  val += 
                    (oppD_spts1D(xii, xij) * dFddU_spts(0, dimj, sptij, vari, vark, ele) +
                    oppDivE_spts1D(0, xii, xij) * dFcddUL + oppDivE_spts1D(1, xii, xij) * dFcddUR) *
                    inv_jaco_spts(1, sptij, dimj, ele) / jaco_det_spts(sptij, ele) *
                    (diag*oppD_spts1D(etai, etaj) + oppDE_spts1D(0, etai, etaj) * dUcdUB + 
                    oppDE_spts1D(1, etai, etaj) * dUcdUT) + 

                    (oppD_spts1D(etai, etaj) * dFddU_spts(1, dimj, sptji, vari, vark, ele) +
                    oppDivE_spts1D(0, etai, etaj) * dFcddUB + oppDivE_spts1D(1, etai, etaj) * dFcddUT) *
                    inv_jaco_spts(0, sptji, dimj, ele) / jaco_det_spts(sptji, ele) *
                    (diag*oppD_spts1D(xii, xij) + oppDE_spts1D(0, xii, xij) * dUcdUL + 
                    oppDE_spts1D(1, xii, xij) * dUcdUR);
                }
              const unsigned int spti = etai * nSpts1D + xii;
              const unsigned int sptj = etaj * nSpts1D + xij;
              LHS(ele, varj, sptj, vari, spti) += val;
            }
    }

    else if (nDims == 3)
    {
      const unsigned int nSpts2D = nSpts1D * nSpts1D;

      /* Multiply like terms */
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
        for (unsigned int fj = 0; fj < nSpts1D; fj++)
          for (unsigned int si = 0; si < nSpts1D; si++)
            for (unsigned int sj = 0; sj < nSpts1D; sj++)
            {
              /* xi-direction */
              double val = 0.0;
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                for (unsigned int vark = 0; vark < nVars; vark++)
                {
                  const double dFcddUL = dFcddU(0, dimj, 2*nSpts2D + fi*nSpts1D + fj, vari, vark, ele);
                  const double dFcddUR = dFcddU(0, dimj, 3*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, vark, ele);

                  const bool diag = (vark == varj); 
                  const double dUcdUL = dUcdU(2*nSpts2D + fi*nSpts1D + fj, vark, varj, ele);
                  const double dUcdUR = dUcdU(3*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vark, varj, ele);

                  for (unsigned int sk = 0; sk < nSpts1D; sk++)
                  {
                    const unsigned int sptk = fi*nSpts2D + fj*nSpts1D + sk;
                    val += (oppD_spts1D(si, sk) * dFddU_spts(0, dimj, sptk, vari, vark, ele) +
                      oppDivE_spts1D(0, si, sk) * dFcddUL + oppDivE_spts1D(1, si, sk) * dFcddUR) *
                      inv_jaco_spts(0, sptk, dimj, ele) / jaco_det_spts(sptk, ele) *
                      (diag*oppD_spts1D(sk, sj) + oppDE_spts1D(0, sk, sj) * dUcdUL + oppDE_spts1D(1, sk, sj) * dUcdUR);
                  }
                }
              unsigned int spti = fi*nSpts2D + fj*nSpts1D + si;
              unsigned int sptj = fi*nSpts2D + fj*nSpts1D + sj;
              LHS(ele, varj, sptj, vari, spti) += val;

              /* eta-direction */
              val = 0.0;
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                for (unsigned int vark = 0; vark < nVars; vark++)
                {
                  const double dFcddUF  = dFcddU(0, dimj, 4*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, vark, ele);
                  const double dFcddUBk = dFcddU(0, dimj, 5*nSpts2D + fi*nSpts1D + fj, vari, vark, ele);

                  const bool diag = (vark == varj); 
                  const double dUcdUF  = dUcdU(4*nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vark, varj, ele);
                  const double dUcdUBk = dUcdU(5*nSpts2D + fi*nSpts1D + fj, vark, varj, ele);

                  for (unsigned int sk = 0; sk < nSpts1D; sk++)
                  {
                    const unsigned int sptk = fi*nSpts2D + sk*nSpts1D + fj;
                    val += (oppD_spts1D(si, sk) * dFddU_spts(1, dimj, sptk, vari, vark, ele) +
                      oppDivE_spts1D(0, si, sk) * dFcddUF + oppDivE_spts1D(1, si, sk) * dFcddUBk) *
                      inv_jaco_spts(1, sptk, dimj, ele) / jaco_det_spts(sptk, ele) *
                      (diag*oppD_spts1D(sk, sj) + oppDE_spts1D(0, sk, sj) * dUcdUF + oppDE_spts1D(1, sk, sj) * dUcdUBk);
                  }
                }
              spti = fi*nSpts2D + si*nSpts1D + fj;
              sptj = fi*nSpts2D + sj*nSpts1D + fj;
              LHS(ele, varj, sptj, vari, spti) += val;

              /* zeta-direction */
              val = 0.0;
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                for (unsigned int vark = 0; vark < nVars; vark++)
                {
                  const double dFcddUB = dFcddU(0, dimj, fi*nSpts1D + fj, vari, vark, ele);
                  const double dFcddUT = dFcddU(0, dimj, nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vari, vark, ele);

                  const bool diag = (vark == varj); 
                  const double dUcdUB = dUcdU(fi*nSpts1D + fj, vark, varj, ele);
                  const double dUcdUT = dUcdU(nSpts2D + fi*nSpts1D + (nSpts1D-fj-1), vark, varj, ele);

                  for (unsigned int sk = 0; sk < nSpts1D; sk++)
                  {
                    const unsigned int sptk = sk*nSpts2D + fi*nSpts1D + fj;
                    val += (oppD_spts1D(si, sk) * dFddU_spts(2, dimj, sptk, vari, vark, ele) +
                      oppDivE_spts1D(0, si, sk) * dFcddUB + oppDivE_spts1D(1, si, sk) * dFcddUT) *
                      inv_jaco_spts(2, sptk, dimj, ele) / jaco_det_spts(sptk, ele) *
                      (diag*oppD_spts1D(sk, sj) + oppDE_spts1D(0, sk, sj) * dUcdUB + oppDE_spts1D(1, sk, sj) * dUcdUT);
                  }
                }
              spti = si*nSpts2D + fi*nSpts1D + fj;
              sptj = sj*nSpts2D + fi*nSpts1D + fj;
              LHS(ele, varj, sptj, vari, spti) += val;
            }

      /* Multiply cross terms */
      for (unsigned int zetak = 0; zetak < nSpts1D; zetak++)
        for (unsigned int etai = 0; etai < nSpts1D; etai++)
          for (unsigned int etaj = 0; etaj < nSpts1D; etaj++)
            for (unsigned int xii = 0; xii < nSpts1D; xii++)
              for (unsigned int xij = 0; xij < nSpts1D; xij++)
              {
                /* Cross term: xi*eta + eta*xi */
                double val = 0.0;
                unsigned int sptij = zetak*nSpts2D + etai*nSpts1D + xij;
                unsigned int sptji = zetak*nSpts2D + etaj*nSpts1D + xii;
                for (unsigned int dimj = 0; dimj < nDims; dimj++)
                  for (unsigned int vark = 0; vark < nVars; vark++)
                  {
                    const double dFcddUL  = dFcddU(0, dimj, 2*nSpts2D + zetak*nSpts1D + etai, vari, vark, ele);
                    const double dFcddUR  = dFcddU(0, dimj, 3*nSpts2D + zetak*nSpts1D + (nSpts1D-etai-1), vari, vark, ele);
                    const double dFcddUF  = dFcddU(0, dimj, 4*nSpts2D + zetak*nSpts1D + (nSpts1D-xii-1), vari, vark, ele);
                    const double dFcddUBk = dFcddU(0, dimj, 5*nSpts2D + zetak*nSpts1D + xii, vari, vark, ele);

                    const bool diag = (vark == varj); 
                    const double dUcdUL  = dUcdU(2*nSpts2D + zetak*nSpts1D + etaj, vark, varj, ele);
                    const double dUcdUR  = dUcdU(3*nSpts2D + zetak*nSpts1D + (nSpts1D-etaj-1), vark, varj, ele);
                    const double dUcdUF  = dUcdU(4*nSpts2D + zetak*nSpts1D + (nSpts1D-xij-1), vark, varj, ele);
                    const double dUcdUBk = dUcdU(5*nSpts2D + zetak*nSpts1D + xij, vark, varj, ele);

                    val += 
                      (oppD_spts1D(xii, xij) * dFddU_spts(0, dimj, sptij, vari, vark, ele) +
                      oppDivE_spts1D(0, xii, xij) * dFcddUL + oppDivE_spts1D(1, xii, xij) * dFcddUR) *
                      inv_jaco_spts(1, sptij, dimj, ele) / jaco_det_spts(sptij, ele) *
                      (diag*oppD_spts1D(etai, etaj) + oppDE_spts1D(0, etai, etaj) * dUcdUF + 
                      oppDE_spts1D(1, etai, etaj) * dUcdUBk) + 

                      (oppD_spts1D(etai, etaj) * dFddU_spts(1, dimj, sptji, vari, vark, ele) +
                      oppDivE_spts1D(0, etai, etaj) * dFcddUF + oppDivE_spts1D(1, etai, etaj) * dFcddUBk) *
                      inv_jaco_spts(0, sptji, dimj, ele) / jaco_det_spts(sptji, ele) *
                      (diag*oppD_spts1D(xii, xij) + oppDE_spts1D(0, xii, xij) * dUcdUL + 
                      oppDE_spts1D(1, xii, xij) * dUcdUR);
                  }
                unsigned int spti = zetak*nSpts2D + etai*nSpts1D + xii;
                unsigned int sptj = zetak*nSpts2D + etaj*nSpts1D + xij;
                LHS(ele, varj, sptj, vari, spti) += val;

                /* Cross term: xi*zeta + zeta*xi */
                val = 0.0;
                sptij = etai*nSpts2D + zetak*nSpts1D + xij;
                sptji = etaj*nSpts2D + zetak*nSpts1D + xii;
                for (unsigned int dimj = 0; dimj < nDims; dimj++)
                  for (unsigned int vark = 0; vark < nVars; vark++)
                  {
                    const double dFcddUB  = dFcddU(0, dimj, zetak*nSpts1D + xii, vari, vark, ele);
                    const double dFcddUT  = dFcddU(0, dimj, nSpts2D + zetak*nSpts1D + (nSpts1D-xii-1), vari, vark, ele);
                    const double dFcddUL  = dFcddU(0, dimj, 2*nSpts2D + etai*nSpts1D + zetak, vari, vark, ele);
                    const double dFcddUR  = dFcddU(0, dimj, 3*nSpts2D + etai*nSpts1D + (nSpts1D-zetak-1), vari, vark, ele);

                    const bool diag = (vark == varj); 
                    const double dUcdUB  = dUcdU(zetak*nSpts1D + xij, vark, varj, ele);
                    const double dUcdUT  = dUcdU(nSpts2D + zetak*nSpts1D + (nSpts1D-xij-1), vark, varj, ele);
                    const double dUcdUL  = dUcdU(2*nSpts2D + etaj*nSpts1D + zetak, vark, varj, ele);
                    const double dUcdUR  = dUcdU(3*nSpts2D + etaj*nSpts1D + (nSpts1D-zetak-1), vark, varj, ele);

                    val += 
                      (oppD_spts1D(xii, xij) * dFddU_spts(0, dimj, sptij, vari, vark, ele) +
                      oppDivE_spts1D(0, xii, xij) * dFcddUL + oppDivE_spts1D(1, xii, xij) * dFcddUR) *
                      inv_jaco_spts(2, sptij, dimj, ele) / jaco_det_spts(sptij, ele) *
                      (diag*oppD_spts1D(etai, etaj) + oppDE_spts1D(0, etai, etaj) * dUcdUB + 
                      oppDE_spts1D(1, etai, etaj) * dUcdUT) + 

                      (oppD_spts1D(etai, etaj) * dFddU_spts(2, dimj, sptji, vari, vark, ele) +
                      oppDivE_spts1D(0, etai, etaj) * dFcddUB + oppDivE_spts1D(1, etai, etaj) * dFcddUT) *
                      inv_jaco_spts(0, sptji, dimj, ele) / jaco_det_spts(sptji, ele) *
                      (diag*oppD_spts1D(xii, xij) + oppDE_spts1D(0, xii, xij) * dUcdUL + 
                      oppDE_spts1D(1, xii, xij) * dUcdUR);
                  }
                spti = etai*nSpts2D + zetak*nSpts1D + xii;
                sptj = etaj*nSpts2D + zetak*nSpts1D + xij;
                LHS(ele, varj, sptj, vari, spti) += val;

                /* Cross term: eta*zeta + zeta*eta */
                val = 0.0;
                sptij = etai*nSpts2D + xij*nSpts1D + zetak;
                sptji = etaj*nSpts2D + xii*nSpts1D + zetak;
                for (unsigned int dimj = 0; dimj < nDims; dimj++)
                  for (unsigned int vark = 0; vark < nVars; vark++)
                  {
                    const double dFcddUB  = dFcddU(0, dimj, xii*nSpts1D + zetak, vari, vark, ele);
                    const double dFcddUT  = dFcddU(0, dimj, nSpts2D + xii*nSpts1D + (nSpts1D-zetak-1), vari, vark, ele);
                    const double dFcddUF  = dFcddU(0, dimj, 4*nSpts2D + etai*nSpts1D + (nSpts1D-zetak-1), vari, vark, ele);
                    const double dFcddUBk = dFcddU(0, dimj, 5*nSpts2D + etai*nSpts1D + zetak, vari, vark, ele);

                    const bool diag = (vark == varj); 
                    const double dUcdUB  = dUcdU(xij*nSpts1D + zetak, vark, varj, ele);
                    const double dUcdUT  = dUcdU(nSpts2D + xij*nSpts1D + (nSpts1D-zetak-1), vark, varj, ele);
                    const double dUcdUF  = dUcdU(4*nSpts2D + etaj*nSpts1D + (nSpts1D-zetak-1), vark, varj, ele);
                    const double dUcdUBk = dUcdU(5*nSpts2D + etaj*nSpts1D + zetak, vark, varj, ele);

                    val += 
                      (oppD_spts1D(xii, xij) * dFddU_spts(1, dimj, sptij, vari, vark, ele) +
                      oppDivE_spts1D(0, xii, xij) * dFcddUF + oppDivE_spts1D(1, xii, xij) * dFcddUBk) *
                      inv_jaco_spts(2, sptij, dimj, ele) / jaco_det_spts(sptij, ele) *
                      (diag*oppD_spts1D(etai, etaj) + oppDE_spts1D(0, etai, etaj) * dUcdUB + 
                      oppDE_spts1D(1, etai, etaj) * dUcdUT) + 

                      (oppD_spts1D(etai, etaj) * dFddU_spts(2, dimj, sptji, vari, vark, ele) +
                      oppDivE_spts1D(0, etai, etaj) * dFcddUB + oppDivE_spts1D(1, etai, etaj) * dFcddUT) *
                      inv_jaco_spts(1, sptji, dimj, ele) / jaco_det_spts(sptji, ele) *
                      (diag*oppD_spts1D(xii, xij) + oppDE_spts1D(0, xii, xij) * dUcdUF + 
                      oppDE_spts1D(1, xii, xij) * dUcdUBk);
                  }
                spti = etai*nSpts2D + xii*nSpts1D + zetak;
                sptj = etaj*nSpts2D + xij*nSpts1D + zetak;
                LHS(ele, varj, sptj, vari, spti) += val;
              }
    }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_KPF_Jac_grad_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD_spts1D, 
    mdvector_gpu<double> &oppDivE_spts1D, mdvector_gpu<double> &oppDE_spts1D, mdvector_gpu<double> &dUcdU, 
    mdvector_gpu<double> &dFddU_spts, mdvector_gpu<double> &dFcddU, mdvector_gpu<double> &inv_jaco_spts, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts1D, unsigned int nVars, unsigned int nEles, 
    unsigned int nDims)
{
  dim3 threads(192);
  dim3 blocks(std::min((nEles*nVars*nVars + threads.x - 1) / threads.x, MAX_GRID_DIM));

  if (nDims == 2)
    compute_KPF_Jac_grad<2><<<blocks, threads>>>(LHS, oppD_spts1D, oppDivE_spts1D, oppDE_spts1D, 
        dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nSpts1D, nVars, nEles);
  else if (nDims == 3)
    compute_KPF_Jac_grad<3><<<blocks, threads>>>(LHS, oppD_spts1D, oppDivE_spts1D, oppDE_spts1D,
        dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nSpts1D, nVars, nEles);
}


__global__
void compute_Jac_spts(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD, 
    mdvector_gpu<double> dFdU_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;
  const unsigned int tidy = blockIdx.y * blockDim.y  + threadIdx.y;

  for (unsigned int elevarj = tidy; elevarj < nEles * nVars; elevarj += gridDim.y * blockDim.y)
  {
    const unsigned int ele = elevarj / nVars;
    const unsigned int varj = elevarj % nVars;

    for (unsigned int varispti = tidx; varispti < nSpts * nVars; varispti += blockDim.x)
    {
      const unsigned int vari = varispti / nSpts;
      const unsigned int spti = varispti % nSpts;

      for (unsigned int sptj = 0; sptj < nSpts; sptj++)
      {
        double val = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          val += oppD(dim, spti, sptj) * dFdU_spts(dim, sptj, vari, varj, ele);
        LHS(ele, varj, sptj, vari, spti) = val;
      }
    }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_Jac_spts_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &dFdU_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims)
{
  dim3 threads(32, 6);
  dim3 blocks(1, std::min((nVars * nEles + threads.y - 1) / threads.y, MAX_GRID_DIM));

  compute_Jac_spts<<<blocks, threads>>>(LHS, oppD, dFdU_spts, nSpts, nVars, nEles, nDims);
}

__global__
void compute_Jac_fpts(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> oppE, mdvector_gpu<double> dFcdU, unsigned int nSpts, unsigned int nFpts, 
    unsigned int nVars, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;
  const unsigned int tidy = blockIdx.y * blockDim.y  + threadIdx.y;

  for (unsigned int elevarj = tidy; elevarj < nEles * nVars; elevarj += gridDim.y * blockDim.y)
  {
    const unsigned int ele = elevarj / nVars;
    const unsigned int varj = elevarj % nVars;

    for (unsigned int varispti = tidx; varispti < nSpts * nVars; varispti += blockDim.x)
    {
      const unsigned int vari = varispti / nSpts;
      const unsigned int spti = varispti % nSpts;

      for (unsigned int sptj = 0; sptj < nSpts; sptj++)
      {
        double val = 0.0;
        for (unsigned int fptk = 0; fptk < nFpts; fptk++)
          val += oppDiv_fpts(spti, fptk) * dFcdU(fptk, vari, varj, ele) * oppE(fptk, sptj);
        LHS(ele, varj, sptj, vari, spti) += val;
      }
    }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_Jac_fpts_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> oppE, mdvector_gpu<double> dFcdU, unsigned int nSpts, unsigned int nFpts, 
    unsigned int nVars, unsigned int nEles)
{
  dim3 threads(32, 6);
  dim3 blocks(1, std::min((nVars * nEles + threads.y - 1) / threads.y, MAX_GRID_DIM));

  compute_Jac_fpts<<<blocks, threads>>>(LHS, oppDiv_fpts, oppE, dFcdU, nSpts, nFpts, nVars, nEles);
}

template<unsigned int nDims, unsigned int nVars, unsigned int nSpts, unsigned int nFpts>
__global__
void compute_Jac_grad(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD, 
    mdvector_gpu<double> oppDiv_fpts, mdvector_gpu<double> oppD_fpts, mdvector_gpu<double> oppE, 
    mdvector_gpu<double> dUcdU, mdvector_gpu<double> dFddU_spts, mdvector_gpu<double> dFcddU, 
    mdvector_gpu<double> inv_jaco_spts, mdvector_gpu<double> jaco_det_spts, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;

  /* Allocate temporary data structures */
  double Cvisc0[nDims][nVars][nSpts];
  double CtempD[nDims];
  double CdFcddU0[nFpts];
  double CdFddU0[nSpts];

  for (unsigned int ele = tidx; ele < nEles; ele += gridDim.x * blockDim.x)
  {
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int sptj = 0; sptj < nSpts; sptj++)
      {
        /* Compute solution gradient Jacobians in reference space */
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int spti = 0; spti < nSpts; spti++)
            {
              double val = 0.0;
              for (unsigned int fptk = 0; fptk < nFpts; fptk++)
                val += oppD_fpts(dim, spti, fptk) * dUcdU(fptk, vari, varj, ele) * oppE(fptk, sptj);
              Cvisc0[dim][vari][spti] = val;
            }

        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int spti = 0; spti < nSpts; spti++)
            Cvisc0[dim][varj][spti] += oppD(dim, spti, sptj);

        /* Transform gradient Jacobians to physical space */
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int spti = 0; spti < nSpts; spti++)
          {
            for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
            {
              double val = 0.0;
              for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
                val += Cvisc0[dim2][vari][spti] * inv_jaco_spts(dim2, spti, dim1, ele);
              CtempD[dim1] = val / jaco_det_spts(spti, ele);
            }

            for (unsigned int dim = 0; dim < nDims; dim++)
              Cvisc0[dim][vari][spti] = CtempD[dim];
          }

        /* Compute viscous Jacobian at flux points (dFcddU only) */
        for (unsigned int vari = 0; vari < nVars; vari++)
        {
          for (unsigned int fpti = 0; fpti < nFpts; fpti++)
          {
            double val = 0.0;
            for (unsigned int dim = 0; dim < nDims; dim++)
              for (unsigned int vark = 0; vark < nVars; vark++)
                for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                  val += dFcddU(0, dim, fpti, vari, vark, ele) * oppE(fpti, sptk) * Cvisc0[dim][vark][sptk];
            CdFcddU0[fpti] = val;
          }

          for (unsigned int spti = 0; spti < nSpts; spti++)
          {
            double val = 0.0;
            for (unsigned int fptk = 0; fptk < nFpts; fptk++)
              val += oppDiv_fpts(spti, fptk) * CdFcddU0[fptk];
            LHS(ele, varj, sptj, vari, spti) += val;
          }
        }

        /* Compute viscous Jacobian at solution points (dFddU only) */
        for (unsigned int dimi = 0; dimi < nDims; dimi++)
          for (unsigned int vari = 0; vari < nVars; vari++)
          {
            for (unsigned int spti = 0; spti < nSpts; spti++)
            {
              double val = 0.0;
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                for (unsigned int vark = 0; vark < nVars; vark++)
                  val += dFddU_spts(dimi, dimj, spti, vari, vark, ele) * Cvisc0[dimj][vark][spti];
              CdFddU0[spti] = val;
            }

            for (unsigned int spti = 0; spti < nSpts; spti++)
            {
              double val = 0.0;
              for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                val += oppD(dimi, spti, sptk) * CdFddU0[sptk];
              LHS(ele, varj, sptj, vari, spti) += val;
            }
          }
      }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_Jac_grad_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &oppDiv_fpts, mdvector_gpu<double> &oppD_fpts, mdvector_gpu<double> &oppE, 
    mdvector_gpu<double> &dUcdU, mdvector_gpu<double> &dFddU_spts, mdvector_gpu<double> &dFcddU, 
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &jaco_det_spts, unsigned int nVars, 
    unsigned int nEles, unsigned int nDims, unsigned int order)
{
  dim3 threads(192, 1);
  dim3 blocks(std::min((nEles + threads.x - 1) / threads.x, MAX_GRID_DIM), 1);

  /* Note: Add kernels here as needed */
  if (nDims == 2)
  {
    if (nVars == 1)
    {
      if (order == 2)
        compute_Jac_grad<2,1,9,12><<<blocks, threads>>>(LHS, oppD, oppDiv_fpts, oppD_fpts, 
            oppE, dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nEles);
      else
        ThrowException("Viscous grad Jacobian kernel not added for this order!");
    }
    else if (nVars == 4)
    {
      if (order == 4)
        compute_Jac_grad<2,4,25,20><<<blocks, threads>>>(LHS, oppD, oppDiv_fpts, oppD_fpts, 
            oppE, dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nEles);
      else
        ThrowException("Viscous grad Jacobian kernel not added for this order!");
    }
  }
  else if (nDims == 3)
  {
    if (nVars == 1)
    {
      if (order == 2)
        compute_Jac_grad<3,1,27,54><<<blocks, threads>>>(LHS, oppD, oppDiv_fpts, oppD_fpts, 
            oppE, dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nEles);
      else
        ThrowException("Viscous grad Jacobian kernel not added for this order!");
    }
    else if (nVars == 5)
    {
      if (order == 4)
        compute_Jac_grad<3,5,125,150><<<blocks, threads>>>(LHS, oppD, oppDiv_fpts, oppD_fpts, 
            oppE, dUcdU, dFddU_spts, dFcddU, inv_jaco_spts, jaco_det_spts, nEles);
      else
        ThrowException("Viscous grad Jacobian kernel not added for this order!");
    }
  }
}

template<unsigned int nDims, unsigned int nVars, unsigned int nSpts, unsigned int nFpts>
__global__
void compute_Jac_gradN(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> oppD_fpts, mdvector_gpu<double> oppE, mdvector_gpu<double> dUcdU, 
    mdvector_gpu<double> dFcddU, mdview_gpu<double> inv_jacoN_spts, mdview_gpu<double> jacoN_det_spts, 
    mdvector_gpu<unsigned int> eleID_in, mdvector_gpu<int> ele2eleN, mdvector_gpu<int> face2faceN, 
    mdvector_gpu<int> fpt2fptN, unsigned int startEle, unsigned int nFptsPerFace, 
    unsigned int nFaces, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;

  /* Allocate temporary data structures */
  double CviscN[nDims][nSpts];
  double CtempD[nDims];
  double CdFcddU0[nVars][nFpts];

  for (unsigned int ele = tidx; ele < nEles; ele += gridDim.x * blockDim.x)
  {
    /* Add center contribution to Neighbor gradient */
    const unsigned int eleID = eleID_in(ele + startEle);

    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int sptj = 0; sptj < nSpts; sptj++)
      {
        for (unsigned int face = 0; face < nFaces; face++)
        {
          /* Note: No contribution if element neighbor is a boundary */
          const int eleNID = ele2eleN(face, eleID);
          if (eleNID == -1) 
          {
            for (unsigned int vari = 0; vari < nVars; vari++)
              for (unsigned int fpti = face * nFptsPerFace; fpti < (face+1) * nFptsPerFace; fpti++)
                CdFcddU0[vari][fpti] = 0.0;
          }
          else
          {
            const unsigned int faceN = face2faceN(face, eleID);

            /* Compute Neighbor gradient Jacobian in reference space */
            for (unsigned int dim = 0; dim < nDims; dim++)
              for (unsigned int spti = 0; spti < nSpts; spti++)
              {
                double val = 0.0;
                for (unsigned int fptk = 0; fptk < nFptsPerFace; fptk++)
                {
                  unsigned int fptN = faceN * nFptsPerFace + fptk;

                  // HACK: fpt2fptN of eleNID doesn't exist for mpi faces so search for fpt manually
                  // Note: Consider creating this connectivity during preprocessing
                  unsigned int fpt = face * nFptsPerFace;
                  while (fpt2fptN(fpt, eleID) != (int)fptN) fpt++;
                  //unsigned int fpt = fpt2fptN(fptN, eleNID);

                  val += oppD_fpts(dim, spti, fptN) * dUcdU(fpt, varj, varj, ele) * oppE(fpt, sptj);
                }
                CviscN[dim][spti] = val;
              }

            /* Transform Neighbor gradient Jacobian to physical space */
            for (unsigned int spti = 0; spti < nSpts; spti++)
            {
              for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
              {
                double val = 0.0;
                for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
                  val += CviscN[dim2][spti] * inv_jacoN_spts(face, dim2, spti, dim1, ele);
                CtempD[dim1] = val / jacoN_det_spts(face, spti, ele);
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
                CviscN[dim][spti] = CtempD[dim];
            }

            /* Compute viscous Jacobian at flux points (dFcddU only) */
            for (unsigned int vari = 0; vari < nVars; vari++)
              for (unsigned int fpti = 0; fpti < nFptsPerFace; fpti++)
              {
                unsigned int fpt = face * nFptsPerFace + fpti;
                int fptN = fpt2fptN(fpt, eleID);

                double val = 0.0;
                for (unsigned int dim = 0; dim < nDims; dim++)
                  for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                    val += dFcddU(1, dim, fpt, vari, varj, ele) * oppE(fptN, sptk) * CviscN[dim][sptk];
                CdFcddU0[vari][fpt] = val;
              }
          }

          __syncthreads(); /* To avoid divergence */
        }

        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int spti = 0; spti < nSpts; spti++)
          {
            double val = 0.0;
            for (unsigned int fptk = 0; fptk < nFpts; fptk++)
              val += oppDiv_fpts(spti, fptk) * CdFcddU0[vari][fptk];
            LHS(ele, varj, sptj, vari, spti) += val;
          }
      }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_Jac_gradN_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppDiv_fpts, 
    mdvector_gpu<double> &oppD_fpts, mdvector_gpu<double> &oppE, mdvector_gpu<double> &dUcdU, 
    mdvector_gpu<double> &dFcddU, mdview_gpu<double> &inv_jacoN_spts, mdview_gpu<double> &jacoN_det_spts, 
    mdvector_gpu<unsigned int> &eleID, mdvector_gpu<int> &ele2eleN, mdvector_gpu<int> &face2faceN, 
    mdvector_gpu<int> &fpt2fptN, unsigned int startEle, unsigned int nFptsPerFace, unsigned int nFaces, 
    unsigned int nVars, unsigned int nEles, unsigned int nDims, unsigned int order)
{
  dim3 threads(192, 1);
  dim3 blocks(std::min((nEles + threads.x - 1) / threads.x, MAX_GRID_DIM), 1);

  /* Note: Add kernels here as needed */
  if (nDims == 2)
  {
    if (nVars == 1)
    {
      if (order == 2)
        compute_Jac_gradN<2,1,9,12><<<blocks, threads>>>(LHS, oppDiv_fpts, oppD_fpts, oppE, 
            dUcdU, dFcddU, inv_jacoN_spts, jacoN_det_spts, eleID, ele2eleN, face2faceN, fpt2fptN,
            startEle, nFptsPerFace, nFaces, nEles);
      else
        ThrowException("Viscous gradN Jacobian kernel not added for this order!");
    }
    else if (nVars == 4)
    {
      if (order == 4)
        compute_Jac_gradN<2,4,25,20><<<blocks, threads>>>(LHS, oppDiv_fpts, oppD_fpts, oppE, 
            dUcdU, dFcddU, inv_jacoN_spts, jacoN_det_spts, eleID, ele2eleN, face2faceN, fpt2fptN,
            startEle, nFptsPerFace, nFaces, nEles);
      else
        ThrowException("Viscous gradN Jacobian kernel not added for this order!");
    }
  }
  else if (nDims == 3)
  {
    if (nVars == 1)
    {
      if (order == 2)
        compute_Jac_gradN<3,1,27,54><<<blocks, threads>>>(LHS, oppDiv_fpts, oppD_fpts, oppE, 
            dUcdU, dFcddU, inv_jacoN_spts, jacoN_det_spts, eleID, ele2eleN, face2faceN, fpt2fptN,
            startEle, nFptsPerFace, nFaces, nEles);
      else
        ThrowException("Viscous gradN Jacobian kernel not added for this order!");
    }
    else if (nVars == 5)
    {
      if (order == 4)
        compute_Jac_gradN<3,5,125,150><<<blocks, threads>>>(LHS, oppDiv_fpts, oppD_fpts, oppE, 
            dUcdU, dFcddU, inv_jacoN_spts, jacoN_det_spts, eleID, ele2eleN, face2faceN, fpt2fptN,
            startEle, nFptsPerFace, nFaces, nEles);
      else
        ThrowException("Viscous gradN Jacobian kernel not added for this order!");
    }
  }
}

__global__
void scale_Jac(mdvector_gpu<double> LHS, mdvector_gpu<double> jaco_det_spts, 
    unsigned int nSpts, unsigned int nVars, unsigned int nEles)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;
  const unsigned int tidy = blockIdx.y * blockDim.y  + threadIdx.y;

  for (unsigned int elevarj = tidy; elevarj < nEles * nVars; elevarj += gridDim.y * blockDim.y)
  {
    const unsigned int ele = elevarj / nVars;
    const unsigned int varj = elevarj % nVars;

    for (unsigned int varispti = tidx; varispti < nSpts * nVars; varispti += blockDim.x)
    {
      const unsigned int vari = varispti / nSpts;
      const unsigned int spti = varispti % nSpts;
      const double jaco_det = jaco_det_spts(spti, ele);

      for (unsigned int sptj = 0; sptj < nSpts; sptj++)
        LHS(ele, varj, sptj, vari, spti) /= jaco_det;
    }

    __syncthreads(); /* To avoid divergence */
  }
}

void scale_Jac_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &jaco_det_spts, 
    unsigned int nSpts, unsigned int nVars, unsigned int nEles)
{
  dim3 threads(32, 6);
  dim3 blocks(1, std::min((nVars * nEles + threads.y - 1) / threads.y, MAX_GRID_DIM));

  scale_Jac<<<blocks, threads>>>(LHS, jaco_det_spts, nSpts, nVars, nEles);
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void compute_dFdU(mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> dFddU_spts,
    const mdvector_gpu<double> U_spts, const mdvector_gpu<double> dU_spts,
    const mdvector_gpu<double> inv_jaco_spts, unsigned int nSpts, unsigned int nEles,
    const mdvector_gpu<double> AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu, bool viscous)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts) 
    return;

  double U[nVars];
  double dU[nVars][nDims];
  double dFdU[nVars][nVars][nDims] = {0};
  double dFddU[nVars][nVars][nDims][nDims] = {0};
  double inv_jaco[nDims][nDims];
  double tdFdU[nVars][nVars][nDims] = {{0.0}};
  double tdFddU[nVars][nVars][nDims][nDims] = {{0.0}};

  /* Get state variables and physical space gradients */
  for (unsigned int var = 0; var < nVars; var++)
  {
    U[var] = U_spts(spt, var, ele);

    if(viscous) 
      for(unsigned int dim = 0; dim < nDims; dim++)
        dU[var][dim] = dU_spts(dim, spt, var, ele);
  }

  /* Compute flux derivatives */
  if (equation == AdvDiff)
  {
    double A[nDims];
    for(unsigned int dim = 0; dim < nDims; dim++)
      A[dim] = AdvDiff_A(dim);

    compute_dFdUconv_AdvDiff<nVars, nDims>(dFdU, A);
    if(viscous) compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddU, AdvDiff_D);
  }
  else if (equation == EulerNS)
  {
    compute_dFdUconv_EulerNS<nVars, nDims>(U, dFdU, gamma);
    if(viscous)
    {
      compute_dFdUvisc_EulerNS_add<nVars, nDims>(U, dU, dFdU, gamma, prandtl, mu);
      compute_dFddUvisc_EulerNS<nVars, nDims>(U, dFddU, gamma, prandtl, mu);
    }
  }

  /* Get metric terms */
  for (int dim1 = 0; dim1 < nDims; dim1++)
    for (int dim2 = 0; dim2 < nDims; dim2++)
      inv_jaco[dim1][dim2] = inv_jaco_spts(dim1, spt, dim2, ele);

  /* Transform flux derivative to reference space */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
        for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
          tdFdU[vari][varj][dim1] += dFdU[vari][varj][dim2] * inv_jaco[dim1][dim2];

  /* Write out transformed flux derivatives */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int dim = 0; dim < nDims; dim++)
        dFdU_spts(dim, spt, vari, varj, ele) = tdFdU[vari][varj][dim];

  if(viscous)
  {
    /* Transform flux derivative to reference space */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
          for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
            for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
              tdFddU[vari][varj][dim1][dimj] += dFddU[vari][varj][dim2][dimj] * inv_jaco[dim1][dim2];

    /* Write out transformed flux derivatives */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dimi = 0; dimi < nDims; dimi++)
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
            dFddU_spts(dimi, dimj, spt, vari, varj, ele) = tdFddU[vari][varj][dimi][dimj];
  }
}

void compute_dFdU_wrapper(mdvector_gpu<double> &dFdU_spts, mdvector_gpu<double> &dFddU_spts,
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts,
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, unsigned int nEles, unsigned int nDims,
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu, bool viscous)
{
  //unsigned int threads = 128;
  //unsigned int blocks = (nEles + threads - 1)/threads;
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_dFdU<1, 2, AdvDiff><<<blocks, threads>>>(dFdU_spts, dFddU_spts, U_spts, dU_spts, inv_jaco_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu, viscous);
    else if (nDims == 3)
      compute_dFdU<1, 3, AdvDiff><<<blocks, threads>>>(dFdU_spts, dFddU_spts, U_spts, dU_spts, inv_jaco_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu, viscous);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_dFdU<4, 2, EulerNS><<<blocks, threads>>>(dFdU_spts, dFddU_spts, U_spts, dU_spts, inv_jaco_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu, viscous);
    else if (nDims == 3)
      compute_dFdU<5, 3, EulerNS><<<blocks, threads>>>(dFdU_spts, dFddU_spts, U_spts, dU_spts, inv_jaco_spts, nSpts, nEles, AdvDiff_A,
          AdvDiff_D, gamma, prandtl, mu, viscous);
  }
}

__global__
void compute_KPF_dFcdU_gradN(mdvector_gpu<double> dFcdU, mdvector_gpu<double> dFcddU,
    mdvector_gpu<double> ddUdUc, mdvector_gpu<double> dUcdU, unsigned int nFpts, 
    unsigned int nVars, unsigned int nEles, unsigned int nDims)
{
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;

  for (unsigned int ele = tidx; ele < nEles; ele += gridDim.x * blockDim.x)
  {
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        {
          double val = 0.0;
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
            val += dFcddU(1, dimj, fpt, vari, varj, ele) * ddUdUc(fpt, dimj, ele) * dUcdU(fpt, varj, varj, ele);
          dFcdU(fpt, vari, varj, ele) += val;
        }

    __syncthreads(); /* To avoid divergence */
  }
}

void compute_KPF_dFcdU_gradN_wrapper(mdvector_gpu<double> &dFcdU, mdvector_gpu<double> &dFcddU,
    mdvector_gpu<double> &ddUdUc, mdvector_gpu<double> &dUcdU, unsigned int nFpts, 
    unsigned int nVars, unsigned int nEles, unsigned int nDims)
{
  dim3 threads(192, 1);
  dim3 blocks(std::min((nEles + threads.x - 1) / threads.x, MAX_GRID_DIM), 1);

  compute_KPF_dFcdU_gradN<<<blocks, threads>>>(dFcdU, dFcddU, ddUdUc, dUcdU, nFpts, nVars, nEles, nDims);
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
      sum += weights_spts(spt) * jaco_det_spts(spt, ele) * U_spts(spt, n, ele);
    }

    Uavg(n, ele) = sum / vol(ele); 

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

  check_error();
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
  double minRho = U_spts(0, 0, ele);

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    if (U_spts(spt, 0, ele) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_spts(spt, 0, ele));
    }
  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    if (U_fpts(fpt, 0, ele) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_fpts(fpt, 0, ele));
    }
  }

  /* If negative density found, squeeze density */
  if (negRho)
  {
    double theta = (Uavg(0, ele) - tol) / (Uavg(0, ele) - minRho); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, 0, ele) = theta * U_spts(spt, 0, ele) + (1.0 - theta) * Uavg(0, ele);

    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      U_fpts(fpt, 0, ele) = theta * U_fpts(fpt, 0, ele) + (1.0 - theta) * Uavg(0, ele);
    
  }

  /* For each element, check for entropy loss */
  double minTau = 1.0;

  /* Get minimum tau value */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double rho = U_spts(spt, 0, ele);
    double momF = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
      momF += U_spts(spt, dim + 1, ele) * U_spts(spt, dim + 1, ele);

    momF /= rho;
    double P = (gamma - 1.0) * (U_spts(spt, nDims + 1, ele) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    double rho = U_fpts(fpt, 0, ele);
    double momF = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
      momF += U_fpts(fpt, dim + 1, ele) * U_fpts(fpt, dim + 1, ele);

    momF /= U_fpts(fpt, 0, ele);
    double P = (gamma - 1.0) * (U_fpts(fpt, nDims + 1, ele) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }

  /* If minTau is negative, squeeze solution */
  if (minTau < 0)
  {
    double rho = Uavg(0, ele);
    double Vsq = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      V[dim] = Uavg(dim + 1, ele) / rho;
      Vsq += V[dim] * V[dim];
    }

    double e = Uavg(nDims + 1, ele);
    double P = (gamma - 1.0) * (e - 0.5 * rho * Vsq);

    double eps = minTau / (minTau - P + exps0 * pow(rho, gamma));

//      if (P < input->exps0 * std::pow(rho, input->gamma))
//        std::cout << "Constraint violated. Lower CFL?" << std::endl;

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        U_spts(spt, n, ele) = eps * Uavg(n, ele) + (1.0 - eps) * U_spts(spt, n, ele);
      }
    }

    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        U_fpts(fpt, n, ele) = eps * Uavg(n, ele) + (1.0 - eps) * U_fpts(fpt, n, ele);
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

  check_error();
}

//! Copy node positions from GeoStruct's array to ele's array
__global__
void copy_coords_ele(mdvector_gpu<double> nodes, mdvector_gpu<double> g_nodes,
    mdvector_gpu<int> ele2node, unsigned int nEles, unsigned int nNodes)
{
  int node = (blockDim.x * blockIdx.x + threadIdx.x) % nNodes;
  int ele =  (blockDim.x * blockIdx.x + threadIdx.x) / nNodes;
  int dim = blockIdx.y;

  if (ele >= nEles)
    return;

  nodes(node, dim, ele) = g_nodes(ele2node(ele, node), dim);
}

//! Copy fpt positions from ele's array to face's array
template<int nDims>
__global__
void copy_coords_face(mdvector_gpu<double> coord, mdvector_gpu<double> e_coord,
    mdvector_gpu<int> fpt2gfpt, unsigned int nEles, unsigned int nFpts)
{
  int fpt = (blockDim.y * blockIdx.y + threadIdx.y);
  int ele =  (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles || fpt >= nFpts)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  if (gfpt < 0) return;

  for (int dim = 0; dim < nDims; dim++)
    coord(dim, gfpt) = e_coord(fpt, dim, ele);
}

void copy_coords_ele_wrapper(mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &g_nodes, mdvector_gpu<int> &ele2node,
    unsigned int nNodes, unsigned int nEles, unsigned int nDims)
{
  int threads = 128;
  dim3 blocksE((nEles * nNodes + threads - 1) / threads, nDims);

  copy_coords_ele<<<blocksE,threads>>>(nodes, g_nodes, ele2node, nEles, nNodes);
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

  int m = nEles * nDims;
  int k = nNodes;
  int ns = nSpts;
  int nf = nFpts;

  double *A = nodes.data();

  double *Bs = shape_spts.data();
  double *Cs = coord_spts.data();

  cublasDGEMM_wrapper(m, ns, k, 1.0, A, nodes.ldim(), Bs, shape_spts.ldim(),
                      0.0, Cs, coord_spts.ldim());

  double *Bf = shape_fpts.data();
  double *Cf = coord_fpts.data();

  cublasDGEMM_wrapper(m, nf, k, 1.0, A, nodes.ldim(), Bf, shape_fpts.ldim(),
                      0.0, Cf, coord_fpts.ldim());

  dim3 threadsF(32, 4);
  dim3 blocksF((nEles + threadsF.x - 1) / threadsF.x, (nFpts + threadsF.y - 1) / threadsF.y);

  if (nDims == 3)
    copy_coords_face<3><<<blocksF,threadsF>>>(coord_faces, coord_fpts, fpt2gfpt, nEles, nFpts);
  else
    copy_coords_face<2><<<blocksF,threadsF>>>(coord_faces, coord_fpts, fpt2gfpt, nEles, nFpts);

  check_error();
}

template<unsigned int nDims>
__global__
void add_cg_offset(mdvector_gpu<double> nodes, mdvector_gpu<double> x_cg, unsigned int nNodes)
{
  int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= nNodes)
    return;

  for (unsigned int i = node; i < nNodes; i += gridDim.x * blockDim.x)
    for (unsigned int d = 0; d < nDims; d++)
      nodes(i,d) += x_cg(d);
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

    double dx = coord_fpts(fpt1,0,ele) - coord_fpts(fpt2,0,ele);
    double dy = coord_fpts(fpt1,1,ele) - coord_fpts(fpt2,1,ele);
    double dist = std::sqrt(dx*dx + dy*dy);

    h_ref(fpt1, ele) = dist;
    h_ref(fpt2, ele) = dist;
  }
  else
  {
    /// TODO: 3D h_ref from CPU version?
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
void inverse_transform_2d(mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, double *jaco_det, int nEles, int nPts)
{
  int pt = (blockDim.x * blockIdx.x + threadIdx.x) % nPts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nPts;

  if (ele >= nEles)
    return;

  // Determinant of transformation matrix
  if (jaco_det != NULL)
    jaco_det[pt+nPts*ele] = jaco(0,pt,0,ele)*jaco(1,pt,1,ele)-jaco(0,pt,1,ele)*jaco(1,pt,0,ele);

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(0,pt,0,ele) = jaco(1,pt,1,ele);  inv_jaco(0,pt,1,ele) =-jaco(1,pt,0,ele);
  inv_jaco(1,pt,0,ele) =-jaco(0,pt,1,ele);  inv_jaco(1,pt,1,ele) = jaco(0,pt,0,ele);
}

__global__
void inverse_transform_3d(mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, double* jaco_det, int nEles, int nPts)
{
  int pt = (blockDim.x * blockIdx.x + threadIdx.x) % nPts;
  int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nPts;

  if (ele >= nEles)
    return;

  double xr = jaco(0,pt,0,ele);  double xs = jaco(1,pt,0,ele);  double xt = jaco(2,pt,0,ele);
  double yr = jaco(0,pt,1,ele);  double ys = jaco(1,pt,1,ele);  double yt = jaco(2,pt,1,ele);
  double zr = jaco(0,pt,2,ele);  double zs = jaco(1,pt,2,ele);  double zt = jaco(2,pt,2,ele);

  // Determinant of transformation matrix (not always needed)
  if (jaco_det != NULL)
    jaco_det[pt+nPts*ele] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(0,pt,0,ele) = ys*zt - yt*zs;  inv_jaco(0,pt,1,ele) = xt*zs - xs*zt;  inv_jaco(0,pt,2,ele) = xs*yt - xt*ys;
  inv_jaco(1,pt,0,ele) = yt*zr - yr*zt;  inv_jaco(1,pt,1,ele) = xr*zt - xt*zr;  inv_jaco(1,pt,2,ele) = xt*yr - xr*yt;
  inv_jaco(2,pt,0,ele) = yr*zs - ys*zr;  inv_jaco(2,pt,1,ele) = xs*zr - xr*zs;  inv_jaco(2,pt,2,ele) = xr*ys - xs*yr;
}

void calc_transforms_wrapper(mdvector_gpu<double> &nodes, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &jaco_fpts, mdvector_gpu<double> &inv_jaco_spts,
    mdvector_gpu<double> &inv_jaco_fpts, mdvector_gpu<double> &jaco_det_spts,
    mdvector_gpu<double> &dshape_spts, mdvector_gpu<double> &dshape_fpts,
    int nSpts, int nFpts, int nNodes, int nEles, int nDims)
{
  // Calculate forward transform (reference -> physical)
  int ms = nSpts * nDims;
  int mf = nFpts * nDims;
  int k = nNodes;
  int n = nEles * nDims;

  double* B = nodes.data();

  double *As = dshape_spts.data();
  double *Af = dshape_fpts.data();
  double *Cs = jaco_spts.data();
  double *Cf = jaco_fpts.data();

  cublasDGEMM_wrapper(n, ms, k, 1.0, B, n, As, k, 0.0, Cs, n);
  cublasDGEMM_wrapper(n, mf, k, 1.0, B, n, Af, k, 0.0, Cf, n);

  // Calculate inverse transform (physical -> reference) at spts, fpts
  int threads = 128;

  int blocksS = (nSpts * nEles + threads - 1) / threads;
  int blocksF = (nFpts * nEles + threads - 1) / threads;

  if (nDims == 2)
  {
    inverse_transform_2d<<<blocksS,threads>>>(jaco_spts,inv_jaco_spts,
        jaco_det_spts.data(),nEles,nSpts);

    inverse_transform_2d<<<blocksF,threads>>>(jaco_fpts,inv_jaco_fpts,
        NULL,nEles,nFpts);
  }
  else
  {
    inverse_transform_3d<<<blocksS,threads>>>(jaco_spts,inv_jaco_spts,
        jaco_det_spts.data(),nEles,nSpts);

    inverse_transform_3d<<<blocksF,threads>>>(jaco_fpts,inv_jaco_fpts,
        NULL,nEles,nFpts);
  }

  check_error();
}

template<unsigned int nDims>
__global__
void update_transform_rmat(mdvector_gpu<double> jaco_init, mdvector_gpu<double> jaco,
    mdvector_gpu<double> Rmat, unsigned int nEles, unsigned int nSpts)
{
  const int ele = (blockDim.x * blockIdx.x + threadIdx.x) % nEles;
  const int spt = (blockDim.x * blockIdx.x + threadIdx.x) / nEles;

  if (spt >= nSpts) return;

  double J[nDims][nDims] = {{0.0}};
  double R[nDims][nDims];

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      R[i][j] = Rmat(i,j);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      for (int k = 0; k < nDims; k++)
        J[i][j] += R[j][k] * jaco_init(i,spt,k,ele);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      jaco(i,spt,j,ele) = J[i][j];
}

__global__
void update_inv_transform_rmat_2d(mdvector_gpu<double> jaco_init, mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, mdvector_gpu<double> Rmat, unsigned int nEles,
    unsigned int nSpts)
{
  const unsigned int nDims = 2;
  const int ele = (blockDim.x * blockIdx.x + threadIdx.x) % nEles;
  const int spt = (blockDim.x * blockIdx.x + threadIdx.x) / nEles;

  if (spt >= nSpts) return;

  double J[nDims][nDims] = {{0.0}};
  double R[nDims][nDims];

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      R[i][j] = Rmat(i,j);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      for (int k = 0; k < nDims; k++)
        J[i][j] += R[j][k] * jaco_init(i,spt,k,ele);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      jaco(i,spt,j,ele) = J[i][j];

  double xr = J[0][0];  double xs = J[1][0];
  double yr = J[0][1];  double ys = J[1][1];
  double invdet = xr*ys - xs*yr;

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(0,spt,0,ele) =  ys/invdet;  inv_jaco(0,spt,1,ele) = -xs/invdet;
  inv_jaco(1,spt,0,ele) = -yr/invdet;  inv_jaco(1,spt,1,ele) =  xr/invdet;
}

__global__
void update_inv_transform_rmat_3d(mdvector_gpu<double> jaco_init, mdvector_gpu<double> jaco,
    mdvector_gpu<double> inv_jaco, mdvector_gpu<double> Rmat, unsigned int nEles,
    unsigned int nSpts)
{
  const unsigned int nDims = 3;
  const int ele = (blockDim.x * blockIdx.x + threadIdx.x) % nEles;
  const int spt = (blockDim.x * blockIdx.x + threadIdx.x) / nEles;

  if (spt >= nSpts) return;

  double J[nDims][nDims] = {{0.0}};
  double R[nDims][nDims];

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      R[i][j] = Rmat(i,j);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      for (int k = 0; k < nDims; k++)
        J[i][j] += R[j][k] * jaco_init(i,spt,k,ele);

  for (int i = 0; i < nDims; i++)
    for (int j = 0; j < nDims; j++)
      jaco(i,spt,j,ele) = J[i][j];

  double xr = J[0][0];  double xs = J[1][0];  double xt = J[2][0];
  double yr = J[0][1];  double ys = J[1][1];  double yt = J[2][1];
  double zr = J[0][2];  double zs = J[1][2];  double zt = J[2][2];

  // Inverse of transformation matrix (times its determinant)
  inv_jaco(0,spt,0,ele) = ys*zt - yt*zs;  inv_jaco(0,spt,1,ele) = xt*zs - xs*zt;  inv_jaco(0,spt,2,ele) = xs*yt - xt*ys;
  inv_jaco(1,spt,0,ele) = yt*zr - yr*zt;  inv_jaco(1,spt,1,ele) = xr*zt - xt*zr;  inv_jaco(1,spt,2,ele) = xt*yr - xr*yt;
  inv_jaco(2,spt,0,ele) = yr*zs - ys*zr;  inv_jaco(2,spt,1,ele) = xs*zr - xr*zs;  inv_jaco(2,spt,2,ele) = xr*ys - xs*yr;
}

void update_transforms_rigid_wrapper(mdvector_gpu<double> &jaco_spts_init,
    mdvector_gpu<double> &jaco_spts, mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &norm_init,
    mdvector_gpu<double> &norm, mdvector_gpu<double> &Rmat, unsigned int nSpts,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims, bool need_inv)
{
  // Apply rotation matrix to body-frame jacobian
  int threads = 128;
  int blocks = (nSpts*nEles + threads - 1) / threads;
  if (need_inv)
  {
    if (nDims == 3)
      update_inv_transform_rmat_3d<<<blocks,threads>>>(jaco_spts_init,jaco_spts,inv_jaco_spts,Rmat,nEles,nSpts);
    else
      update_inv_transform_rmat_2d<<<blocks,threads>>>(jaco_spts_init,jaco_spts,inv_jaco_spts,Rmat,nEles,nSpts);
  }
  else
  {
    if (nDims == 3)
      update_transform_rmat<3><<<blocks,threads>>>(jaco_spts_init,jaco_spts,Rmat,nEles,nSpts);
    else
      update_transform_rmat<2><<<blocks,threads>>>(jaco_spts_init,jaco_spts,Rmat,nEles,nSpts);
  }

  // Apply rotation matrix to body-frame normals
  double* A = norm_init.data();
  double* B = Rmat.data();
  double* C = norm.data();

  cublasDGEMM_wrapper(nFpts, nDims, nDims, 1.0, A, norm_init.ldim(),
      B, Rmat.ldim(), 0.0, C, norm.ldim());

  check_error();
}

void update_nodes_rigid_wrapper(mdvector_gpu<double> &nodes_init, mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &Rmat, mdvector_gpu<double> &x_cg, unsigned int nNodes, unsigned int nDims)
{
  // Apply rotation matrix to body-frame nodes
  double *A = Rmat.data();
  double *B = nodes_init.data();
  double *C = nodes.data();

  cublasDGEMM_transA_wrapper(nDims, nNodes, nDims, 1.0, A, nDims, B, nDims, 0.0, C, nDims);

  // Add in translation of body's CG
  int threads = 192;
  int blocks = min((nNodes + threads - 1) / threads, MAX_GRID_DIM);

  if (nDims == 3)
    add_cg_offset<3><<<blocks,threads>>>(nodes, x_cg, nNodes);
  else
    add_cg_offset<2><<<blocks,threads>>>(nodes, x_cg, nNodes);

  check_error();
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
      norm(dim1,gfpt) = 0.0;
      for (int dim2 = 0; dim2 < nDims; dim2++)
      {
        norm(dim1, gfpt) += inv_jaco(dim2, fpt, dim1, ele) * tnorm(fpt,dim2);
      }

      DA += norm(dim1, gfpt) * norm(dim1, gfpt);
    }

    DA = sqrt(DA);

    for (int dim = 0; dim < nDims; dim++)
    {
      norm(dim, gfpt) /= DA;
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

  check_error();
}
