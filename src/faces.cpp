#include "faces.hpp"

Faces::Faces(unsigned int nFpts, const InputStruct *input)
{
  this->nFpts = nFpts;
  this->input = input;
}

void Faces::setup(unsigned int nDims, unsigned int nVars)
{
  /* Allocate memory for solution structures */
  U.assign({nVars, nFpts, 2});
  dU.assign({nDims, nVars, nFpts, 2});
  F.assign({nDims, nVars, nFpts, 2});
  Fcomm.assign({nVars, nFpts, 2});

  /* Allocate memory for geometry structures */
  norm.assign({nDims, nFpts, 2});
  outnorm.assign({nFpts,2});
  dA.assign(nFpts,0.0);
  jaco.assign({nFpts, nDims, nDims , 2});
}

void Faces::compute_Fconv()
{  
  if (input->equation == "AdvDiff")
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
          F(0, n, fpt, 0) = input->AdvDiff_Ax * U(n, fpt, 0);
          F(0, n, fpt, 1) = input->AdvDiff_Ax * U(n, fpt, 1);
          F(1, n, fpt, 0) = input->AdvDiff_Ay * U(n, fpt, 0);
          F(1, n, fpt, 1) = input->AdvDiff_Ay * U(n, fpt, 1);
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("Euler flux not implemented yet!");
  }
}

void Faces::compute_common_F()
{
  rusanov_flux();

}
void Faces::rusanov_flux()
{
  std::vector<double> FL(nVars,0.0);
  std::vector<double> FR(nVars,0.0);
  std::vector<double> WL(nVars,0.0);
  std::vector<double> WR(nVars,0.0);

  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Get interface-normal flux components */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += F(dim, n, fpt, 0) * norm(dim, fpt, 0);
        FR[n] += F(dim, n, fpt, 1) * norm(dim, fpt, 1);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(n, fpt, 0); WR[n] = U(n, fpt, 1);
    }

    /* Get numerical wavespeed */
    // TODO: Add generic wavespeed calculation */
    double waveSp = input->AdvDiff_Ax * norm(0,fpt,0);
    waveSp += input->AdvDiff_Ay * norm(1,fpt,0);

    /* Compute common normal flux */
    double k = 0.0;
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(n, fpt, 0) = 0.5 * (FR[n] + FL[n]) - 0.5 * (1.0-k) * (WR[n] - WL[n]);
      Fcomm(n, fpt, 1) = 0.5 * (FR[n] + FL[n]) - 0.5 * (1.0-k) * (WR[n] - WL[n]);

      Fcomm(n, fpt, 0) *= outnorm(fpt,0);
      Fcomm(n, fpt, 1) *= -outnorm(fpt,0);
    }


  }
}
