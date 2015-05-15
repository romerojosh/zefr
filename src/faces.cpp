#include <cmath>

#include <omp.h>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"


Faces::Faces(GeoStruct *geo, const InputStruct *input)
{
  this->input = input;
  this->geo = geo;
  nFpts = geo->nGfpts;
}

void Faces::setup(unsigned int nDims, unsigned int nVars)
{
  this->nVars = nVars;
  this->nDims = nDims;

  /* Allocate memory for solution structures */
  U.assign({2, nFpts, nVars});
  dU.assign({2, nFpts, nVars, nDims});
  Fconv.assign({2, nFpts, nVars, nDims});
  Fvisc.assign({2, nFpts, nVars, nDims});
  Fcomm.assign({2, nFpts, nVars});
  Ucomm.assign({2, nFpts, nVars});

  /* Allocate memory for geometry structures */
  norm.assign({2, nFpts, nDims});
  outnorm.assign({2, nFpts});
  dA.assign(nFpts,0.0);
  jaco.assign({2, nDims, nDims , nFpts});
}

void Faces::apply_bcs()
{
  /* Loop over boundary flux points */
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < nVars; n++)
  {
    //unsigned int i = 0;
    for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
    {
      unsigned int bnd_id = geo->gfpt2bnd[fpt - geo->nGfpts_int];
      //unsigned int bnd_id = geo->gfpt2bnd[i];

      /* Apply specified boundary condition */
      if (bnd_id == 1) /* Periodic */
      {
        unsigned int per_fpt = geo->per_fpt_pairs[fpt];
        U(1, fpt, n) = U(0, per_fpt, n);
      }
      else if (bnd_id == 2) /* Farfield */
      {
        U(1, fpt, n) = 0.; // Hardcode for sine case
      }

      //i++;
    }
  } 
}

void Faces::apply_bcs_dU()
{
  /* Apply periodic boundaries to solution derivative */
#pragma omp parallel for collapse(3)
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      //unsigned int i = 0;
      for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
      {
        unsigned int bnd_id = geo->gfpt2bnd[fpt - geo->nGfpts_int];

        /* Apply specified boundary condition */
        if (bnd_id == 1) /* Periodic */
        {
          unsigned int per_fpt = geo->per_fpt_pairs[fpt];
          dU(1, fpt, n, dim) = dU(0, per_fpt, n, dim);
        }

        //i++;
      }
    }
  }
}


void Faces::compute_Fconv()
{  
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(3)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        {
          Fconv(0, fpt, n, dim) = input->AdvDiff_A[dim] * U(0, fpt, n);

          Fconv(1, fpt, n, dim) = input->AdvDiff_A[dim] * U(1, fpt, n);

        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    if (nDims == 2)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        for (unsigned int slot = 0; slot < 2; slot ++)
        {
          /* Compute some primitive variables */
          double momF = (U(slot, fpt, 1) * U(slot, fpt ,1) + U(slot, fpt, 2) * 
              U(slot, fpt, 2)) / U(slot, fpt, 0);
          double P = (input->gamma - 1.0) * (U(slot, fpt, 3)) - 0.5 * momF;
          double H = (U(slot, fpt, 3) + P) / U(slot, fpt, 0);

          Fconv(slot, fpt, 0, 0) = U(slot, fpt, 1);
          Fconv(slot, fpt, 1, 0) = U(slot, fpt, 1) * U(slot, fpt, 1) / U(slot, fpt, 0) + P;
          Fconv(slot, fpt, 2, 0) = U(slot, fpt, 1) * U(slot, fpt, 2) / U(slot, fpt, 0);
          Fconv(slot, fpt, 3, 0) = U(slot, fpt, 1) * H;

          Fconv(slot, fpt, 0, 1) = U(slot, fpt, 2);
          Fconv(slot, fpt, 1, 1) = U(slot, fpt, 1) * U(slot, fpt, 2) / U(slot, fpt, 0);
          Fconv(slot, fpt, 2, 1) = U(slot, fpt, 2) * U(slot, fpt, 2) / U(slot, fpt, 0) + P;
          Fconv(slot, fpt, 3, 1) = U(slot, fpt, 2) * H;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("3D Euler not implemented yet!");
    }
  }


}

void Faces::compute_Fvisc()
{  
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(3)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        {
          Fvisc(0, fpt, n, dim) = -input->AdvDiff_D * dU(0, fpt, n, dim);

          Fvisc(1, fpt, n, dim) = -input->AdvDiff_D * dU(1, fpt, n, dim);

        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("NS flux not implemented yet!");
  }
}

void Faces::compute_common_F()
{
  if (input->fconv_type == "Rusanov")
    rusanov_flux();
  else
    ThrowException("Numerical convective flux type not recognized!");

  if (input->viscous)
  {
    if (input->fvisc_type == "LDG")
      LDG_flux();
    else if (input->fvisc_type == "Central")
      central_flux();
    else
      ThrowException("Numerical viscous flux type not recognized!");
  }

  transform_flux();
}

void Faces::compute_common_U()
{
  
  double beta = input->ldg_b;

  /* Compute common solution */
  if (input->fvisc_type == "LDG")
  {
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Get left and right state variables */
      // TODO: Verify that this is the correct formula. Seem different than papers...
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(0, fpt, n); double UR = U(1, fpt, n);

         Ucomm(0, fpt, n) = 0.5*(UL + UR) - beta*(UL - UR);
         Ucomm(1, fpt, n) = 0.5*(UL + UR) - beta*(UL - UR);

      }

    }
  }

  // TODO: Can potentially remove central treatment since LDG recovers.
  else if (input->fvisc_type == "Central")
  {
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Get left and right state variables */
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(0, fpt, n); double UR = U(1, fpt, n);

        Ucomm(0, fpt, n) = 0.5*(UL + UR);
        Ucomm(1, fpt, n) = 0.5*(UL + UR);
      }

    }
  }
  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }

}

void Faces::rusanov_flux()
{

  double k = input->rus_k;

  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);

#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fconv(0, fpt, n, dim) * norm(0, fpt, dim);
        FR[n] += Fconv(1, fpt, n, dim) * norm(0, fpt, dim);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, fpt, n); WR[n] = U(1, fpt, n);
    }

    /* Get numerical wavespeed */
    // TODO: Add generic wavespeed calculation */
    double waveSp = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
      waveSp += input->AdvDiff_A[dim] * norm(0, fpt, dim);

    /* Compute common normal flux */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, fpt, n) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp)*(1.0-k) * (WR[n]-WL[n]);
      Fcomm(1, fpt, n) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp)*(1.0-k) * (WR[n]-WL[n]);

      /* Correct for positive parent space sign convention */
      Fcomm(0, fpt, n) *= outnorm(0, fpt);
      Fcomm(1, fpt, n) *= -outnorm(1, fpt);
    }
  }
}

void Faces::transform_flux()
{
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      Fcomm(0, fpt, n) *= dA[fpt];
      Fcomm(1, fpt, n) *= dA[fpt];
    }
  }
}

void Faces::LDG_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
  double tau = input->ldg_tau;
  double beta = input->ldg_b;

  mdvector<double> Fcomm_temp({nVars,nDims});

#pragma omp parallel for firstprivate(FL, FR, WL, WR, Fcomm_temp, tau, beta)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {

    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(0, fpt, n, dim) * norm(0, fpt, dim);
        FR[n] += Fvisc(1, fpt, n, dim) * norm(0, fpt, dim);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, fpt, n); WR[n] = U(1, fpt, n);
    }

    /* Compute common normal viscous flux and accumulate */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm_temp(n,0) += 0.5*(Fvisc(0,fpt,n,0) + Fvisc(1,fpt,n,0)) + tau * norm(0,fpt,0)* (WL[n]
          - WR[n]) + beta * norm(0, fpt, 0)* (FL[n] - FR[n]);
      Fcomm_temp(n,1) += 0.5*(Fvisc(0,fpt,n,1) + Fvisc(1,fpt,n,1)) + tau * norm(0,fpt,1)* (WL[n]
          - WR[n]) + beta * norm(0, fpt, 1)* (FL[n] - FR[n]);
    }

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm(0, fpt, n) += (Fcomm_temp(n, dim) * norm(0, fpt, dim)) * outnorm(0,fpt);
        Fcomm(1, fpt, n) += (Fcomm_temp(n, dim) * norm(0, fpt, dim)) * -outnorm(1,fpt);
      }
    }

    Fcomm_temp.fill(0);
  }
}

void Faces::central_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(0, fpt, n, dim) * norm(0, fpt, dim);
        FR[n] += Fvisc(1, fpt, n, dim) * norm(0, fpt, dim);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, fpt, n); WR[n] = U(1, fpt, n);
    }

    /* Compute common normal viscous flux and accumulate */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, fpt, n) += (0.5 * (FL[n]+FR[n])) * outnorm(0,fpt); 
      Fcomm(1, fpt, n) += (0.5 * (FL[n]+FR[n])) * -outnorm(1,fpt); 
    }
  }
}



