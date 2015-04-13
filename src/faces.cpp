#include <cmath>
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
  U.assign({nVars, nFpts, 2});
  dU.assign({nDims, nVars, nFpts, 2});
  Fconv.assign({nDims, nVars, nFpts, 2});
  Fvisc.assign({nDims, nVars, nFpts, 2});
  Fcomm.assign({nVars, nFpts, 2});
  Ucomm.assign({nVars, nFpts, 2});

  /* Allocate memory for geometry structures */
  norm.assign({nDims, nFpts, 2});
  outnorm.assign({nFpts,2});
  dA.assign(nFpts,0.0);
  jaco.assign({nFpts, nDims, nDims , 2});
}

void Faces::apply_bcs()
{
  /* Loop over boundary flux points */
  for (unsigned int n = 0; n < nVars; n++)
  {
    unsigned int i = 0;
    for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
    {
      unsigned int bnd_id = geo->gfpt2bnd[i];

      /* Apply specified boundary condition */
      if (bnd_id == 1) /* Periodic */
      {
        unsigned int per_fpt = geo->per_fpt_pairs[fpt];
        U(n, fpt, 1) = U(n, per_fpt, 0);
      }

      i++;
    }
  } 
}

void Faces::apply_bcs_dU()
{
  /* Apply periodic boundaries to solution derivative */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      unsigned int i = 0;
      for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
      {
        unsigned int bnd_id = geo->gfpt2bnd[i];

        /* Apply specified boundary condition */
        if (bnd_id == 1) /* Periodic */
        {
          unsigned int per_fpt = geo->per_fpt_pairs[fpt];
          dU(dim, n, fpt, 1) = dU(dim, n, per_fpt, 0);
        }

        i++;
      }
    }
  }
}


void Faces::compute_Fconv()
{  
  if (input->equation == "AdvDiff")
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        Fconv(0, n, fpt, 0) = input->AdvDiff_Ax * U(n, fpt, 0);
        Fconv(1, n, fpt, 0) = input->AdvDiff_Ay * U(n, fpt, 0);

        Fconv(0, n, fpt, 1) = input->AdvDiff_Ax * U(n, fpt, 1);
        Fconv(1, n, fpt, 1) = input->AdvDiff_Ay * U(n, fpt, 1);

      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("Euler flux not implemented yet!");
  }


}

void Faces::compute_Fvisc()
{  
  if (input->equation == "AdvDiff")
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        Fvisc(0, n, fpt, 0) = -input->AdvDiff_D * dU(0, n, fpt, 0);
        Fvisc(1, n, fpt, 0) = -input->AdvDiff_D * dU(1, n, fpt, 0);

        Fvisc(0, n, fpt, 1) = -input->AdvDiff_D * dU(0, n, fpt, 1);
        Fvisc(1, n, fpt, 1) = -input->AdvDiff_D * dU(1, n, fpt, 1);

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
  
  double beta = 0.5; 

  /* Compute common solution */
  if (input->fvisc_type == "LDG")
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Get left and right state variables */
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(n, fpt, 0); double UR = U(n, fpt, 1);

        Ucomm(n, fpt, 0) = 0.5*(UL + UR) - beta*(UL - UR);
        Ucomm(n, fpt, 1) = 0.5*(UL + UR) - beta*(UL - UR);

        /* Trying pure central */
        Ucomm(n, fpt, 0) = 0.5*(UL + UR);
        Ucomm(n, fpt, 1) = 0.5*(UL + UR);
      }

    }
  }

  else if (input->fvisc_type == "Central")
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Get left and right state variables */
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(n, fpt, 0); double UR = U(n, fpt, 1);

        Ucomm(n, fpt, 0) = 0.5*(UL + UR);
        Ucomm(n, fpt, 1) = 0.5*(UL + UR);
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
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);

  double k = input->rus_k;

  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fconv(dim, n, fpt, 0) * norm(dim, fpt, 0);
        FR[n] += Fconv(dim, n, fpt, 1) * norm(dim, fpt, 0);
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
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(n, fpt, 0) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp)*(1.0-k) * (WR[n]-WL[n]);
      Fcomm(n, fpt, 1) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp)*(1.0-k) * (WR[n]-WL[n]);

      /* Correct for positive parent space sign convention */
      Fcomm(n, fpt, 0) *= outnorm(fpt,0);
      Fcomm(n, fpt, 1) *= -outnorm(fpt,1);
    }
  }
}

void Faces::transform_flux()
{
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      Fcomm(n, fpt, 0) *= dA[fpt];
      Fcomm(n, fpt, 1) *= dA[fpt];
    }
  }
}

void Faces::LDG_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
  double beta = input->ldg_b;
  double tau = input->ldg_tau;

  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(dim, n, fpt, 0) * norm(dim, fpt, 0);
        FR[n] += Fvisc(dim, n, fpt, 1) * norm(dim, fpt, 0);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(n, fpt, 0); WR[n] = U(n, fpt, 1);
    }

    /* Compute common normal viscous flux and accumulate */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(n, fpt, 0) += (0.5 * (FL[n]+FR[n]) + tau * (WL[n] - WR[n]) + beta * (FL[n] - FR[n]))
                          * outnorm(fpt,0); 
      Fcomm(n, fpt, 1) += (0.5 * (FL[n]+FR[n]) + tau * (WL[n] - WR[n]) + beta * (FL[n] - FR[n]))
                          * -outnorm(fpt,1); 
      
    }
  }
}

void Faces::central_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    FL.assign(nVars,0.0); FR.assign(nVars,0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(dim, n, fpt, 0) * norm(dim, fpt, 0);
        FR[n] += Fvisc(dim, n, fpt, 1) * norm(dim, fpt, 0);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(n, fpt, 0); WR[n] = U(n, fpt, 1);
    }

    /* Compute common normal viscous flux and accumulate */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(n, fpt, 0) += (0.5 * (FL[n]+FR[n])) * outnorm(fpt,0); 
      Fcomm(n, fpt, 1) += (0.5 * (FL[n]+FR[n])) * -outnorm(fpt,1); 
    }
  }
}



