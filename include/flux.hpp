template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Fconv_AdvDiff(double U[nVars], double F[nVars][nDims], double A[nDims], double Vg[nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    F[0][dim] = (A[dim] - Vg[dim]) * U[0];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Fconv_EulerNS(double U[nVars], double F[nVars][nDims], double Vg[nDims], double &P, double gamma)
{
  if (nDims == 2)
  {
    double invrho = 1./U[0];
    double momF = (U[1] * U[1] + U[2] * U[2]) * invrho;

    P = (gamma - 1.0) * (U[3] - 0.5 * momF);

    double H = (U[3] + P) * invrho;

    F[0][0] = -(U[0] * Vg[0]) + U[1];
    F[1][0] = -(U[1] * Vg[0]) + U[1] * U[1] * invrho + P - U[1]*Vg[0];
    F[2][0] = -(U[2] * Vg[0]) + U[1] * U[2] * invrho - U[2]*Vg[0];
    F[3][0] = -(U[3] * Vg[0]) + U[1] * H - U[3]*Vg[0];

    F[0][1] = -(U[0] * Vg[1]) + U[2];
    F[1][1] = -(U[1] * Vg[1]) + U[2] * U[1] * invrho;
    F[2][1] = -(U[2] * Vg[1]) + U[2] * U[2] * invrho + P;
    F[3][1] = -(U[3] * Vg[1]) + U[2] * H;
  }
  else if (nDims == 3)
  {
    double invrho = 1./U[0];
    double momF = (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) * invrho;

    P = (gamma - 1.0) * (U[4] - 0.5 * momF);

    double H = (U[4] + P) * invrho; 

    F[0][0] = -(U[0] * Vg[0]) + U[1];
    F[1][0] = -(U[1] * Vg[0]) + U[1] * U[1] * invrho + P;
    F[2][0] = -(U[2] * Vg[0]) + U[1] * U[2] * invrho;
    F[3][0] = -(U[3] * Vg[0]) + U[1] * U[3] * invrho;
    F[4][0] = -(U[4] * Vg[0]) + U[1] * H;

    F[0][1] = -(U[0] * Vg[1]) + U[2];
    F[1][1] = -(U[1] * Vg[1]) + U[2] * U[1] * invrho;
    F[2][1] = -(U[2] * Vg[1]) + U[2] * U[2] * invrho + P;
    F[3][1] = -(U[3] * Vg[1]) + U[2] * U[3] * invrho;
    F[4][1] = -(U[4] * Vg[1]) + U[2] * H;

    F[0][2] = -(U[0] * Vg[2]) + U[3];
    F[1][2] = -(U[1] * Vg[2]) + U[3] * U[1] * invrho;
    F[2][2] = -(U[2] * Vg[2]) + U[3] * U[2] * invrho;
    F[3][2] = -(U[3] * Vg[2]) + U[3] * U[3] * invrho + P;
    F[4][2] = -(U[4] * Vg[2]) + U[3] * H;
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Fvisc_AdvDiff_add(double dU[nVars][nDims], double F[nVars][nDims], double D)
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    F[0][dim] -= D * dU[0][dim];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Fvisc_EulerNS_add(double U[nVars], double dU[nVars][nDims], double F[nVars][nDims], 
    double gamma, double prandtl, double mu_in, double rt, double c_sth, bool fix_vis)
{
  if (nDims == 2)
  {
    double invrho = 1.0 / U[0];

    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e_int = U[3] * invrho - 0.5 * (u*u + v*v);

    /* Set viscosity */
    double mu;
    if (fix_vis)
    {
      mu = mu_in;
    }
    /* If desired, use Sutherland's law */
    else
    {
      double rt_ratio = (gamma - 1.0) * e_int / (rt);
      mu = mu_in * pow(rt_ratio, 1.5) * (1. + c_sth) / (rt_ratio + c_sth);
    }

    double du_dx = (dU[1][0] - dU[0][0] * u) * invrho;
    double du_dy = (dU[1][1] - dU[0][1] * u) * invrho;

    double dv_dx = (dU[2][0] - dU[0][0] * v) * invrho;
    double dv_dy = (dU[2][1] - dU[0][1] * v) * invrho;

    double dke_dx = 0.5 * (u*u + v*v) * dU[0][0] + U[0] * (u * du_dx + v * dv_dx);
    double dke_dy = 0.5 * (u*u + v*v) * dU[0][1] + U[0] * (u * du_dy + v * dv_dy);

    double de_dx = (dU[3][0] - dke_dx - dU[0][0] * e_int) * invrho;
    double de_dy = (dU[3][1] - dke_dy - dU[0][1] * e_int) * invrho;

    double diag = (du_dx + dv_dy) / 3.0;

    double tauxx = 2.0 * mu * (du_dx - diag);
    double tauxy = mu * (du_dy + dv_dx);
    double tauyy = 2.0 * mu * (dv_dy - diag);

    /*  Add viscous flux values */
    F[1][0] -= tauxx;
    F[2][0] -= tauxy;
    F[3][0] -= (u * tauxx + v * tauxy + (mu / prandtl) *
        gamma * de_dx);

    F[1][1] -= tauxy;
    F[2][1] -= tauyy;
    F[3][1] -= (u * tauxy + v * tauyy + (mu / prandtl) *
        gamma * de_dy);
  }
  else if (nDims == 3)
  {
    double invrho = 1.0 / U[0];

    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e_int = U[4] * invrho - 0.5 * (u*u + v*v + w*w);

    /* Set viscosity */
    double mu;
    if (fix_vis)
    {
      mu = mu_in;
    }
    else
    {
      double rt_ratio = (gamma - 1.0) * e_int / (rt);
      mu = mu_in * std::pow(rt_ratio,1.5) * (1. + c_sth) / (rt_ratio + c_sth);
    }

    double du_dx = (dU[1][0] - dU[0][0] * u) * invrho;
    double du_dy = (dU[1][1] - dU[0][1] * u) * invrho;
    double du_dz = (dU[1][2] - dU[0][2] * u) * invrho;

    double dv_dx = (dU[2][0] - dU[0][0] * v) * invrho;
    double dv_dy = (dU[2][1] - dU[0][1] * v) * invrho;
    double dv_dz = (dU[2][2] - dU[0][2] * v) * invrho;

    double dw_dx = (dU[3][0] - dU[0][0] * w) * invrho;
    double dw_dy = (dU[3][1] - dU[0][1] * w) * invrho;
    double dw_dz = (dU[3][2] - dU[0][2] * w) * invrho;

    double dke_dx = 0.5 * (u*u + v*v + w*w) * dU[0][0] + U[0] * (u * du_dx + v * dv_dx + w * dw_dx);
    double dke_dy = 0.5 * (u*u + v*v + w*w) * dU[0][1] + U[0] * (u * du_dy + v * dv_dy + w * dw_dy);
    double dke_dz = 0.5 * (u*u + v*v + w*w) * dU[0][2] + U[0] * (u * du_dz + v * dv_dz + w * dw_dz);

    double de_dx = (dU[4][0] - dke_dx - dU[0][0] * e_int) * invrho;
    double de_dy = (dU[4][1] - dke_dy - dU[0][1] * e_int) * invrho;
    double de_dz = (dU[4][2] - dke_dz - dU[0][2] * e_int) * invrho;

    double diag = (du_dx + dv_dy + dw_dz) / 3.0;

    double tauxx = 2.0 * mu * (du_dx - diag);
    double tauyy = 2.0 * mu * (dv_dy - diag);
    double tauzz = 2.0 * mu * (dw_dz - diag);
    double tauxy = mu * (du_dy + dv_dx);
    double tauxz = mu * (du_dz + dw_dx);
    double tauyz = mu * (dv_dz + dw_dy);

    /* Add viscous flux values */
    F[1][0] -= tauxx;
    F[2][0] -= tauxy;
    F[3][0] -= tauxz;
    F[4][0] -= (u * tauxx + v * tauxy + w * tauxz + (mu / prandtl) *
      gamma * de_dx);

    F[1][1] -= tauxy;
    F[2][1] -= tauyy;
    F[3][1] -= tauyz;
    F[4][1] -= (u * tauxy + v * tauyy + w * tauyz + (mu / prandtl) *
      gamma * de_dy);

    F[1][2] -= tauxz;
    F[2][2] -= tauyz;
    F[3][2] -= tauzz;
    F[4][2] -= (u * tauxz + v * tauyz + w * tauzz + (mu / prandtl) *
      gamma * de_dz);
  }

}


template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUconv_AdvDiff(double dFdU[nVars][nVars][nDims], double A[nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    dFdU[0][0][dim] = A[dim];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUconv_EulerNS(double U[nVars], double dFdU[nVars][nVars][nDims], double gamma)
{
  if (nDims == 2)
  {
    /* Primitive Variables */
    double invrho = 1.0 / U[0];
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Set convective dFdU values in the x-direction */
    dFdU[1][0][0] = 0.5 * ((gamma-3.0) * u*u + (gamma-1.0) * v*v);
    dFdU[2][0][0] = -u * v;
    dFdU[3][0][0] = -gamma * e * u * invrho + (gamma-1.0) * u * (u*u + v*v);

    dFdU[0][1][0] = 1;
    dFdU[1][1][0] = (3.0-gamma) * u;
    dFdU[2][1][0] = v;
    dFdU[3][1][0] = gamma * e * invrho + 0.5 * (1.0-gamma) * (3.0*u*u + v*v);

    dFdU[1][2][0] = (1.0-gamma) * v;
    dFdU[2][2][0] = u;
    dFdU[3][2][0] = (1.0-gamma) * u * v;

    dFdU[1][3][0] = (gamma-1.0);
    dFdU[3][3][0] = gamma * u;

    /* Set convective dFdU values in the y-direction */
    dFdU[1][0][1] = -u * v;
    dFdU[2][0][1] = 0.5 * ((gamma-1.0) * u*u + (gamma-3.0) * v*v);
    dFdU[3][0][1] = -gamma * e * v * invrho + (gamma-1.0) * v * (u*u + v*v);

    dFdU[1][1][1] = v;
    dFdU[2][1][1] = (1.0-gamma) * u;
    dFdU[3][1][1] = (1.0-gamma) * u * v;

    dFdU[0][2][1] = 1;
    dFdU[1][2][1] = u;
    dFdU[2][2][1] = (3.0-gamma) * v;
    dFdU[3][2][1] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + 3.0*v*v);

    dFdU[2][3][1] = (gamma-1.0);
    dFdU[3][3][1] = gamma * v;
  }
  else if (nDims == 3)
  {
    /* Primitive Variables */
    double invrho = 1.0 / U[0];
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Set convective dFdU values in the x-direction */
    dFdU[1][0][0] = 0.5 * ((gamma-3.0) * u*u + (gamma-1.0) * (v*v + w*w));
    dFdU[2][0][0] = -u * v;
    dFdU[3][0][0] = -u * w;
    dFdU[4][0][0] = -gamma * e * u * invrho + (gamma-1.0) * u * (u*u + v*v + w*w);

    dFdU[0][1][0] = 1;
    dFdU[1][1][0] = (3.0-gamma) * u;
    dFdU[2][1][0] = v;
    dFdU[3][1][0] = w;
    dFdU[4][1][0] = gamma * e * invrho + 0.5 * (1.0-gamma) * (3.0*u*u + v*v + w*w);

    dFdU[1][2][0] = (1.0-gamma) * v;
    dFdU[2][2][0] = u;
    dFdU[4][2][0] = (1.0-gamma) * u * v;

    dFdU[1][3][0] = (1.0-gamma) * w;
    dFdU[3][3][0] = u;
    dFdU[4][3][0] = (1.0-gamma) * u * w;

    dFdU[1][4][0] = (gamma-1.0);
    dFdU[4][4][0] = gamma * u;

    /* Set convective dFdU values in the y-direction */
    dFdU[1][0][1] = -u * v;
    dFdU[2][0][1] = 0.5 * ((gamma-1.0) * (u*u + w*w) + (gamma-3.0) * v*v);
    dFdU[3][0][1] = -v * w;
    dFdU[4][0][1] = -gamma * e * v * invrho + (gamma-1.0) * v * (u*u + v*v + w*w);

    dFdU[1][1][1] = v;
    dFdU[2][1][1] = (1.0-gamma) * u;
    dFdU[4][1][1] = (1.0-gamma) * u * v;

    dFdU[0][2][1] = 1;
    dFdU[1][2][1] = u;
    dFdU[2][2][1] = (3.0-gamma) * v;
    dFdU[3][2][1] = w;
    dFdU[4][2][1] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + 3.0*v*v + w*w);

    dFdU[2][3][1] = (1.0-gamma) * w;
    dFdU[3][3][1] = v;
    dFdU[4][3][1] = (1.0-gamma) * v * w;

    dFdU[2][4][1] = (gamma-1.0);
    dFdU[4][4][1] = gamma * v;

    /* Set convective dFdU values in the z-direction */
    dFdU[1][0][2] = -u * w;
    dFdU[2][0][2] = -v * w;
    dFdU[3][0][2] = 0.5 * ((gamma-1.0) * (u*u + v*v) + (gamma-3.0) * w*w);
    dFdU[4][0][2] = -gamma * e * w * invrho + (gamma-1.0) * w * (u*u + v*v + w*w);

    dFdU[1][1][2] = w;
    dFdU[3][1][2] = (1.0-gamma) * u;
    dFdU[4][1][2] = (1.0-gamma) * u * w;

    dFdU[2][2][2] = w;
    dFdU[3][2][2] = (1.0-gamma) * v;
    dFdU[4][2][2] = (1.0-gamma) * v * w;

    dFdU[0][3][2] = 1;
    dFdU[1][3][2] = u;
    dFdU[2][3][2] = v;
    dFdU[3][3][2] = (3.0-gamma) * w;
    dFdU[4][3][2] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + v*v + 3.0*w*w);

    dFdU[3][4][2] = (gamma-1.0);
    dFdU[4][4][2] = gamma * w;
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFddUvisc_AdvDiff(double dFddU[nVars][nVars][nDims][nDims], double D)
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    dFddU[0][0][dim][dim] = -D;
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUvisc_EulerNS_add(double U[nVars], double dU[nVars][nDims], double dFdU[nVars][nVars][nDims], 
    double gamma, double prandtl, double mu_in)
{
  // TODO: Add Sutherland's law and store mu in array
  double invrho = 1.0 / U[0];
  double mu = mu_in;
  double diffCo1 = mu * invrho;
  double diffCo2 = gamma * mu * invrho / prandtl;

  if (nDims == 2)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Gradients */
    double rho_dx = dU[0][0];
    double momx_dx = dU[1][0];
    double momy_dx = dU[2][0];
    double e_dx = dU[3][0];
    
    double rho_dy = dU[0][1];
    double momx_dy = dU[1][1];
    double momy_dy = dU[2][1];
    double e_dy = dU[3][1];

    /* Set viscous dFdU values in the x-direction */
    dFdU[1][0][0] -= (2.0/3.0) * (4.0*u*rho_dx - 2.0*(v*rho_dy + momx_dx) + momy_dy) * invrho * diffCo1;
    dFdU[2][0][0] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[3][0][0] -= (1.0/3.0) * (3.0*(4.0*u*u + 3.0*v*v)*rho_dx + 3.0*u*v*rho_dy - 4.0*u*(2.0*momx_dx - momy_dy) - 6.0*v*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                 (-e_dx + (2.0*e*invrho - 3.0*(u*u + v*v))*rho_dx + 2.0*(u*momx_dx + v*momy_dx)) * invrho * diffCo2;

    dFdU[1][1][0] -= -(4.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[2][1][0] -= -rho_dy * invrho * diffCo1;
    dFdU[3][1][0] -= -(1.0/3.0) * (8.0*u*rho_dx + v*rho_dy - 4.0*momx_dx + 2.0*momy_dy) * invrho * diffCo1 + 
                                  (2.0*u*rho_dx - momx_dx) * invrho * diffCo2;

    dFdU[1][2][0] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[2][2][0] -= -rho_dx * invrho * diffCo1;
    dFdU[3][2][0] -= -(1.0/3.0) * (6.0*v*rho_dx + u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dx - momy_dx) * invrho * diffCo2;

    dFdU[3][3][0] -= -rho_dx * invrho * diffCo2;

    /* Set viscous dFdU values in the y-direction */
    dFdU[1][0][1] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[2][0][1] -= (2.0/3.0) * (4.0*v*rho_dy - 2.0*(u*rho_dx + momy_dy) + momx_dx) * invrho * diffCo1;
    dFdU[3][0][1] -= (1.0/3.0) * (3.0*(3.0*u*u + 4.0*v*v)*rho_dy + 3.0*u*v*rho_dx - 6.0*u*(momx_dy + momy_dx) - 4.0*v*(-momx_dx + 2.0*momy_dy)) * invrho * diffCo1 + 
                                 (-e_dy + (2.0*e*invrho - 3.0*(u*u + v*v))*rho_dy + 2.0*(u*momx_dy + v*momy_dy)) * invrho * diffCo2;

    dFdU[1][1][1] -= -rho_dy * invrho * diffCo1;
    dFdU[2][1][1] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[3][1][1] -= -(1.0/3.0) * (v*rho_dx + 6.0*u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dy - momx_dy) * invrho * diffCo2;

    dFdU[1][2][1] -= -rho_dx * invrho * diffCo1;
    dFdU[2][2][1] -= -(4.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[3][2][1] -= -(1.0/3.0) * (u*rho_dx + 8.0*v*rho_dy + 2.0*momx_dx - 4.0*momy_dy) * invrho * diffCo1 + 
                                  (2.0*v*rho_dy - momy_dy) * invrho * diffCo2;

    dFdU[3][3][1] -= -rho_dy * invrho * diffCo2;
  }
  else if (nDims == 3)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Gradients */
    double rho_dx = dU[0][0];
    double momx_dx = dU[1][0];
    double momy_dx = dU[2][0];
    double momz_dx = dU[3][0];
    double e_dx = dU[4][0];
    
    double rho_dy = dU[0][1];
    double momx_dy = dU[1][1];
    double momy_dy = dU[2][1];
    double momz_dy = dU[3][1];
    double e_dy = dU[4][1];

    double rho_dz = dU[0][2];
    double momx_dz = dU[1][2];
    double momy_dz = dU[2][2];
    double momz_dz = dU[3][2];
    double e_dz = dU[4][2];

    /* Set viscous dFdU values in the x-direction */
    dFdU[1][0][0] -= (2.0/3.0) * (4.0*u*rho_dx - 2.0*(momx_dx + v*rho_dy + w*rho_dz) + momy_dy + momz_dz) * invrho * diffCo1;
    dFdU[2][0][0] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[3][0][0] -= (2.0*(w*rho_dx + u*rho_dz) - (momx_dz + momz_dx)) * invrho * diffCo1;
    dFdU[4][0][0] -= (1.0/3.0) * (3.0*(4.0*u*u + 3.0*(v*v + w*w))*rho_dx + 3.0*(u*v*rho_dy + u*w*rho_dz) 
                                  - 4.0*u*(2.0*momx_dx - momy_dy - momz_dz) - 6.0*(v*(momy_dx + momx_dy) + w*(momz_dx + momx_dz))) * invrho * diffCo1 + 
                                 (-e_dx + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dx + 2.0*(u*momx_dx + v*momy_dx + w*momz_dx)) * invrho * diffCo2;

    dFdU[1][1][0] -= -(4.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[2][1][0] -= -rho_dy * invrho * diffCo1;
    dFdU[3][1][0] -= -rho_dz * invrho * diffCo1;
    dFdU[4][1][0] -= -(1.0/3.0) * (8.0*u*rho_dx + v*rho_dy + w*rho_dz - 4.0*momx_dx + 2.0*(momy_dy + momz_dz)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dx - momx_dx) * invrho * diffCo2;

    dFdU[1][2][0] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[2][2][0] -= -rho_dx * invrho * diffCo1;
    dFdU[4][2][0] -= -(1.0/3.0) * (6.0*v*rho_dx + u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dx - momy_dx) * invrho * diffCo2;

    dFdU[1][3][0] -= (2.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[3][3][0] -= -rho_dx * invrho * diffCo1;
    dFdU[4][3][0] -= -(1.0/3.0) * (6.0*w*rho_dx + u*rho_dz - 3.0*(momx_dz + momz_dx)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dx - momz_dx) * invrho * diffCo2;

    dFdU[4][4][0] -= -rho_dx * invrho * diffCo2;

    /* Set viscous dFdU values in the y-direction */
    dFdU[1][0][1] -= (2.0*(u*rho_dy + v*rho_dx) - (momy_dx + momx_dy)) * invrho * diffCo1;
    dFdU[2][0][1] -= (2.0/3.0) * (4.0*v*rho_dy - 2.0*(momy_dy + u*rho_dx + w*rho_dz) + momx_dx + momz_dz) * invrho * diffCo1;
    dFdU[3][0][1] -= (2.0*(w*rho_dy + v*rho_dz) - (momy_dz + momz_dy)) * invrho * diffCo1;
    dFdU[4][0][1] -= (1.0/3.0) * (3.0*(4.0*v*v + 3.0*(u*u + w*w))*rho_dy + 3.0*(v*u*rho_dx + v*w*rho_dz) 
                                  - 4.0*v*(2.0*momy_dy - momx_dx - momz_dz) - 6.0*(u*(momx_dy + momy_dx) + w*(momz_dy + momy_dz))) * invrho * diffCo1 + 
                                 (-e_dy + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dy + 2.0*(u*momx_dy + v*momy_dy + w*momz_dy)) * invrho * diffCo2;

    dFdU[1][1][1] -= -rho_dy * invrho * diffCo1;
    dFdU[2][1][1] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[4][1][1] -= -(1.0/3.0) * (6.0*u*rho_dy + v*rho_dx - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dy - momx_dy) * invrho * diffCo2;

    dFdU[1][2][1] -= -rho_dx * invrho * diffCo1;
    dFdU[2][2][1] -= -(4.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[3][2][1] -= -rho_dz * invrho * diffCo1;
    dFdU[4][2][1] -= -(1.0/3.0) * (8.0*v*rho_dy + u*rho_dx + w*rho_dz - 4.0*momy_dy + 2.0*(momx_dx + momz_dz)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dy - momy_dy) * invrho * diffCo2;

    dFdU[2][3][1] -= (2.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[3][3][1] -= -rho_dy * invrho * diffCo1;
    dFdU[4][3][1] -= -(1.0/3.0) * (6.0*w*rho_dy + v*rho_dz - 3.0*(momz_dy + momy_dz)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dy - momz_dy) * invrho * diffCo2;

    dFdU[4][4][1] -= -rho_dy * invrho * diffCo2;

    /* Set viscous dFdU values in the z-direction */
    dFdU[1][0][2] -= (2.0*(u*rho_dz + w*rho_dx) - (momz_dx + momx_dz)) * invrho * diffCo1;
    dFdU[2][0][2] -= (2.0*(v*rho_dz + w*rho_dy) - (momz_dy + momy_dz)) * invrho * diffCo1;
    dFdU[3][0][2] -= (2.0/3.0) * (4.0*w*rho_dz - 2.0*(momz_dz + u*rho_dx + v*rho_dy) + momx_dx + momy_dy) * invrho * diffCo1;
    dFdU[4][0][2] -= (1.0/3.0) * (3.0*(4.0*w*w + 3.0*(u*u + v*v))*rho_dz + 3.0*(w*u*rho_dx + w*v*rho_dy) 
                                  - 4.0*w*(2.0*momz_dz - momx_dx - momy_dy) - 6.0*(u*(momx_dz + momz_dx) + v*(momy_dz + momz_dy))) * invrho * diffCo1 + 
                                 (-e_dz + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dz + 2.0*(u*momx_dz + v*momy_dz + w*momz_dz)) * invrho * diffCo2;

    dFdU[1][1][2] -= -rho_dz * invrho * diffCo1;
    dFdU[3][1][2] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[4][1][2] -= -(1.0/3.0) * (6.0*u*rho_dz + w*rho_dx - 3.0*(momx_dz + momz_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dz - momx_dz) * invrho * diffCo2;

    dFdU[2][2][2] -= -rho_dz * invrho * diffCo1;
    dFdU[3][2][2] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[4][2][2] -= -(1.0/3.0) * (6.0*v*rho_dz + w*rho_dy - 3.0*(momy_dz + momz_dy)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dz - momy_dz) * invrho * diffCo2;

    dFdU[1][3][2] -= -rho_dx * invrho * diffCo1;
    dFdU[2][3][2] -= -rho_dy * invrho * diffCo1;
    dFdU[3][3][2] -= -(4.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[4][3][2] -= -(1.0/3.0) * (8.0*w*rho_dz + u*rho_dx + v*rho_dy - 4.0*momz_dz + 2.0*(momx_dx + momy_dy)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dz - momz_dz) * invrho * diffCo2;

    dFdU[4][4][2] -= -rho_dz * invrho * diffCo2;
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFddUvisc_EulerNS(double U[nVars], double dFddU[nVars][nVars][nDims][nDims], 
    double gamma, double prandtl, double mu_in)
{
  /* Set viscosity */
  // TODO: Add Sutherland's law and store mu in array
  double invrho = 1.0 / U[0];
  double mu = mu_in;
  double diffCo1 = mu * invrho;
  double diffCo2 = gamma * mu * invrho / prandtl;

  if (nDims == 2)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Set viscous dFxddUx values */
    dFddU[1][0][0][0] = 4.0/3.0 * u * diffCo1;
    dFddU[2][0][0][0] = v * diffCo1;
    dFddU[3][0][0][0] = (4.0/3.0 * u*u + v*v) * diffCo1 - (u*u + v*v - e*invrho) * diffCo2;

    dFddU[1][1][0][0] = -4.0/3.0 * diffCo1;
    dFddU[3][1][0][0] = -u * (4.0/3.0 * diffCo1 - diffCo2);

    dFddU[2][2][0][0] = -diffCo1;
    dFddU[3][2][0][0] = -v * (diffCo1 - diffCo2);

    dFddU[3][3][0][0] = -diffCo2;

    /* Set viscous dFyddUx values */
    dFddU[1][0][1][0] = v * diffCo1;
    dFddU[2][0][1][0] = -2.0/3.0 * u * diffCo1;
    dFddU[3][0][1][0] = 1.0/3.0 * u * v * diffCo1;

    dFddU[2][1][1][0] = 2.0/3.0 * diffCo1;
    dFddU[3][1][1][0] = 2.0/3.0 * v * diffCo1;

    dFddU[1][2][1][0] = -diffCo1;
    dFddU[3][2][1][0] = -u * diffCo1;

    /* Set viscous dFxddUy values */
    dFddU[1][0][0][1] = -2.0/3.0 * v * diffCo1;
    dFddU[2][0][0][1] = u * diffCo1;
    dFddU[3][0][0][1] = 1.0/3.0 * u * v * diffCo1;

    dFddU[2][1][0][1] = -diffCo1;
    dFddU[3][1][0][1] = -v * diffCo1;

    dFddU[1][2][0][1] = 2.0/3.0 * diffCo1;
    dFddU[3][2][0][1] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFyddUy values */
    dFddU[1][0][1][1] = u * diffCo1;
    dFddU[2][0][1][1] = 4.0/3.0 * v * diffCo1;
    dFddU[3][0][1][1] = (u*u + 4.0/3.0 * v*v) * diffCo1 - (u*u + v*v - e*invrho) * diffCo2;

    dFddU[1][1][1][1] = -diffCo1;
    dFddU[3][1][1][1] = -u * (diffCo1 - diffCo2);

    dFddU[2][2][1][1] = -4.0/3.0 * diffCo1;
    dFddU[3][2][1][1] = -v * (4.0/3.0 * diffCo1 - diffCo2);

    dFddU[3][3][1][1] = -diffCo2;
  }
  else if (nDims == 3)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Set viscous dFxddUx values */
    dFddU[1][0][0][0] = 4.0/3.0 * u * diffCo1;
    dFddU[2][0][0][0] = v * diffCo1;
    dFddU[3][0][0][0] = w * diffCo1;
    dFddU[4][0][0][0] = (4.0/3.0 * u*u + v*v + w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFddU[1][1][0][0] = -4.0/3.0 * diffCo1;
    dFddU[4][1][0][0] = -u * (4.0/3.0 * diffCo1 - diffCo2);

    dFddU[2][2][0][0] = -diffCo1;
    dFddU[4][2][0][0] = -v * (diffCo1 - diffCo2);

    dFddU[3][3][0][0] = -diffCo1;
    dFddU[4][3][0][0] = -w * (diffCo1 - diffCo2);

    dFddU[4][4][0][0] = -diffCo2;

    /* Set viscous dFyddUx values */
    dFddU[1][0][1][0] = v * diffCo1;
    dFddU[2][0][1][0] = -2.0/3.0 * u * diffCo1;
    dFddU[4][0][1][0] = 1.0/3.0 * u * v * diffCo1;

    dFddU[2][1][1][0] = 2.0/3.0 * diffCo1;
    dFddU[4][1][1][0] = 2.0/3.0 * v * diffCo1;

    dFddU[1][2][1][0] = -diffCo1;
    dFddU[4][2][1][0] = -u * diffCo1;

    /* Set viscous dFzddUx values */
    dFddU[1][0][2][0] = w * diffCo1;
    dFddU[3][0][2][0] = -2.0/3.0 * u * diffCo1;
    dFddU[4][0][2][0] = 1.0/3.0 * u * w * diffCo1;

    dFddU[3][1][2][0] = 2.0/3.0 * diffCo1;
    dFddU[4][1][2][0] = 2.0/3.0 * w * diffCo1;

    dFddU[1][3][2][0] = -diffCo1;
    dFddU[4][3][2][0] = -u * diffCo1;

    /* Set viscous dFxddUy values */
    dFddU[1][0][0][1] = -2.0/3.0 * v * diffCo1;
    dFddU[2][0][0][1] = u * diffCo1;
    dFddU[4][0][0][1] = 1.0/3.0 * u * v * diffCo1;

    dFddU[2][1][0][1] = -diffCo1;
    dFddU[4][1][0][1] = -v * diffCo1;

    dFddU[1][2][0][1] = 2.0/3.0 * diffCo1;
    dFddU[4][2][0][1] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFyddUy values */
    dFddU[1][0][1][1] = u * diffCo1;
    dFddU[2][0][1][1] = 4.0/3.0 * v * diffCo1;
    dFddU[3][0][1][1] = w * diffCo1;
    dFddU[4][0][1][1] = (u*u + 4.0/3.0 * v*v + w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFddU[1][1][1][1] = -diffCo1;
    dFddU[4][1][1][1] = -u * (diffCo1 - diffCo2);

    dFddU[2][2][1][1] = -4.0/3.0 * diffCo1;
    dFddU[4][2][1][1] = -v * (4.0/3.0 * diffCo1 - diffCo2);

    dFddU[3][3][1][1] = -diffCo1;
    dFddU[4][3][1][1] = -w * (diffCo1 - diffCo2);

    dFddU[4][4][1][1] = -diffCo2;

    /* Set viscous dFzddUy values */
    dFddU[2][0][2][1] = w * diffCo1;
    dFddU[3][0][2][1] = -2.0/3.0 * v * diffCo1;
    dFddU[4][0][2][1] = 1.0/3.0 * v * w * diffCo1;

    dFddU[3][2][2][1] = 2.0/3.0 * diffCo1;
    dFddU[4][2][2][1] = 2.0/3.0 * w * diffCo1;

    dFddU[2][3][2][1] = -diffCo1;
    dFddU[4][3][2][1] = -v * diffCo1;

    /* Set viscous dFxddUz values */
    dFddU[1][0][0][2] = -2.0/3.0 * w * diffCo1;
    dFddU[3][0][0][2] = u * diffCo1;
    dFddU[4][0][0][2] = 1.0/3.0 * u * w * diffCo1;

    dFddU[3][1][0][2] = -diffCo1;
    dFddU[4][1][0][2] = -w * diffCo1;

    dFddU[1][3][0][2] = 2.0/3.0 * diffCo1;
    dFddU[4][3][0][2] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFyddUz values */
    dFddU[2][0][1][2] = -2.0/3.0 * w * diffCo1;
    dFddU[3][0][1][2] = v * diffCo1;
    dFddU[4][0][1][2] = 1.0/3.0 * v * w * diffCo1;

    dFddU[3][2][1][2] = -diffCo1;
    dFddU[4][2][1][2] = -w * diffCo1;

    dFddU[2][3][1][2] = 2.0/3.0 * diffCo1;
    dFddU[4][3][1][2] = 2.0/3.0 * v * diffCo1;

    /* Set viscous dFzddUz values */
    dFddU[1][0][2][2] = u * diffCo1;
    dFddU[2][0][2][2] = v * diffCo1;
    dFddU[3][0][2][2] = 4.0/3.0 * w * diffCo1;
    dFddU[4][0][2][2] = (u*u + v*v + 4.0/3.0*w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFddU[1][1][2][2] = -diffCo1;
    dFddU[4][1][2][2] = -u * (diffCo1 - diffCo2);

    dFddU[2][2][2][2] = -diffCo1;
    dFddU[4][2][2][2] = -v * (diffCo1 - diffCo2);

    dFddU[3][3][2][2] = -4.0/3.0 * diffCo1;
    dFddU[4][3][2][2] = -w * (4.0/3.0 * diffCo1 - diffCo2);

    dFddU[4][4][2][2] = -diffCo2;
  }
}
