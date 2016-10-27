template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fconv_AdvDiff(double U[nVars], double F[nVars][nDims], double A[nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    F[0][dim] = A[dim] * U[0];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fconv_Burgers(double U[nVars], double F[nVars][nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    F[0][dim] = 0.5 * U[0] * U[0];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fconv_EulerNS(double U[nVars], double F[nVars][nDims], double &P, double gamma)
{
  if (nDims == 2)
  {
    double invrho = 1./U[0];
    double momF = (U[1] * U[1] + U[2] * U[2]) * invrho;

    P = (gamma - 1.0) * (U[3] - 0.5 * momF);

    double H = (U[3] + P) * invrho;

    F[0][0] = U[1];
    F[1][0] = U[1] * U[1] * invrho + P;
    F[2][0] = U[1] * U[2] * invrho;
    F[3][0] = U[1] * H;

    F[0][1] = U[2];
    F[1][1] = U[2] * U[1] * invrho;
    F[2][1] = U[2] * U[2] * invrho + P;
    F[3][1] = U[2] * H;
  }
  else if (nDims == 3)
  {
    double invrho = 1./U[0];
    double momF = (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) * invrho;

    P = (gamma - 1.0) * (U[4] - 0.5 * momF);

    double H = (U[4] + P) * invrho; 

    F[0][0] = U[1];
    F[1][0] = U[1] * U[1] * invrho + P;
    F[2][0] = U[1] * U[2] * invrho;
    F[3][0] = U[1] * U[3] * invrho;
    F[4][0] = U[1] * H;

    F[0][1] = U[2];
    F[1][1] = U[2] * U[1] * invrho;
    F[2][1] = U[2] * U[2] * invrho + P;
    F[3][1] = U[2] * U[3] * invrho;
    F[4][1] = U[2] * H;

    F[0][2] = U[3];
    F[1][2] = U[3] * U[1] * invrho;
    F[2][2] = U[3] * U[2] * invrho;
    F[3][2] = U[3] * U[3] * invrho + P;
    F[4][2] = U[3] * H;
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fvisc_AdvDiff_add(double dU[nVars][nDims], double F[nVars][nDims], double D)
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    F[0][dim] -= D * dU[0][dim];
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
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
