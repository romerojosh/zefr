template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fconv_AdvDiff(double U[nVars], double F[nVars][nDims], double A[nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    F[0][dim] = A[dim] * U[0];
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__
#endif
void compute_Fconv_Burgers(double U[nVars], double F[nVars][nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    F[0][dim] = 0.5 * U[0] * U[0];
  }
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
