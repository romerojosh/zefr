#ifndef polynomials_hpp
#define polynomials_hpp

#include <vector>

/*! Evaluates the Lagrange function corresponding to the specified mode on xiGrid at location xi.
 *
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Lagrange function at xi.
 */
static inline
double Lagrange(const std::vector<double> &xiGrid, unsigned int mode, double xi)
{
  double val = 1.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  for (unsigned int i = 0; i < mode; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  for (unsigned int i = mode + 1; i < npts; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

/*! Because Jacob is too lazy to convert all his copied code from Flurry */
static inline
double Lagrange(const std::vector<double> &xiGrid, double xi, unsigned int mode)
{
  return Lagrange(xiGrid, mode, xi);
}

/*! Evaluates the first derivative of the Lagrange function corresponding to the specified mode
 *  on xiGrid at location xi.
 *
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 *
 * \return Value of first derivative of the Lagrange function at xi.
 */
static inline
double Lagrange_d1(const std::vector<double>& xiGrid, unsigned int mode, double xi)
{
  double val = 0.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  /* Compute normalization constant */
  double den = 1.0;
  for (unsigned int i = 0; i < mode; i++)
    den *= (xiGrid[mode] - xiGrid[i]);

  for (unsigned int i = mode+1; i < npts; i++)
    den *= (xiGrid[mode] - xiGrid[i]);

  /* Compute sum of products */
  for (unsigned int j = 0; j < npts; j++)
  {
    if (j == mode)
      continue;

    double term = 1.0;
    for (unsigned int i = 0; i < npts; i++)
      if (i != mode and i != j)
        term *= (xi - xiGrid[i]);

    val += term;
  }

  return val/den;
}

/*! Evaluates the Legendre polynomial of degree P at location xi.
 *
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Legendre polynomial at xi.
 */
double Legendre(unsigned int P, double xi);

/*! Evaluates the first derivative of the Legendre polynomial of degree P at location xi.
 *
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of derivative of Legendre polynomial at xi.
 */
double Legendre_d1(unsigned int P, double xi);

//! Multi-dimensional Legendre polynomials
double Legendre2D(unsigned int P, double xi, double eta, unsigned int mode);
double Legendre3D(unsigned int P, double xi, double eta, double mu, unsigned int mode);

double Jacobi(double xi, double a, double b, unsigned int mode);
double dJacobi(double xi, double a, double b, unsigned int mode);
double Dubiner2D(unsigned int P, double xi, double eta, unsigned int mode);
double dDubiner2D(unsigned int P, double xi, double eta, double dim, unsigned int mode);
double RTMonomial2D(unsigned int P, double xi, double eta, unsigned int dim, unsigned int mode);
double divRTMonomial2D(unsigned int P, double xi, double eta, unsigned int mode);

double Dubiner3D(unsigned int P, double xi, double eta, double zeta, unsigned int mode);
double dDubiner3D(unsigned int P, double xi, double eta, double zeta, double dim, unsigned int mode);
double RTMonomial3D(unsigned int P, double xi, double eta, double zeta, unsigned int dim, unsigned int mode);
double divRTMonomial3D(unsigned int P, double xi, double eta, double zeta, unsigned int mode);


#endif /* polynomials_hpp */
