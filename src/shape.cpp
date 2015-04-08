#include "polynomials.hpp"

double calc_shape_quad(unsigned int shape_order, unsigned int idx,
                       double xi, double eta)
{
  double val;

  /* Bilinear quadrilateral/4-node Serendipity */
  if (shape_order == 1)
  {
    double i, j;

    switch(idx)
    {
      case 0:
        i = 0; j = 0; break;
      case 1:
        i = 1; j = 0; break;
      case 2:
        i = 1; j = 1; break;
      case 3:
        i = 0; j = 1; break;
    }

    val = Lagrange({-1,1}, i, xi) * Lagrange({-1,1}, j, eta);
  }

  /* 8-node Serendipity Element */
  if (shape_order == 2)
  {
    switch(idx)
    {
      case 0:
        val = -0.25*(1.-xi)*(1.-eta)*(1.+eta+xi); break;
      case 1:
        val = 0.5*(1.-xi)*(1.+xi)*(1.-eta); break;
      case 2:
        val = -0.25*(1.+xi)*(1.-eta)*(1.+eta-xi); break;
      case 3:
        val = 0.5*(1.+xi)*(1.+eta)*(1.-eta); break;
      case 4:
        val = -0.25*(1.+xi)*(1.+eta)*(1.-eta-xi); break;
      case 5:
        val = 0.5*(1.-xi)*(1.+xi)*(1.+eta); break;
      case 6:
        val = -0.25*(1.-xi)*(1.+eta)*(1.-eta+xi); break;
      case 7:
        val = 0.5*(1.-xi)*(1.+eta)*(1.-eta); break;
    }
  }

  return val;
}

double calc_dshape_quad(unsigned int shape_order, unsigned int idx,
                        double xi, double eta, unsigned int dim)
{
  double val;

  /* Bilinear quadrilateral/4-node Serendipity */
  if (shape_order == 1)
  {
    double i, j;

    switch(idx)
    {
      case 0:
        i = 0; j = 0; break;
      case 1:
        i = 1; j = 0; break;
      case 2:
        i = 1; j = 1; break;
      case 3:
        i = 0; j = 1; break;
    }

    if (dim == 0)
      val = Lagrange_d1({-1,1}, i, xi) * Lagrange({-1,1}, j, eta);
    else
      val = Lagrange({-1,1}, i, xi) * Lagrange_d1({-1,1}, j, eta);
  }

  /* 8-node Serendipity Element */
  else if (shape_order == 2)
  {
    if (dim == 0)
    {
      switch(idx)
      {
        case 0:
          val = -0.25*(-1.+eta)*(2.*xi+eta); break;
        case 1:
          val = xi*(-1.+eta); break;
        case 2:
          val = 0.25*(-1.+eta)*(eta - 2.*xi); break;
        case 3:
          val = -0.5*(1+eta)*(-1.+eta); break;
        case 4:
          val = 0.25*(1.+eta)*(2.*xi+eta); break;
        case 5:
          val = -xi*(1.+eta); break;
        case 6:
          val = -0.25*(1.+eta)*(eta-2.*xi); break;
        case 7:
          val = 0.5*(1+eta)*(-1.+eta); break;
      }
    }

    else if (dim == 1)
    {
      switch(idx)
      {
        case 0:
          val = -0.25*(-1.+xi)*(2.*eta+xi); break;
        case 1:
          val = 0.5*(1.+xi)*(-1.+xi); break;
        case 2:
          val = 0.25*(1.+xi)*(2.*eta - xi); break;
        case 3:
          val = -eta*(1.+xi); break;
        case 4:
          val = 0.25*(1.+xi)*(2.*eta+xi); break;
        case 5:
          val = -0.5*(1.+xi)*(-1.+xi); break;
        case 6:
          val = -0.25*(-1.+xi)*(2.*eta-xi); break;
        case 7:
          val = eta*(-1.+xi); break;
      }
    }

  }

  return val;

}
