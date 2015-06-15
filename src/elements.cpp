#include <iostream>
#include <memory>
#include <string>

#include "elements.hpp"
#include "faces.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

void Elements::associate_faces(std::shared_ptr<Faces> faces)
{
  this->faces = faces;
  this->faces->setup(nDims, nVars);
}

void Elements::setup()
{
  set_locs();
  set_shape();
  set_transforms();
  set_normals();
  setup_FR();
  setup_aux();
  set_coords();

  /* If P-multigrid is enabled, create associated operators */
  if (input->p_multi)
    setup_PMG();
}

void Elements::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nNodes, nSpts},1);
  shape_fpts.assign({nNodes, nFpts},1);
  shape_ppts.assign({nNodes, nPpts},1);
  shape_qpts.assign({nNodes, nQpts},1);
  dshape_spts.assign({nNodes, nSpts, nDims},1);
  dshape_fpts.assign({nNodes, nFpts, nDims},1);
  dshape_ppts.assign({nNodes, nPpts, nDims},1);
  dshape_qpts.assign({nNodes, nQpts, nDims},1);


  std::vector<double> loc(nDims,0.0);

  /* Shape functions and derivatives at solution points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_spts(spt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(node,spt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_spts(node,spt,dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_fpts(fpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(node, fpt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_fpts(node, fpt, dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_ppts(ppt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(node,ppt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_ppts(node,ppt,dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(node,qpt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_qpts(node,qpt,dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }
}

void Elements::set_coords()
{
  /* Allocate memory for physical coordinates */
  geo->coord_spts.assign({nSpts, nEles, nDims});
  geo->coord_fpts.assign({nFpts, nEles, nDims});
  geo->coord_ppts.assign({nPpts, nEles, nDims});
  geo->coord_qpts.assign({nQpts, nEles, nDims});

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {

      /* Setup physical coordinates at solution points */
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_spts(spt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_spts(node, spt);
        }
      }
  
      /* Setup physical coordinates at flux points */
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_fpts(fpt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_fpts(node, fpt);
        }
      }

      /* Setup physical coordinates at plot points */
      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_ppts(ppt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_ppts(node, ppt);
        }
      }

      /* Setup physical coordinates at quadrature points */
      for (unsigned int qpt = 0; qpt < nQpts; qpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_qpts(qpt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_qpts(node, qpt);
        }
      }

    }
  }
}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppD.assign({nSpts, nSpts, nDims});
  oppD_fpts.assign({nSpts, nFpts, nDims});

  std::vector<double> loc(nDims, 0.0);
  /* Setup spt to fpt extrapolation operator (oppE) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      loc[0] = loc_fpts(fpt,0);
      loc[1] = loc_fpts(fpt,1);

      oppE(fpt,spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Setup differentiation operator (oppD) for solution points */
  /* Note: Can set up for standard FR eventually. Trying to keep things simple.. */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int jspt = 0; jspt < nSpts; jspt++)
    {
      for (unsigned int ispt = 0; ispt < nSpts; ispt++)
      {
        loc[0] = loc_spts(ispt,0);
        loc[1] = loc_spts(ispt,1);

        oppD(ispt,jspt,dim) = calc_d_nodal_basis_spts(jspt, loc, dim);
      }
    }
  }

  /* Setup differentiation operator (oppD_fpts) for flux points (DFR Specific)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        loc[0] = loc_spts(spt,0);
        loc[1] = loc_spts(spt,1);
        oppD_fpts(spt,fpt,dim) = calc_d_nodal_basis_fpts(fpt, loc, dim);
      }
    }
  }

}

void Elements::setup_aux()
{
  /* Allocate memory for plot point and quadrature point interpolation operator */
  oppE_ppts.assign({nPpts, nSpts});
  oppE_qpts.assign({nQpts, nSpts});

  std::vector<double> loc(nDims, 0.0);

  /* Setup spt to ppt extrapolation operator (oppE_ppts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      loc[0] = loc_ppts(ppt,0);
      loc[1] = loc_ppts(ppt,1);

      oppE_ppts(ppt, spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Setup spt to qpt extrapolation operator (oppE_qpts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      loc[0] = loc_qpts(qpt,0);
      loc[1] = loc_qpts(qpt,1);

      oppE_qpts(qpt,spt) = calc_nodal_basis(spt, loc);
    }
  }

}

void Elements::compute_Fconv()
{
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            F_spts(spt, ele, n, dim) = input->AdvDiff_A[dim] * U_spts(spt, ele, n);
          }
        }
      }
    }
  }

  else if (input->equation == "EulerNS")
  {
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Compute some primitive variables */
          double momF = (U_spts(spt, ele, 1) * U_spts(spt,ele,1) + U_spts(spt, ele, 2) * 
              U_spts(spt, ele,2)) / U_spts(spt, ele, 0);
          double P = (input->gamma - 1.0) * (U_spts(spt, ele, 3) - 0.5 * momF);
          double H = (U_spts(spt, ele, 3) + P) / U_spts(spt, ele, 0);


          F_spts(spt, ele, 0, 0) = U_spts(spt, ele, 1);
          F_spts(spt, ele, 1, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 2, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 3, 0) = U_spts(spt, ele, 1) * H;

          F_spts(spt, ele, 0, 1) = U_spts(spt, ele, 2);
          F_spts(spt, ele, 1, 1) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 2, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 3, 1) = U_spts(spt, ele, 2) * H;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("3D Euler flux not implemented!");
    }
  }

}

void Elements::compute_Fvisc()
{
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            /* Can just add viscous flux to existing convective flux */
            F_spts(spt, ele, n, dim) += -input->AdvDiff_D * dU_spts(spt, ele, n, dim);
          }
        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {

    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        /* Setting variables for convenience */
        double rho = U_spts(spt, ele, 0);
        double momx = U_spts(spt, ele, 1);
        double momy = U_spts(spt, ele, 2);
        double e = U_spts(spt, ele, 3);

        double u = mom_x / rho;
        double v = mom_y / rho;
        double e_int = e / rho - 0.5 * (u*u + v*v);

        double rho_dx = dU_spts(spt, ele, 0, 0);
        double momx_dx = dU_spts(spt, ele, 1, 0);
        double momy_dx = dU_spts(spt, ele, 2, 0);
        double e_dx = dU_spts(spt, ele, 3, 0);
        
        double rho_dy = dU_spts(spt, ele, 0, 1);
        double momx_dy = dU_spts(spt, ele, 1, 1);
        double momy_dy = dU_spts(spt, ele, 2, 1);
        double e_dy = dU_spts(spt, ele, 3, 1);

        double rt_ratio = (input->gamma - 1.0) * U_spts(spt, ele, 3) / input-> rt_inf;
        double mu = (input->mu_inf) * std::pow(rt_ratio, 1.5) * (1.0 + input->c_sth) / 
          (rt_ratio + input->c_sth);
        mu += input->fix_vis * (input->mu_inf - mu);

        double du_dx = (mom_x_dx - rho_dx * u) / rho;
        double du_dy = (mom_x_dy - rho_dy * u) / rho;

        double dv_dx = (mom_y_dx - rho_dx * v) / rho;
        double dv_dy = (mom_y_dy - rho_dy * v) / rho;

        double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
        double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

        double de_dx = (e_dx - dke_dx - rho_dx * e_int) / rho;
        double de_dy = (e_dy - dke_dy - rho_dy * e_int) / rho;

        double diag = (du_dx + dv_dy) / 3.0;

        double tauxx = 2.0 * mu * (du_dx - diag);
        double tauxy = mu * (du_dy + dv_dx);
        double tauyy = 2.0 * mu * (dv_dy - diag);

        F_spts(spt, ele, 1, 0) -= tauxx;
        F_spts(spt, ele, 2, 0) -= tauxy;
        F_spts(spt, ele, 3, 0) -= (u * tauxy + v * tauyy + (mu / input->prandtl)) *
            input-> gamma * de_dx;

        F_spts(spt, ele, 1, 1) += -tauxy;
        F_spts(spt, ele, 2, 1) += -tauyy;
        F_spts(spt, ele, 3, 1) -= (u * tauxy + v * tauyy + (mu / input->prandtl)) *
            input->gamma * de_dy;
      }
    }
  }

}
