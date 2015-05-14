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
}

void Elements::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nSpts, nNodes},1);
  shape_fpts.assign({nFpts, nNodes},1);
  shape_ppts.assign({nPpts, nNodes},1);
  shape_qpts.assign({nQpts, nNodes},1);
  dshape_spts.assign({nDims, nSpts, nNodes},1);
  dshape_fpts.assign({nDims, nFpts, nNodes},1);
  dshape_ppts.assign({nDims, nPpts, nNodes},1);
  dshape_qpts.assign({nDims, nQpts, nNodes},1);


  std::vector<double> loc(nDims,0.0);

  /* Shape functions and derivatives at solution points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_spts(spt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(spt,node) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_spts(dim,spt,node) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_fpts(fpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(fpt,node) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_fpts(dim,fpt,node) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_ppts(ppt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(ppt,node) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_ppts(dim,ppt,node) = calc_d_shape(shape_order, node, loc, dim);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(qpt,node) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_qpts(dim,qpt,node) = calc_d_shape(shape_order, node, loc, dim);
    }
  }
}

void Elements::set_coords()
{
  /* Allocate memory for physical coordinates */
  geo->coord_spts.assign({nDims, nEles, nSpts});
  geo->coord_fpts.assign({nDims, nEles, nFpts});
  geo->coord_ppts.assign({nDims, nEles, nPpts});
  geo->coord_qpts.assign({nDims, nEles, nQpts});

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {

      /* Setup physical coordinates at solution points */
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_spts(dim, ele, spt) += geo->coord_nodes(gnd,dim) * shape_spts(spt, node);
        }
      }
  
      /* Setup physical coordinates at flux points */
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_fpts(dim, ele, fpt) += geo->coord_nodes(gnd,dim) * shape_fpts(fpt, node);
        }
      }

      /* Setup physical coordinates at plot points */
      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_ppts(dim, ele, ppt) += geo->coord_nodes(gnd,dim) * shape_ppts(ppt, node);
        }
      }

      /* Setup physical coordinates at quadrature points */
      for (unsigned int qpt = 0; qpt < nQpts; qpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_qpts(dim, ele, qpt) += geo->coord_nodes(gnd,dim) * shape_qpts(qpt, node);
        }
      }

    }
  }
}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nSpts, nFpts});
  oppD.assign({nDims, nSpts, nSpts});
  oppD_fpts.assign({nDims, nFpts, nSpts});

  std::vector<double> loc(nDims, 0.0);
  /* Setup spt to fpt extrapolation operator (oppE) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      loc[0] = loc_fpts(fpt,0);
      loc[1] = loc_fpts(fpt,1);

      oppE(spt,fpt) = calc_nodal_basis(spt, loc);
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

        oppD(dim,jspt,ispt) = calc_d_nodal_basis_spts(jspt, loc, dim);
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
        oppD_fpts(dim,fpt,spt) = calc_d_nodal_basis_fpts(fpt, loc, dim);
      }
    }
  }

}

void Elements::setup_aux()
{
  /* Allocate memory for plot point and quadrature point interpolation operator */
  oppE_ppts.assign({nSpts, nPpts});
  oppE_qpts.assign({nSpts, nQpts});

  std::vector<double> loc(nDims, 0.0);

  /* Setup spt to ppt extrapolation operator (oppE_ppts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      loc[0] = loc_ppts(ppt,0);
      loc[1] = loc_ppts(ppt,1);

      oppE_ppts(spt,ppt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Setup spt to qpt extrapolation operator (oppE_qpts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      loc[0] = loc_qpts(qpt,0);
      loc[1] = loc_qpts(qpt,1);

      oppE_qpts(spt,qpt) = calc_nodal_basis(spt, loc);
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
            F_spts(dim, n, ele, spt) = input->AdvDiff_A[dim] * U_spts(n, ele, spt);
          }
        }
      }
    }
  }

  else if (input->equation == "EulerNS")
  {
    ThrowException("Euler flux not implemented yet!");
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
            F_spts(dim, n, ele, spt) += -input->AdvDiff_D * dU_spts(dim, n, ele, spt);
          }
        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("NS flux not implemented yet!");
  }

}
