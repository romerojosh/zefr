#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>

#include "cblas.h"

#include "elements.hpp"
#include "faces.hpp"
#include "funcs.hpp"
#include "geometry.hpp"
#include "hexas.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#include "metis.h"

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#endif

//FRSolver::FRSolver(const InputStruct *input, int order)
FRSolver::FRSolver(InputStruct *input, int order, bool FV_mode)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;

  this->FV_mode = FV_mode;
}

void FRSolver::setup()
{
  if (input->rank == 0) std::cout << "Reading mesh: " << input->meshfile << std::endl;
  geo = process_mesh(input, order, input->nDims);

  if (input->rank == 0) std::cout << "Setting up elements and faces..." << std::endl;

  if (input->nDims == 2)
    eles = std::make_shared<Quads>(&geo, input, order);
  else if (input->nDims == 3)
    eles = std::make_shared<Hexas>(&geo, input, order);

  faces = std::make_shared<Faces>(&geo, input);

  faces->setup(eles->nDims, eles->nVars);
  eles->setup(faces);

  if (input->rank == 0) std::cout << "Setting up timestepping..." << std::endl;
  setup_update();

  if (input->rank == 0) std::cout << "Setting up output..." << std::endl;
  setup_output();

  if (input->rank == 0) std::cout << "Initializing solution..." << std::endl;
  initialize_U();

  if (FV_mode)
  {
    if (input->rank == 0) std::cout << "Setting up H levels..." << std::endl;
    setup_h_levels();
    write_partition_file();
  }

  if (input->restart)
  {
    if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_file +" ..." << std::endl;
    restart(input->restart_file);
  }

  
#ifdef _GPU
  solver_data_to_device();
#endif

}

void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  if (input->dt_scheme == "Euler")
  {
    nStages = 1;
    rk_beta.assign({nStages},1.0);

  }
  else if (input->dt_scheme == "RK44")
  {
    nStages = 4;
    
    rk_alpha.assign({nStages-1});
    rk_alpha(0) = 0.5; rk_alpha(1) = 0.5; rk_alpha(2) = 1.0;

    rk_beta.assign({nStages});
    rk_beta(0) = 1./6.; rk_beta(1) = 1./3.; 
    rk_beta(2) = 1./3.; rk_beta(3) = 1./6.;
  }
  else if (input->dt_scheme == "RK54")
  {
    nStages = 5;

    rk_alpha.assign({nStages-1});
    rk_alpha(0) = -0.417890474499852; rk_alpha(1) = -1.192151694642677; 
    rk_alpha(2) = -1.697784692471528; rk_alpha(3) = -1.514183444257156;

    rk_beta.assign({nStages});
    rk_beta(0) = 0.149659021999229; rk_beta(1) = 0.379210312999627; 
    rk_beta(3) = 0.822955029386982; rk_beta(3) = 0.699450455949122;
    rk_beta(4) = 0.153057247968152;
  }
  else if (input->dt_scheme == "RKj")
  {
    nStages = 4;
    rk_alpha.assign({nStages});
    rk_alpha(0) = 1./4.; rk_alpha(1) = 1./3.; rk_alpha(2) = 1./2.; rk_alpha(3) = 1.0;
    //rk_alpha(0) = 1./4.; rk_alpha(1) = 1./2.; rk_alpha(2) = 0.55; rk_alpha(3) = 1.0;
    /*
    rk_alpha(0) = 0.038631946268902;
    rk_alpha(1) = 0.279767066975738;
    rk_alpha(2) = 0.613275407706588;
    rk_alpha(3) = 1.0;
    */
  }
  else
  {
    ThrowException("dt_scheme not recognized!");
  }

  U_ini.assign({eles->nSpts, eles->nEles, eles->nVars});
  dt.assign({eles->nEles},input->dt);
  if (FV_mode)
    dt_part.assign({eles->nEles},input->dt);

}

void FRSolver::setup_output()
{
  /* Create output directory to store data files */
  if (input->rank == 0)
  {
    std::string cmd = "mkdir " + input->output_prefix;
    system(cmd.c_str());
  }

  if (eles->nDims == 2)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 4;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({4, eles->nSubelements});

    /* Setup plot "subelement" connectivity */
    std::vector<unsigned int> nd(4,0);

    unsigned int ele = 0;
    nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;

    for (unsigned int i = 0; i < nSubelements1D; i++)
    {
      for (unsigned int j = 0; j < nSubelements1D; j++)
      {
        for (unsigned int node = 0; node < 4; node ++)
        {
          geo.ppt_connect(node, ele) = nd[node] + j;
        }

        ele++;
      }

      for (unsigned int node = 0; node < 4; node ++)
        nd[node] += nSubelements1D + 1;
    }
  }
  else if (eles->nDims == 3)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 8;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({8, eles->nSubelements});

    /* Setup plot "subelement" connectivity */
    std::vector<unsigned int> nd(8,0);

    unsigned int ele = 0;
    nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;
    nd[4] = (nSubelements1D + 1) * (nSubelements1D + 1); nd[5] = nd[4] + 1; 
    nd[6] = nd[4] + nSubelements1D + 2; nd[7] = nd[4] + nSubelements1D + 1;

    for (unsigned int i = 0; i < nSubelements1D; i++)
    {
      for (unsigned int j = 0; j < nSubelements1D; j++)
      {
        for (unsigned int k = 0; k < nSubelements1D; k++)
        {
          for (unsigned int node = 0; node < 8; node ++)
          {
            geo.ppt_connect(node, ele) = nd[node] + k;
          }

          ele++;
        }

        for (unsigned int node = 0; node < 8; node ++)
          nd[node] += (nSubelements1D + 1);

      }

      for (unsigned int node = 0; node < 8; node ++)
        nd[node] += (nSubelements1D + 1);
    }

  }

}

void FRSolver::setup_h_levels()
{
  //Allocate space for partition average storage, eparts
  eparts.assign({eles->nEles, input->hmg_levels});
  U_avg.assign({eles->nEles, eles->nVars});
  vols.resize(input->hmg_levels);

  //Setup partitions using METIS
  /* Setup METIS */
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_MINCONN] = 1;
  options[METIS_OPTION_CONTIG] = 1;
  options[METIS_OPTION_NCUTS] = 5;
  options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;


  /* Form eptr and eind arrays */
  std::vector<int> eptr(geo.nEles + 1); 
  std::vector<int> eind(geo.nEles * geo.nCornerNodes); 
  std::set<unsigned int> nodes;

  int n = 0;
  eptr[0] = 0;
  for (unsigned int i = 0; i < geo.nEles; i++)
  {
    for (unsigned int j = 0; j < geo.nCornerNodes;  j++)
    {
      eind[j + n] = geo.nd2gnd(j, i);
      nodes.insert(geo.nd2gnd(j,i));
    } 

    /* Check for collapsed edge (not fully general yet)*/
    if (nodes.size() < geo.nCornerNodes)
    {
      n += geo.nCornerNodes - 1;
    }
    else
    {
      n += geo.nCornerNodes;
    }
    eptr[i + 1] = n;
    nodes.clear();
  }

  int scale_fac = 1 << eles->nDims;

  for (unsigned int H = 0; H < input->hmg_levels; H++)
  {
    int nPartitions = eles->nEles / (unsigned int) std::pow(2, eles->nDims*(H + 1));
    //int nPartitions = eles->nEles / (unsigned int) std::pow(2, H + 1);
    std::cout << nPartitions << std::endl;
    //int nPartitions = eles->nEles / (unsigned int) std::pow(2, H);
    vols[H].assign(nPartitions, 0);

    /* Coarsening using METIS (not too great) */
    if (input->coarse_mode == 0)
    {
      int objval;
      std::vector<int> npart(geo.nNodes);
      int nNodesPerFace = geo.nNodesPerFace; // TODO: What should this be?
      int nEles = geo.nEles;
      int nNodes = geo.nNodes;

      METIS_PartMeshDual(&nEles, &nNodes, eptr.data(), eind.data(), nullptr, 
          nullptr, &nNodesPerFace, &nPartitions, nullptr, options, &objval, &eparts(0,H), 
          npart.data());  


    }
    /* Coarsening using structured blocking (better, but specific) */
    else if (input->coarse_mode == 1)
    {
      if (eles->nDims != 2)
        ThrowException("Structured coarsening not supported in 3D yet!");
      unsigned int nElesX = input->nElesX;
      unsigned int nElesY = input->nElesY;
      unsigned int nSegmentsX = nElesX / (1 << (H+1));
      unsigned int nSegmentsY = nElesY / (1 << (H+1));
      unsigned int segmentWidthX = nElesX / nSegmentsX;
      unsigned int segmentWidthY = nElesY / nSegmentsY;
      unsigned int part = 0;

      for (unsigned int I = 0; I < nSegmentsX; I++)
      {
        for (unsigned int J = 0; J < nSegmentsY; J++)
        {
          for (unsigned int i = 0; i < segmentWidthX; i++)
          {
            for (unsigned int j = 0; j < segmentWidthY; j++)
            {
              eparts((I*segmentWidthX + i) * nElesY + (J*segmentWidthY +j), H) = part;
            }
          }
          part++;
        }
      }
    }
    else
    {
      ThrowException("coarse_mode not recognized!");
    }

    double sum = 0;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      //TESTING:
      //eparts(ele, H) = ele;
      vols[H][eparts(ele, H)] += scale_fac * eles->jaco_det_spts(0, ele);
      sum += scale_fac * eles->jaco_det_spts(0,ele);
    }
    std::cout << sum << std::endl;

  }
  
  
}

void FRSolver::restart(std::string restart_file)
{
  size_t pos;
#ifdef _MPI
  /* From .pvtu, form partition specific filename */
  pos = restart_file.rfind(".pvtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .pvtu file for parallel restart!");
  }

  restart_file = restart_file.substr(0, pos);

  std::stringstream ss;
  ss << std::setw(3) << std::setfill('0') << input->rank;

  restart_file += "_" + ss.str() + ".vtu";
#endif

  /* Open .vtu file */
  std::ifstream f(restart_file);
  pos = restart_file.rfind(".vtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .vtu file for restart!");
  }

  if (!f.is_open())
  {
    ThrowException("Could not open restart file " + restart_file + "!");
  }

  std::string param, line;
  double val;
  unsigned int order_restart;
  mdvector<double> U_restart, oppRestart;

  /* Load data from restart file */
  while (f >> param)
  {
    if (param == "TIME")
    {
      f >> flow_time;
    }
    if (param == "ITER")
    {
      f >> current_iter;
    }
    if (param == "ORDER")
    {
      f >> order_restart;
    }

    if (param == "<PointData>")
    {

      unsigned int nSpts1D_restart = order_restart + 1;
      unsigned int nSpts2D_restart = nSpts1D_restart * nSpts1D_restart;
      unsigned int nPpts1D = nSpts1D_restart + 2;
      unsigned int nPpts2D = nPpts1D * nPpts1D;

      unsigned int nSpts_restart = (unsigned int) std::pow(nSpts1D_restart, input->nDims);
      unsigned int nPpts = (unsigned int) std::pow(nPpts1D, input->nDims);

      U_restart.assign({nSpts_restart, eles->nEles, eles->nVars});

      /* Setup extrapolation operator from restart points */
      oppRestart.assign({eles->nSpts, nSpts_restart});
      auto loc_spts_restart_1D = Gauss_Legendre_pts(order_restart + 1); 

      std::vector<double> loc(input->nDims);
      for (unsigned int rpt = 0; rpt < nSpts_restart; rpt++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          for (unsigned int dim = 0; dim < input->nDims; dim++)
            loc[dim] = eles->loc_spts(spt , dim);

          if (input->nDims == 2)
          {
            int i = rpt % nSpts1D_restart;
            int j = rpt / nSpts1D_restart;
            oppRestart(spt,rpt) = Lagrange(loc_spts_restart_1D, i, loc[0]) * 
                                  Lagrange(loc_spts_restart_1D, j, loc[1]);
          }
          else
          {
            int i = rpt % nSpts1D_restart;
            int j = (rpt / nSpts1D_restart) % nSpts1D_restart;
            int k = rpt / nSpts2D_restart;
            oppRestart(spt,rpt) = Lagrange(loc_spts_restart_1D, i, loc[0]) * 
                                  Lagrange(loc_spts_restart_1D, j, loc[1]) *
                                  Lagrange(loc_spts_restart_1D, k, loc[2]);
          }
        }
      }

      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        std::getline(f,line);
        std::getline(f,line);

        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          unsigned int spt = 0;
          for (unsigned int ppt = 0; ppt < nPpts; ppt++)
          {
            f >> val;

            /* Logic to deal with extra plot point (corner nodes and flux points). */
            if (input->nDims == 2)
            {
              if (ppt < nPpts1D || ppt > nPpts1D * (nPpts1D-1) || ppt%nPpts1D == 0 || 
                  (ppt+1)%nPpts1D == 0)
                continue;
            }
            else
            {
              int shift = (ppt / nPpts2D) * nPpts2D;
              if (ppt < nPpts2D || ppt < nPpts1D + shift || ppt > nPpts1D * (nPpts1D-1) + shift || 
                  (ppt-shift) % nPpts1D == 0 || (ppt+1-shift)%nPpts1D == 0 || ppt > nPpts2D * (nPpts2D - 1))
                continue;
            }

            U_restart(spt, ele, n) = val;
            spt++;
          }
        }
        std::getline(f,line);
      }

      /* Extrapolate values from restart points to solution points */
      auto &A = oppRestart(0, 0);
      auto &B = U_restart(0, 0, 0);
      auto &C = eles->U_spts(0, 0, 0);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nSpts_restart, 1.0, &A, eles->nSpts, &B, 
          nSpts_restart, 0.0, &C, eles->nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nSpts_restart, 1.0, &A, eles->nSpts, &B, 
          nSpts_restart, 0.0, &C, eles->nSpts);
#endif

    }
  }

  f.close();
}

#ifdef _GPU
void FRSolver::solver_data_to_device()
{
  /* Initial copy of data to GPU. Assignment operator will allocate data on device when first
   * used. */

  /* FR operators */
  eles->oppE_d = eles->oppE;
  eles->oppD_d = eles->oppD;
  eles->oppD_fpts_d = eles->oppD_fpts;

  /* If using multigrid, copy relevant operators */
  if (input->p_multi)
  {
    eles->oppPro_d = eles->oppPro;
    eles->oppRes_d = eles->oppRes;
  }

  /* Solver data structures */
  U_ini_d = U_ini;
  rk_alpha_d = rk_alpha;
  rk_beta_d = rk_beta;
  dt_d = dt;

  /* Solution data structures (element local) */
  eles->U_spts_d = eles->U_spts;
  eles->U_fpts_d = eles->U_fpts;
  eles->Ucomm_d = eles->Ucomm;
  eles->Uavg_d = eles->Uavg;
  eles->weights_spts_d = eles->weights_spts;
  eles->dU_spts_d = eles->dU_spts;
  eles->dU_fpts_d = eles->dU_fpts;
  eles->Fcomm_d = eles->Fcomm;
  eles->F_spts_d = eles->F_spts;
  eles->dF_spts_d = eles->dF_spts;
  eles->divF_spts_d = eles->divF_spts;
  eles->jaco_spts_d = eles->jaco_spts;
  eles->inv_jaco_spts_d = eles->inv_jaco_spts;
  eles->jaco_det_spts_d = eles->jaco_det_spts;

  /* Solution data structures (faces) */
  faces->U_d = faces->U;
  faces->dU_d = faces->dU;
  faces->Fconv_d = faces->Fconv;
  faces->Fvisc_d = faces->Fvisc;
  faces->P_d = faces->P;
  faces->Ucomm_d = faces->Ucomm;
  faces->Fcomm_d = faces->Fcomm;
  faces->Fcomm_temp_d = faces->Fcomm_temp;
  faces->norm_d = faces->norm;
  faces->outnorm_d = faces->outnorm;
  faces->dA_d = faces->dA;
  faces->waveSp_d = faces->waveSp;
  faces->LDG_bias_d = faces->LDG_bias;

  /* Additional data */
  /* Geometry */
  geo.fpt2gfpt_d = geo.fpt2gfpt;
  geo.fpt2gfpt_slot_d = geo.fpt2gfpt_slot;
  geo.gfpt2bnd_d = geo.gfpt2bnd;
  geo.per_fpt_list_d = geo.per_fpt_list;
  geo.coord_spts_d = geo.coord_spts;

  /* Input parameters */
  input->V_fs_d = input->V_fs;
  input->V_wall_d = input->V_wall;
  input->norm_fs_d = input->norm_fs;
  input->AdvDiff_A_d = input->AdvDiff_A;

#ifdef _MPI
  /* MPI data */
  for (auto &entry : geo.fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    auto &fpts = entry.second;
    geo.fpt_buffer_map_d[pairedRank] = fpts;
    faces->U_sbuffs_d[pairedRank] = faces->U_sbuffs[pairedRank];
    faces->U_rbuffs_d[pairedRank] = faces->U_rbuffs[pairedRank];
  }

#endif

}
#endif

void FRSolver::compute_residual(unsigned int stage, int level)
//void FRSolver::compute_residual(unsigned int stage, bool FV_mode)
{
  /* Extrapolate solution to flux points */
  eles->extrapolate_U();

  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    eles->compute_Uavg();
    eles->poly_squeeze();
  }

  /* Copy flux point data from element local to face local storage */
  U_to_faces();

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  /* Compute convective flux at solution points */
  eles->compute_Fconv();

  /* If running inviscid, use this scheduling. */
  if(!input->viscous)
  {

#ifdef _MPI
  /* Compute convective flux and common flux at non-MPI flux points */
  faces->compute_Fconv(0, geo.nGfpts_int + geo.nGfpts_bnd);
  faces->compute_common_F(0, geo.nGfpts_int + geo.nGfpts_bnd);
  
  /* Receive U data */
  faces->recv_U_data();

  /* Complete computation on remaning flux points. */
  faces->compute_Fconv(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
  faces->compute_common_F(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
  /* Compute convective and common fluxes at flux points */
  faces->compute_Fconv(0, geo.nGfpts);
  faces->compute_common_F(0, geo.nGfpts);
#endif
  }

  /* If running viscous, use this scheduling */
  else
  {
#ifdef _MPI
    /* Compute common interface solution at non-MPI flux points */
    faces->compute_common_U(0, geo.nGfpts_int + geo.nGfpts_bnd);

    /* Receieve U data */
    faces->recv_U_data();

    /* Finish computation of common interface solution */
    faces->compute_common_U(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
    /* Compute common interface solution at flux points */
    faces->compute_common_U(0, geo.nGfpts);
#endif

    /* Copy solution data at flux points from face local to element local
     * storage */
    U_from_faces();

    /* Compute gradient of state variables at solution points */
    eles->compute_dU();

    /* Transform gradient of state variables to physical space from 
     * reference space */
    eles->transform_dU();

    /* Extrapolate solution gradient to flux points */
    eles->extrapolate_dU();

    /* Copy gradient data from element local to face local storage */
    dU_to_faces();

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* Compute viscous flux at solution points */
    eles->compute_Fvisc();

    /* Compute viscous and convective flux and common interface flux 
     * at non-MPI flux points */
    faces->compute_Fvisc(0, geo.nGfpts_int + geo.nGfpts_bnd);
    faces->compute_Fconv(0, geo.nGfpts_int + geo.nGfpts_bnd);
    faces->compute_common_F(0, geo.nGfpts_int + geo.nGfpts_bnd);

    /* Receive gradient data */
    faces->recv_dU_data();

    /* Complete computation of fluxes */
    faces->compute_Fvisc(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
    faces->compute_Fconv(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
    faces->compute_common_F(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* Compute viscous flux at solution points */
    eles->compute_Fvisc();

    /* Compute viscous and convective flux and common interface fluxes 
     * at flux points*/ 
    faces->compute_Fvisc(0, geo.nGfpts);
    faces->compute_Fconv(0, geo.nGfpts);
    faces->compute_common_F(0, geo.nGfpts);
#endif

  }

  /* Transform fluxes from physical to reference space */
  eles->transform_flux();
  faces->transform_flux();

  /* Copy flux data from face local storage to element local storage */
  F_from_faces();

  if (FV_mode)
  {
    eles->compute_intF(stage);

    /* Add source term (if required) */
    if (input->source)
      add_source(stage);

    accumulate_partition_divF(stage, level);
  }
  else
  {
    /* Compute flux gradients and divergence */
    eles->compute_dF();
    eles->compute_divF(stage);

    /* Add source term (if required) */
    if (input->source)
      add_source(stage);
  }
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  eles->U_fpts.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->Ucomm.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->U_ppts.assign({eles->nPpts, eles->nEles, eles->nVars});
  eles->U_qpts.assign({eles->nQpts, eles->nEles, eles->nVars});
  eles->Uavg.assign({eles->nEles, eles->nVars});

  eles->F_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->F_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->Fcomm.assign({eles->nFpts, eles->nEles, eles->nVars});

  eles->dU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dU_qpts.assign({eles->nQpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});

  eles->divF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, nStages});

  /* Initialize solution */
  if (input->equation == AdvDiff)
  {
    if (input->ic_type == 0)
    {
      // Do nothing for now
    }
    else if (input->ic_type == 1)
    {
      if (input->nDims == 2)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);

            eles->U_spts(spt, ele, 0) = compute_U_true(x, y, 0, 0, 0, input);
          }
        }
      }
      else if (input->nDims == 3)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);
            double z = geo.coord_spts(spt, ele, 2);

            eles->U_spts(spt, ele, 0) = compute_U_true(x, y, z, 0, 0, input);

          }
        }

      }
    }
    else
    {
      ThrowException("ic_type not recognized!");
    }
  }
  else if (input->equation == EulerNS)
  {
    if (input->ic_type == 0)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          eles->U_spts(spt, ele, 0)  = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < eles->nDims; dim++)
          {
            eles->U_spts(spt, ele, dim+1)  = input->rho_fs * input->V_fs(dim);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          eles->U_spts(spt, ele, eles->nDims + 1)  = input->P_fs/(input->gamma-1.0) +
            0.5*input->rho_fs * Vsq;
        }
      }

    }
    else if (input->ic_type == 1)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);

            eles->U_spts(spt, ele, n) = compute_U_true(x, y, 0, 0, n, input);
          }
        }
      }
    }
  }
  else
  {
    ThrowException("Solution initialization not recognized!");
  }
}

void FRSolver::U_to_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
        {
          if (input->viscous) // if viscous, put extrapolated solution into Ucomm
            eles->Ucomm(fpt, ele, n) = eles->U_fpts(fpt, ele, n);
          continue;
        }
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        faces->U(gfpt, n, slot) = eles->U_fpts(fpt, ele, n);
      }
    }
  }
#endif

#ifdef _GPU
  U_to_faces_wrapper(eles->U_fpts_d, faces->U_d, eles->Ucomm_d, geo.fpt2gfpt_d,
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts, eles->nDims,
      input->equation, input->viscous);

  check_error();
#endif
}

void FRSolver::U_from_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Ucomm(fpt, ele, n) = faces->Ucomm(gfpt, n, slot);
      }
    }
  }
#endif

#ifdef _GPU
  U_from_faces_wrapper(faces->Ucomm_d, eles->Ucomm_d, geo.fpt2gfpt_d,
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts,
      eles->nDims, input->equation);

  check_error();
#endif

}

void FRSolver::dU_to_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(4)
  for (unsigned int dim = 0; dim < eles->nDims; dim++) 
  {
    for (unsigned int n = 0; n < eles->nVars; n++) 
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfpt(fpt,ele);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          int slot = geo.fpt2gfpt_slot(fpt,ele);

          faces->dU(gfpt, n, dim, slot) = eles->dU_fpts(fpt, ele, n, dim);
        }
      }
    }
  }
#endif

#ifdef _GPU
  dU_to_faces_wrapper(eles->dU_fpts_d, faces->dU_d, geo.fpt2gfpt_d, geo.fpt2gfpt_slot_d, 
      eles->nVars, eles->nEles, eles->nFpts, eles->nDims, input->equation);

  check_error();
#endif
}

void FRSolver::F_from_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Fcomm(fpt, ele, n) = faces->Fcomm(gfpt, n, slot);

      }
    }
  }
#endif

#ifdef _GPU
  /* Can reuse kernel here */
  U_from_faces_wrapper(faces->Fcomm_d, eles->Fcomm_d, geo.fpt2gfpt_d, 
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts, 
      eles->nDims, input->equation);

  check_error();
#endif
}

void FRSolver::add_source(unsigned int stage)
{
  int scale_fac = 1;
  if (FV_mode)
    scale_fac = 1 << eles->nDims;

#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele =0; ele < eles->nEles; ele++)
    {
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
          double x = geo.coord_spts(spt, ele, 0);
          double y = geo.coord_spts(spt, ele, 1);
          double z = 0;
          if (eles->nDims == 3)
            z = geo.coord_spts(spt, ele, 2);

            eles->divF_spts(spt, ele, n, stage) += compute_source_term(x, y, z, flow_time, n, input) * 
              scale_fac * eles->jaco_det_spts(spt, ele);

          //eles->divF_spts(spt, ele, n, stage) += compute_source_term(x, y, z, flow_time, n, input);
      }
    }
  }


#endif

#ifdef _GPU
  add_source_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, geo.coord_spts_d, eles->nSpts, eles->nEles,
      eles->nVars, eles->nDims, input->equation, flow_time, stage);
  check_error();
#endif

}

void FRSolver::add_source(unsigned int stage, mdvector<double> &source)
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele =0; ele < eles->nEles; ele++)
    {
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {

          eles->divF_spts(spt, ele, n, stage) += source(spt, ele, n);
      }
    }
  }

#endif

#ifdef _GPU
  add_source_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, geo.coord_spts_d, eles->nSpts, eles->nEles,
      eles->nVars, eles->nDims, input->equation, flow_time, stage);
  check_error();
#endif

}

void FRSolver::accumulate_partition_U(int level)
{
  int scale_fac = 1 << eles->nDims; 

  /* Compute partition average solutions */
  U_avg.fill(0.0);
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      int part = eparts(ele, level);
      U_avg(part, n) += eles->U_spts(0, ele, n) * (scale_fac * eles->jaco_det_spts(0, ele));
    }
  }

  /* Set solution point solutions to corresponding partition average */
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      int part = eparts(ele, level);
      eles->U_spts(0, ele, n) = U_avg(part, n)/vols[level][part];
    }
  }
}

void FRSolver::accumulate_partition_divF(unsigned int stage, int level)
{
    /* Accumulate subelement residual components */
    U_avg.fill(0.0);
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        int part = eparts(ele, level);
        U_avg(part, n) += eles->divF_spts(0, ele, n, stage);
      }
    }

    /* Set residual to accumulated partition residuals */
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        int part = eparts(ele, level);
        eles->divF_spts(0, ele, n, stage) = U_avg(part, n);// / vols[level][part];
      }
    }
}

void FRSolver::update()
{
  
#ifdef _CPU
  U_ini = eles->U_spts;
#endif

#ifdef _GPU
  device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.get_nvals());
#endif

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(0) * eles->divF_spts(spt, ele, n, stage) / eles->jaco_det_spts(spt, ele);
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(ele) * eles->divF_spts(spt, ele, n, stage) / eles->jaco_det_spts(spt, ele);
          }
        }
#endif

#ifdef _GPU
    RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
        input->equation, stage, nStages, false);
    check_error();
#endif

  }

  /* Final stage */
  compute_residual(nStages-1);

  if (input->dt_scheme == "RKj")
  {
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt(0) * eles->divF_spts(spt, ele, n, nStages - 1) / eles->jaco_det_spts(spt, ele);
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt(ele) * eles->divF_spts(spt, ele, n, nStages - 1) / eles->jaco_det_spts(spt, ele);
          }
        }
#endif

#ifdef _GPU
    RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
        input->equation, nStages - 1, nStages, false);
    check_error();
#endif

  }
  else
  {
#ifdef _CPU
    eles->U_spts = U_ini;
#endif
#ifdef _GPU
    device_copy(eles->U_spts_d, U_ini_d, eles->U_spts_d.get_nvals());
#endif

#ifdef _CPU
    for (unsigned int stage = 0; stage < nStages; stage++)
    {
#pragma omp parallel for collapse(3)
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(0) * eles->divF_spts(spt, ele, n, stage) / eles->jaco_det_spts(spt, ele);
            }
            else
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(ele) * eles->divF_spts(spt, ele, n, stage) / eles->jaco_det_spts(spt, ele);
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
      RK_update_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
          rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
          input->equation, 0, nStages, true);
      check_error();
#endif
  }

  flow_time += dt(0);
  current_iter++;
 
}

void FRSolver::update_FV(int level)
{
  
#ifdef _CPU
  U_ini = eles->U_spts;
#endif

#ifdef _GPU
  device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.get_nvals());
#endif

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage, level);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt(level);
      }
    }

#ifdef _CPU
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        int part = eparts(ele, level); 
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt_part(0) * eles->divF_spts(spt, ele, n, stage) / vols[level][part];
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt_part(part) * eles->divF_spts(spt, ele, n, stage) / vols[level][part];
          }
        }
      }
    }
#endif

#ifdef _GPU
    RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
        input->equation, stage, nStages, false);
    check_error();
#endif

  }

  /* Final stage */
  compute_residual(nStages-1, level);

  if (input->dt_scheme == "RKj")
  {
#ifdef _CPU
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        int part = eparts(ele, level); 
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt_part(0) * eles->divF_spts(spt, ele, n, nStages - 1) / vols[level][part];
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt_part(part) * eles->divF_spts(spt, ele, n, nStages - 1) / vols[level][part];
          }
        }
      }
    }
#endif

#ifdef _GPU
    RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
        input->equation, nStages - 1, nStages, false);
    check_error();
#endif

  }
  else
  {
#ifdef _CPU
    eles->U_spts = U_ini;
#endif
#ifdef _GPU
    device_copy(eles->U_spts_d, U_ini_d, eles->U_spts_d.get_nvals());
#endif

#ifdef _CPU
    for (unsigned int stage = 0; stage < nStages; stage++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          int part = eparts(ele, level);
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt_part(0) * eles->divF_spts(spt, ele, n, stage) / vols[level][part];
            }
            else
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt_part(part) * eles->divF_spts(spt, ele, n, stage) / vols[level][part];
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
      RK_update_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
          rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
          input->equation, 0, nStages, true);
      check_error();
#endif
  }

  flow_time += dt(0);
  current_iter++;
 
}

void FRSolver::update_with_source(mdvector<double> &source)
{
  
    U_ini = eles->U_spts;

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(0) * (eles->divF_spts(spt, ele, n, stage)
                + source(spt, ele, n))/ eles->jaco_det_spts(spt, ele);
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(ele) * (eles->divF_spts(spt, ele, n, stage)
                + source(spt, ele, n))  / eles->jaco_det_spts(spt, ele);
          }
        }
  }

  /* Final stage */
  compute_residual(nStages-1);

  if (input->dt_scheme == "RKj")
  {
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt(0) * (eles->divF_spts(spt, ele, n, nStages - 1)
                + source(spt, ele, n))/ eles->jaco_det_spts(spt, ele);
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt(ele) * (eles->divF_spts(spt, ele, n, nStages - 1)
                + source(spt, ele, n))  / eles->jaco_det_spts(spt, ele);
          }
        }
  }
  else
  {
    eles->U_spts = U_ini;

    for (unsigned int stage = 0; stage < nStages; stage++)
#pragma omp parallel for collapse(3)
      for (unsigned int n = 0; n < eles->nVars; n++)
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(0) * (eles->divF_spts(spt, ele, n, stage) 
                  + source(spt, ele, n))  / eles->jaco_det_spts(spt, ele);
            }
            else
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(ele) * (eles->divF_spts(spt, ele, n, stage)
                  + source(spt, ele, n))  / eles->jaco_det_spts(spt, ele);
            }
          }
  }

  current_iter++;

}

#ifdef _GPU
void FRSolver::update_with_source(mdvector_gpu<double> &source)
{
  
  device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.get_nvals());

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

    RK_update_source_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, 
        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
        input->equation, stage, nStages, false);
    check_error();

  }

  /* Final stage */
  compute_residual(nStages-1);
  device_copy(eles->U_spts_d, U_ini_d, eles->U_spts_d.get_nvals());

  RK_update_source_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, 
      rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
      input->equation, 0, nStages, true);
  check_error();

}
#endif

void FRSolver::update_with_source_FV(mdvector<double> &source, int level)
{
  U_ini = eles->U_spts;

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage, level);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt(level);
      }
    }

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        int part = eparts(ele, level);
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          if (input->dt_type != 2)
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt_part(0) * (eles->divF_spts(spt, ele, n, stage)
                + source(spt, ele, n)) / vols[level][part];
          }
          else
          {
            eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt_part(part) * (eles->divF_spts(spt, ele, n, stage)
                + source(spt, ele, n)) / vols[level][part];
          }
        }
      }
    }
  }

  /* Final stage */
  compute_residual(nStages-1, level);

  if (input->dt_scheme == "RKj")
  {
    for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          int part = eparts(ele, level);
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt_part(0) * (eles->divF_spts(spt, ele, n, nStages - 1)
                  + source(spt, ele, n)) / vols[level][part];
            }
            else
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(nStages - 1) * dt_part(part) * (eles->divF_spts(spt, ele, n, nStages - 1)
                  + source(spt, ele, n)) / vols[level][part];
            }
          }
        }
      }

  }
  else
  {
    eles->U_spts = U_ini;

    for (unsigned int stage = 0; stage < nStages; stage++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          int part = eparts(ele, level);
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt_part(0) * (eles->divF_spts(spt, ele, n, stage)
                  + source(spt, ele, n))  / vols[level][part];
            }
            else
            {
              eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt_part(part) * (eles->divF_spts(spt, ele, n, stage)
                  + source(spt, ele, n))  / vols[level][part];
            }
          }
        }
      }
    }
  }

  current_iter++;

}

void FRSolver::compute_element_dt(int level)
{
#ifdef _CPU
  if (FV_mode)
    dt_part.fill(100);

//#pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  { 
    double waveSp_max = 0.0;

    /* Compute maximum wavespeed */
    for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
    {
      /* Skip if on ghost edge. */
      int gfpt = geo.fpt2gfpt(fpt,ele);
      if (gfpt == -1)
        continue;

      double waveSp = faces->waveSp(gfpt);

      //if (!FV_mode)
        waveSp /= faces->dA(gfpt);

      waveSp_max = std::max(waveSp, waveSp_max);
    }

    /* Note: CFL is applied to parent space element with width 2 */
   // if (!FV_mode)
      dt(ele) = (input->CFL) * get_cfl_limit(order) * (2.0 / (waveSp_max+1.e-10));
    //else
   //   dt(ele) = (input->CFL_fv) * (vols[level][eparts(ele,level)]) / (waveSp_max+1.e-10);

    if (FV_mode)
      dt_part(eparts(ele,level)) = std::min(dt_part(eparts(ele, level)), dt(ele));
  }


  if (input->dt_type == 1) /* Global minimum */
  {
    if (!FV_mode)
    {
      dt(0) = *std::min_element(dt.data(), dt.data()+eles->nEles);
      //std::cout << eles->order << " " << dt(0) << std::endl;
    }
    if (FV_mode)
    {
      dt_part(0) = *std::min_element(dt_part.data(), dt_part.data() + vols[level].size());
      //std::cout << level << " " << dt_part(0) << std::endl;
    }


#ifdef _MPI
    MPI_Allreduce(MPI_IN_PLACE, &dt(0), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); 
#endif

  }
#endif

#ifdef _GPU
  compute_element_dt_wrapper(dt_d, faces->waveSp_d, faces->dA_d, geo.fpt2gfpt_d, 
      input->CFL, order, input->dt_type, eles->nFpts, eles->nEles);
#endif
}

void FRSolver::write_solution(std::string prefix)
{
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
#endif

  if (input->rank == 0) std::cout << "Writing data to file..." << std::endl;

  std::stringstream ss;
#ifdef _MPI

  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_" << std::setw(9) << std::setfill('0');
    ss << current_iter << ".pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;
    if (input->equation == AdvDiff)
    {
      f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
      f << std::endl;

    }
    else if (input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (eles->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        f << "<PDataArray type=\"Float32\" Name=\"" << var[n];
        f << "\" format=\"ascii\"/>";
        f << std::endl;
      }
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
      ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write parition solution to file in .vtu format */
  std::ofstream f(outputfile);


  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

  /* Write comments for solution order, iteration number and flowtime */
  f << "<!-- ORDER " << input->order << " -->" << std::endl;
  f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << current_iter << " -->" << std::endl;

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << eles->nPpts * eles->nEles << "\" ";
  f << "NumberOfCells=\"" << eles->nSubelements * eles->nEles << "\">";
  f << std::endl;

  
  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl; 

  if (eles->nDims == 2)
  {
    // TODO: Change order of ppt structures for better looping 
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << geo.coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }

  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << geo.ppt_connect(i, subele) + ele*eles->nPpts << " ";
      }
      f << std::endl;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      f << offset << " ";
      offset += eles->nNodesPerSubelement;
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int nCells = eles->nSubelements * eles->nEles;
  if (eles->nDims == 2)
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 9 << " ";
  }
  else
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 12 << " ";
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write solution information */
  f << "<PointData>" << std::endl;

  /* TEST: Write cell average solution */
  //eles->compute_Uavg();

  /* Extrapolate solution to plot points */
  auto &A = eles->oppE_ppts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_ppts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nPpts, &B, 
      eles->nSpts, 0.0, &C, eles->nPpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nPpts, &B, 
      eles->nSpts, 0.0, &C, eles->nPpts);
#endif

  /* Apply squeezing if needed */
  if (input->squeeze)
  {
    eles->compute_Uavg();

#ifdef _GPU
    eles->Uavg = eles->Uavg_d;
#endif

    eles->poly_squeeze_ppts();
  }

  if (input->equation == AdvDiff)
  {
    f << "<DataArray type=\"Float32\" Name=\"u\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << std::scientific << std::setprecision(16) << eles->U_ppts(ppt, ele, 0);
        f  << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }
  else if(input->equation == EulerNS)
  {
    std::vector<std::string> var;
    if (eles->nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
      f << "format=\"ascii\">"<< std::endl;
      
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
        {
          f << std::scientific << std::setprecision(16);
          f << eles->U_ppts(ppt, ele, n) << " ";
        }

        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
  }

  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::write_partition_file()
{

  if (input->rank == 0) std::cout << "Writing data to file..." << std::endl;

  std::stringstream ss;
#ifdef _MPI

  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << "partitions.pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;

    for (unsigned int H = 0; H < input->hmg_levels; h++)
    {
      f << "<PDataArray type=\"Int32\" Name=\"" << H;
      f << "\" format=\"ascii\"/>";
      f << std::endl;
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << "partitions_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << "partitions_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << "partitions.vtu";
#endif

  auto outputfile = ss.str();

  /* Write parition solution to file in .vtu format */
  std::ofstream f(outputfile);


  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

  /* Write comments for iteration number and flowtime */
  f << "<!-- TIME " << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << current_iter << " -->" << std::endl;

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << eles->nPpts * eles->nEles << "\" ";
  f << "NumberOfCells=\"" << eles->nSubelements * eles->nEles << "\">";
  f << std::endl;

  
  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl; 

  if (eles->nDims == 2)
  {
    // TODO: Change order of ppt structures for better looping 
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << geo.coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }

  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << geo.ppt_connect(i, subele) + ele*eles->nPpts << " ";
      }
      f << std::endl;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      f << offset << " ";
      offset += eles->nNodesPerSubelement;
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int nCells = eles->nSubelements * eles->nEles;
  if (eles->nDims == 2)
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 9 << " ";
  }
  else
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 12 << " ";
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write solution information */
  f << "<PointData>" << std::endl;

  /* TEST: Write cell average solution */
  //eles->compute_Uavg();

  for (unsigned int H = 0; H < input->hmg_levels; H++)
  {
    f << "<DataArray type=\"Int32\" Name=\"" << H << "\" ";
    f << "format=\"ascii\">"<< std::endl;
    
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << eparts(ele, H) << " ";
      }

      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1, int level)
{

  /* If running on GPU, copy out divergence */
#ifdef _GPU
  eles->divF_spts = eles->divF_spts_d;
  dt = dt_d;
#endif

  std::vector<double> res(eles->nVars,0.0);

//#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele =0; ele < eles->nEles; ele++)
    {
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        if (!FV_mode)
        {
          //eles->divF_spts(spt, ele, n, nStages-1) /= eles->jaco_det_spts(spt, ele);
          eles->divF_spts(spt, ele, n, 0) /= eles->jaco_det_spts(spt, ele);
        }
        else
        {
          int part = eparts(ele, level);
          //eles->divF_spts(spt, ele, n, nStages-1) /= vols[level][part];
          eles->divF_spts(spt, ele, n, 0) /= vols[level][part];
        }
      }
    }
  }

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    /* Infinity norm */
    if (input->res_type == 0)
      res[n] =*std::max_element(&eles->divF_spts(0, 0, n, 0), 
          &eles->divF_spts(0, 0, n+1, 1));

    /* L1 norm */
    else if (input->res_type == 1)
      res[n] = std::accumulate(&eles->divF_spts(0, 0, n, 0), 
          &eles->divF_spts(0, 0, n+1, 1), 0.0, abs_sum<double>());

    /* L2 norm */
    else if (input->res_type == 2)
      res[n] = std::accumulate(&eles->divF_spts(0, 0, n, 0), 
            &eles->divF_spts(0, 0, n+1, 1), 0.0, square<double>());
  }

  unsigned int nDoF =  (eles->nSpts * eles->nEles);

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), eles->nVars, MPI_DOUBLE, oper, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), eles->nVars, MPI_DOUBLE, oper, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  /* Print residual to terminal (normalized by number of solution points) */
  if (input->rank == 0) 
  {
    if (input->res_type == 2)
    {
      for (auto &val : res)  
        val = std::sqrt(val);
    }

    std::cout << current_iter << " ";
    for (auto val : res)
      std::cout << std::scientific << val / nDoF << " ";

    std::cout << "dt: " << dt(0);
    std::cout << std::endl;
    
    /* Write to history file */
    auto t2 = std::chrono::high_resolution_clock::now();
    auto current_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    f << current_iter << " " << current_runtime.count() << " ";

    for (auto val : res)
      f << std::scientific << val / nDoF << " ";
    f << std::endl;
  }
}

void FRSolver::report_forces(std::ofstream &f)
{
  /* If using GPU, copy out solution, gradient and pressure */
#ifdef _GPU
  faces->U = faces->U_d;
  faces->dU = faces->dU_d;
  faces->P = faces->P_d;
#endif

  std::array<double, 3> force_conv, force_visc, taun;
  force_conv.fill(0.0); force_visc.fill(0.0); taun.fill(0.0);

  std::stringstream ss;
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".cp";
#else
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << ".cp";
#endif

  auto cpfile = ss.str();
  std::ofstream g(cpfile);

  /* Get angle of attack (and sideslip) */
  double aoa = std::atan2(input->V_fs(1), input->V_fs(0)); 
  double aos = 0.0;
  if (eles->nDims == 3)
    aos = std::atan2(input->V_fs(2), input->V_fs(0));

  /* Compute factor for non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);

    if (bnd_id >= 8) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(fpt, 0);

      double CP = (PL - input->P_fs) * fac;

      /* Write CP distrubtion to file */
      for(unsigned int dim = 0; dim < eles->nDims; dim++)
        g << std::scientific << faces->coord(fpt, dim) << " ";
      g << std::scientific << CP << std::endl;

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < eles->nDims; dim++)
      {
        force_conv[dim] += eles->weights_spts(count%eles->nSpts1D) * CP * 
          faces->norm(fpt, dim, 0) * faces->dA(fpt);
      }

      if (input->viscous)
      {
        if (eles->nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double e = faces->U(fpt, 3, 0);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);
          
          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + 
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0, 0) + tauxy * faces->norm(fpt, 1, 0);
          taun[1] = tauxy * faces->norm(fpt, 0, 0) + tauyy * faces->norm(fpt, 1, 0);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_spts(count%eles->nSpts1D) * taun[dim] * 
              faces->dA(fpt) * fac;

        }
        else if (eles->nDims == 3)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double momz = faces->U(fpt, 3, 0);
          double e = faces->U(fpt, 4, 0);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

           /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);
          double momz_dx = faces->dU(fpt, 3, 0, 0);
          
          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);
          double momz_dy = faces->dU(fpt, 3, 1, 0);

          double rho_dz = faces->dU(fpt, 0, 2, 0);
          double momx_dz = faces->dU(fpt, 1, 2, 0);
          double momy_dz = faces->dU(fpt, 2, 2, 0);
          double momz_dz = faces->dU(fpt, 3, 2, 0);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + 
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;
          double du_dz = (momx_dz - rho_dz * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;
          double dv_dz = (momy_dz - rho_dz * v) / rho;

          double dw_dx = (momz_dx - rho_dx * w) / rho;
          double dw_dy = (momz_dy - rho_dy * w) / rho;
          double dw_dz = (momz_dz - rho_dz * w) / rho;

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0, 0) + tauxy * faces->norm(fpt, 1, 0) + tauxz * faces->norm(fpt, 2, 0);
          taun[1] = tauxy * faces->norm(fpt, 0, 0) + tauyy * faces->norm(fpt, 1, 0) + tauyz * faces->norm(fpt, 2, 0);
          taun[3] = tauxz * faces->norm(fpt, 0, 0) + tauyz * faces->norm(fpt, 1, 0) + tauzz * faces->norm(fpt, 2, 0);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_spts(count%eles->nSpts1D) * taun[dim] * 
              faces->dA(fpt) * fac;

        }
        
      }
      count++;
    }
  }

  /* Compute lift and drag coefficients */
  double CL_conv, CD_conv, CL_visc, CD_visc;

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(force_conv.data(), force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(force_visc.data(), force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  if (input->rank == 0)
  {
    if (eles->nDims == 2)
    {
      CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
      CD_conv = force_conv[0] * std::cos(aoa) + force_conv[1] * std::sin(aoa);
      CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
      CD_visc = force_visc[0] * std::cos(aoa) + force_visc[1] * std::sin(aoa);
    }
    else if (eles->nDims == 3)
    {
      CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
      CD_conv = force_conv[0] * std::cos(aoa) * std::cos(aos) + force_conv[1] * std::sin(aoa) + 
        force_conv[2] * std::sin(aoa) * std::cos(aos);
      CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
      CD_visc = force_visc[0] * std::cos(aoa) * std::cos(aos) + force_visc[1] * std::sin(aoa) + 
        force_visc[2] * std::sin(aoa) * cos(aos);
    }

    std::cout << "CL_conv = " << CL_conv << " CD_conv = " << CD_conv;
    f << current_iter << " ";
    f << std::scientific << std::setprecision(16) << CL_conv << " " << CD_conv;

    if (input->viscous)
    {
      std::cout << " CL_visc = " << CL_visc << " CD_visc = " << CD_visc;
      f << std::scientific << std::setprecision(16) << " " << CL_visc << " " << CD_visc;
    }

    std::cout << std::endl;
    f << std::endl;
  }
}

void FRSolver::report_error(std::ofstream &f)
{
  /* If using GPU, copy out solution */
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
  eles->dU_spts = eles->dU_spts_d;
#endif

  /* Extrapolate solution to quadrature points */
  auto &A = eles->oppE_qpts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_qpts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
      eles->nSpts, 0.0, &C, eles->nQpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
      eles->nSpts, 0.0, &C, eles->nQpts);
#endif

  /* Extrapolate derivatives to quadrature points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
      auto &A = eles->oppE_qpts(0, 0);
      auto &B = eles->dU_spts(0, 0, 0, dim);
      auto &C = eles->dU_qpts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
          eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
          eles->nSpts, 0.0, &C, eles->nQpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
          eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
          eles->nSpts, 0.0, &C, eles->nQpts);
#endif

  }

  std::vector<double> l2_error(2,0.0);

  unsigned int n = input->err_field;
  std::vector<double> dU_true(2, 0.0), dU_error(2, 0.0);
#pragma omp for collapse (2)
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int qpt = 0; qpt < eles->nQpts; qpt++)
      {
        double U_true = 0.0;
        double weight = 0.0;

        if (eles->nDims == 2)
        {
          /* Compute true solution and derivatives */
          if (input->test_case == 3) // Isentropic Bump
          {
            U_true = input->P_fs / std::pow(input->rho_fs, input->gamma);
          }
          else 
          {
            U_true = compute_U_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
                flow_time, n, input);
          }

          dU_true[0] = compute_dU_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
              flow_time, n, 0, input);
          dU_true[1] = compute_dU_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
              flow_time, n, 1, input);
          

          /* Get quadrature point index and weight */
          unsigned int i = eles->idx_qpts(qpt,0);
          unsigned int j = eles->idx_qpts(qpt,1);
          weight = eles->weights_qpts[i] * eles->weights_qpts[j];
        }
        else if (eles->nDims == 3)
        {
          ThrowException("Under construction!");
        }

        /* Compute errors */
        double U_error;
        if (input->test_case == 2) // Couette flow case
        {
          double rho = eles->U_qpts(qpt, ele, 0);
          double u =  eles->U_qpts(qpt, ele, 1) / rho;
          double rho_dx = eles->dU_qpts(qpt, ele, 0, 0);
          double rho_dy = eles->dU_qpts(qpt, ele, 0, 1);
          double momx_dx = eles->dU_qpts(qpt, ele, 1, 0);
          double momx_dy = eles->dU_qpts(qpt, ele, 1, 1);

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          U_error = U_true - u;
          dU_error[0] = dU_true[0] - du_dx;
          dU_error[1] = dU_true[1] - du_dy;
        }
        else if (input->test_case == 3) // Isentropic bump
        {
          double momF = 0.0;
          for (unsigned int dim = 0; dim < eles->nDims; dim ++)
          {
            momF += eles->U_qpts(qpt, ele, dim + 1) * eles->U_qpts(qpt, ele, dim + 1);
          }

          momF /= eles->U_qpts(qpt, ele, 0);

          double P = (input->gamma - 1.0) * (eles->U_qpts(qpt, ele, 3) - 0.5 * momF);

          U_error = U_true - P/std::pow(eles->U_qpts(qpt, ele, 0), input->gamma);
        }
        else
        {
          U_error = U_true - eles->U_qpts(qpt, ele, n);
          dU_error[0] = dU_true[0] - eles->dU_qpts(qpt, ele, n, 0); 
          dU_error[1] = dU_true[1] - eles->dU_qpts(qpt, ele, n, 1);
        }

        l2_error[0] += weight * eles->jaco_det_qpts(qpt, ele) * U_error * U_error; 
        l2_error[1] += weight * eles->jaco_det_qpts(qpt, ele) * (U_error * U_error +
            dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1]); 
      }
  }

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(l2_error.data(), l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

#endif


  /* Print to terminal */
  if (input->rank == 0)
  {
    std::cout << "l2_error: ";
    for (auto &val : l2_error)
      std::cout << std::scientific << std::sqrt(val) << " ";
    std::cout << std::endl;

    /* Write to file */
    f << current_iter << " ";
    for (auto &val : l2_error)
      f << std::scientific << std::setprecision(16) << std::sqrt(val) << " ";
    f << std::endl;
  }

}
