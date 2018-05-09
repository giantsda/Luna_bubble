#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_out.h>
#include <time.h>

using namespace dealii;

///////////////////////////
// FOR TRANSPORT PROBLEM //
///////////////////////////
// TIME_INTEGRATION
#define FORWARD_EULER 0
#define SSP33 1   // used in LevelSetSolver
//#define CHKMEMQ
// PROBLEM 
#define FILLING_TANK 0
#define BREAKING_DAM 1 
#define FALLING_DROP 2
#define SMALL_WAVE_PERTURBATION 3

#include "NavierStokesSolver.cc"
#include "LevelSetSolver.cc"
#include "utilities.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template<int dim>
  class MultiPhase
  {
  public:
    MultiPhase (const unsigned int degree_LS, const unsigned int degree_U); // constructor
    ~MultiPhase ();   // destructor
    void
    run (); // main running function.In this function, the matrix is assambled, solved and results are ouputed.

  private:
    void
    set_boundary_inlet ();
    void
    get_boundary_values_U ();
    void
    get_boundary_values_phi (std::vector<unsigned int> &boundary_values_id_phi,
			     std::vector<double> &boundary_values_phi);
    void
    output_results ();
    void
    output_vectors ();
    void
    output_rho ();
    double
    get_u_max (); /* calculate the largest u velocity in the solution so that the time step size
     can be adjusted dynamically. */

    void
    output_rho_mine (); /* This function modified the output function of step 35 to make an joint solution to
     write a vtk output file which allows us to visualize the velocity vectors*/

    void
    setup ();
    void
    initial_condition ();
    void
    init_constraints ();

    MPI_Comm mpi_communicator; // define the mpi_communicator, this variable is initialized in the constructor as MPI_COMM_WORLD.
    parallel::distributed::Triangulation<dim> triangulation;

    int degree_LS;
    DoFHandler<dim> dof_handler_LS; // thisDOF handler has the solution for the volume of fraction.
    FE_Q<dim> fe_LS;
    IndexSet locally_owned_dofs_LS;
    IndexSet locally_relevant_dofs_LS;

    int degree_U;                  // for velocity field.
    DoFHandler<dim> dof_handler_U;
    FE_Q<dim> fe_U;
    IndexSet locally_owned_dofs_U;
    IndexSet locally_relevant_dofs_U;

    DoFHandler<dim> dof_handler_P;     // for pressure field.
    FE_Q<dim> fe_P;
    IndexSet locally_owned_dofs_P;
    IndexSet locally_relevant_dofs_P;

    ConditionalOStream pcout;

    // SOLUTION VECTORS
    PETScWrappers::MPI::Vector locally_relevant_solution_phi;
    PETScWrappers::MPI::Vector locally_relevant_solution_u;
    PETScWrappers::MPI::Vector locally_relevant_solution_v;
    PETScWrappers::MPI::Vector locally_relevant_solution_p;
    PETScWrappers::MPI::Vector completely_distributed_solution_phi;
    PETScWrappers::MPI::Vector completely_distributed_solution_u;
    PETScWrappers::MPI::Vector completely_distributed_solution_v;
    PETScWrappers::MPI::Vector completely_distributed_solution_p;
    // BOUNDARY VECTORS
    std::vector<unsigned int> boundary_values_id_u;
    std::vector<unsigned int> boundary_values_id_v;
    std::vector<unsigned int> boundary_values_id_phi;
    std::vector<double> boundary_values_u;
    std::vector<double> boundary_values_v;
    std::vector<double> boundary_values_phi;

    ConstraintMatrix constraints;

    double time_0;
    double time_step;
    double final_time;
    unsigned int timestep_number;
    double cfl;
    double umax;
    double min_h;

    double sharpness;
    int sharpness_integer;

    unsigned int n_refinement;
    unsigned int output_number;
    double output_time;
    bool get_output;

    bool verbose;

    //FOR NAVIER STOKES
    double rho_fluid;
    double nu_fluid;
    double rho_air;
    double nu_air;
    double nu;
    double eps;

    //FOR TRANSPORT
    double cK; //compression coeff
    double cE; //entropy-visc coeff
    unsigned int TRANSPORT_TIME_INTEGRATION;
    std::string ALGORITHM;
    unsigned int PROBLEM;
  };

template<int dim>
  MultiPhase<dim>::MultiPhase (const unsigned int degree_LS,
			       const unsigned int degree_U) :
      mpi_communicator (MPI_COMM_WORLD), triangulation (
	  mpi_communicator,
	  typename Triangulation<dim>::MeshSmoothing (
	      Triangulation < dim > ::smoothing_on_refinement
		  | Triangulation < dim > ::smoothing_on_coarsening)), degree_LS (
	  degree_LS), dof_handler_LS (triangulation), fe_LS (degree_LS), degree_U (
	  degree_U), dof_handler_U (triangulation), fe_U (degree_U), dof_handler_P (
	  triangulation), fe_P (degree_U - 1), pcout (
	  std::cout, (Utilities::MPI::this_mpi_process (mpi_communicator) == 0))
  {
  }

template<int dim>
  MultiPhase<dim>::~MultiPhase ()
  {
    dof_handler_LS.clear ();
    dof_handler_U.clear ();
    dof_handler_P.clear ();
  }
template<int dim>
  double
  MultiPhase<dim>::get_u_max ()
  {
    double u_max;
    std::pair<int, int> range = locally_relevant_solution_u.local_range ();
    //the range of the local processor owns is returned as a pair.
    for (int i = range.first; i < range.second; i++)
      {
	u_max = (
	    u_max > std::abs (locally_relevant_solution_u[i]) ?
		u_max : std::abs (locally_relevant_solution_u[i]));

	//find out the largest value of u velocity component
      }
    return u_max;
  }

/////////////////////////////////////////
///////////////// SETUP /////////////////
/////////////////////////////////////////
template<int dim>
  void
  MultiPhase<dim>::setup ()
  {

    // initialize all the vector for using later.

    // setup system LS
    dof_handler_LS.distribute_dofs (fe_LS);
    locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler_LS,
					     locally_relevant_dofs_LS);
    // setup system U
    dof_handler_U.distribute_dofs (fe_U);
    locally_owned_dofs_U = dof_handler_U.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler_U,
					     locally_relevant_dofs_U);
    // setup system P //
    dof_handler_P.distribute_dofs (fe_P);
    locally_owned_dofs_P = dof_handler_P.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler_P,
					     locally_relevant_dofs_P);
    // init vectors for phi
    locally_relevant_solution_phi.reinit (locally_owned_dofs_LS,
					  locally_relevant_dofs_LS,
					  mpi_communicator);
    locally_relevant_solution_phi = 0;
    completely_distributed_solution_phi.reinit (locally_owned_dofs_P,
						mpi_communicator);
    //init vectors for u
    locally_relevant_solution_u.reinit (locally_owned_dofs_U,
					locally_relevant_dofs_U,
					mpi_communicator);
    locally_relevant_solution_u = 0;
    completely_distributed_solution_u.reinit (locally_owned_dofs_U,
					      mpi_communicator);
    //init vectors for v
    locally_relevant_solution_v.reinit (locally_owned_dofs_U,
					locally_relevant_dofs_U,
					mpi_communicator);
    locally_relevant_solution_v = 0;
    completely_distributed_solution_v.reinit (locally_owned_dofs_U,
					      mpi_communicator);
    //init vectors for p
    locally_relevant_solution_p.reinit (locally_owned_dofs_P,
					locally_relevant_dofs_P,
					mpi_communicator);
    locally_relevant_solution_p = 0;
    completely_distributed_solution_p.reinit (locally_owned_dofs_P,
					      mpi_communicator);
    // INIT CONSTRAINTS
    init_constraints ();
  }

template<int dim>
  void
  MultiPhase<dim>::initial_condition ()
  {
    time_0 = 0;
    // Initial conditions //
    // init condition for phi
    completely_distributed_solution_phi = 0;
    VectorTools::interpolate (dof_handler_LS,
			      InitialPhi<dim> (PROBLEM, sharpness),
			      completely_distributed_solution_phi);
// InitialPhi<dim> (PROBLEM, sharpness), works as a function.

    constraints.distribute (completely_distributed_solution_phi);
    locally_relevant_solution_phi = completely_distributed_solution_phi;
    // init condition for u=0
    completely_distributed_solution_u = 0;
    VectorTools::interpolate (dof_handler_U, ZeroFunction<dim> (),
			      completely_distributed_solution_u);
    constraints.distribute (completely_distributed_solution_u);
    locally_relevant_solution_u = completely_distributed_solution_u;
    // init condition for v
    completely_distributed_solution_v = 0;
    VectorTools::interpolate (dof_handler_U, ZeroFunction<dim> (),
			      completely_distributed_solution_v);
    constraints.distribute (completely_distributed_solution_v);
    locally_relevant_solution_v = completely_distributed_solution_v;
    // init condition for p
    completely_distributed_solution_p = 0;
    VectorTools::interpolate (dof_handler_P, ZeroFunction<dim> (),
			      completely_distributed_solution_p);
    constraints.distribute (completely_distributed_solution_p);
    locally_relevant_solution_p = completely_distributed_solution_p;
  }

template<int dim>
  void
  MultiPhase<dim>::init_constraints () // define the initial constraints.
  {
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs_LS);
    DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints);
    constraints.close ();
  }

template<int dim>
  void
  MultiPhase<dim>::get_boundary_values_U ()
  {
    /* this function set the no slip boundary ocndition for velocity.
     The velocity is zero at top and bottom wall. BoundaryV, BoundaryU, BoundaryV2, BoundaryU2 defines
     the inlet bounary condition */
    std::map<unsigned int, double> map_boundary_values_u;
    std::map<unsigned int, double> map_boundary_values_v;
    std::map<unsigned int, double> map_boundary_values_w;

    // NO-SLIP CONDITION
    if (PROBLEM == BREAKING_DAM || PROBLEM == FALLING_DROP)
      {
	//LEFT

	VectorTools::interpolate_boundary_values (dof_handler_U, 0,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 0,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
	// RIGHT
	VectorTools::interpolate_boundary_values (dof_handler_U, 1,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 1,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
	// BOTTOM
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
	// TOP
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
      }
    else if (PROBLEM == SMALL_WAVE_PERTURBATION)
      { // no slip in bottom and top and slip in left and right
	//LEFT
	VectorTools::interpolate_boundary_values (dof_handler_U, 0,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	// RIGHT
	VectorTools::interpolate_boundary_values (dof_handler_U, 1,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	// BOTTOM
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
	// TOP
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
      }
    else if (PROBLEM == FILLING_TANK)
      {
	//LEFT: entry in x, zero in y
	VectorTools::interpolate_boundary_values (dof_handler_U, 0,
						  BoundaryU<dim> (PROBLEM),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 0,
						  BoundaryV<dim> (PROBLEM),
						  map_boundary_values_v);
	//RIGHT: no-slip condition
	VectorTools::interpolate_boundary_values (dof_handler_U, 1,
						  BoundaryV2<dim> (PROBLEM),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 1,
						  BoundaryU2<dim> (PROBLEM),
						  map_boundary_values_v);
	//BOTTOM: non-slip
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 2,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
	//TOP: exit in y, zero in x
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_u);
	VectorTools::interpolate_boundary_values (dof_handler_U, 3,
						  ZeroFunction<dim> (),
						  map_boundary_values_v);
      }
    else
      {
	pcout << "Error in type of PROBLEM at Boundary Conditions" << std::endl;
	abort ();
      }
    boundary_values_id_u.resize (map_boundary_values_u.size ());
    boundary_values_id_v.resize (map_boundary_values_v.size ());
    boundary_values_u.resize (map_boundary_values_u.size ());
    boundary_values_v.resize (map_boundary_values_v.size ());
    std::map<unsigned int, double>::const_iterator boundary_value_u =
	map_boundary_values_u.begin ();
    std::map<unsigned int, double>::const_iterator boundary_value_v =
	map_boundary_values_v.begin ();

//    for (auto& t : map_boundary_values_v)
//      std::cout << t.first << " " << t.second << "\n";

    for (int i = 0; boundary_value_u != map_boundary_values_u.end ();
	++boundary_value_u, ++i)
      {
	boundary_values_id_u[i] = boundary_value_u->first;
	boundary_values_u[i] = boundary_value_u->second;
//      std::cout<<boundary_values_u[i]<<std::endl;
      }
    for (int i = 0; boundary_value_v != map_boundary_values_v.end ();
	++boundary_value_v, ++i)
      {
	boundary_values_id_v[i] = boundary_value_v->first;
	boundary_values_v[i] = boundary_value_v->second;
      }
  }

template<int dim>
  void
  MultiPhase<dim>::set_boundary_inlet ()
  {
    const QGauss<dim - 1> face_quadrature_formula (1); // center of the face
    FEFaceValues < dim
	> fe_face_values (
	    fe_U, face_quadrature_formula,
	    update_values | update_quadrature_points | update_normal_vectors);
    const unsigned int n_face_q_points = face_quadrature_formula.size ();
    std::vector<double> u_value (n_face_q_points);
    std::vector<double> v_value (n_face_q_points);

    typename DoFHandler<dim>::active_cell_iterator cell_U =
	dof_handler_U.begin_active (), endc_U = dof_handler_U.end ();
    Tensor < 1, dim > u;

    for (; cell_U != endc_U; ++cell_U)
      if (cell_U->is_locally_owned ())
	for (unsigned int face = 0;
	    face < GeometryInfo < dim > ::faces_per_cell; ++face)
	  if (cell_U->face (face)->at_boundary ())
	    {
	      fe_face_values.reinit (cell_U, face);
	      fe_face_values.get_function_values (locally_relevant_solution_u,
						  u_value);
	      fe_face_values.get_function_values (locally_relevant_solution_v,
						  v_value);
	      u[0] = u_value[0];
	      u[1] = v_value[0];
	      if (fe_face_values.normal_vector (0) * u < -1e-14)
		cell_U->face (face)->set_boundary_id (10); // SET ID 10 to inlet BOUNDARY (10 is an arbitrary number)
	    }
  }

template<int dim>
  void
  MultiPhase<dim>::get_boundary_values_phi (
      std::vector<unsigned int> &boundary_values_id_phi,
      std::vector<double> &boundary_values_phi)
  {
    std::map<unsigned int, double> map_boundary_values_phi;
    unsigned int boundary_id = 0;

    set_boundary_inlet ();
    boundary_id = 10; // inlet
    VectorTools::interpolate_boundary_values (dof_handler_LS, boundary_id,
					      BoundaryPhi<dim> (1.0),
					      map_boundary_values_phi);
    boundary_values_id_phi.resize (map_boundary_values_phi.size ());
    boundary_values_phi.resize (map_boundary_values_phi.size ());
    std::map<unsigned int, double>::const_iterator boundary_value_phi =
	map_boundary_values_phi.begin ();
    for (int i = 0; boundary_value_phi != map_boundary_values_phi.end ();
	++boundary_value_phi, ++i)
      {
	boundary_values_id_phi[i] = boundary_value_phi->first;
	boundary_values_phi[i] = boundary_value_phi->second;
      }

  }

template<int dim>
  void
  MultiPhase<dim>::output_results ()
  {

    output_rho ();
//    output_rho_mine ();
    /* enable output_rho_mine () if you want to see the velocity vectors in the output files. this will produce
     * 100 outfiles if you uses 100 processors, each output file describe the velocity field in its owned reigions.
     * I uses this function to debug. you can also write a pvtu file to visualize them together as descrived in
     * step 40, for example.
     */
    output_number++;

  }

template<int dim>
  void
  MultiPhase<dim>::output_vectors ()
  {
    DataOut < dim > data_out;
    data_out.attach_dof_handler (dof_handler_LS);
    data_out.add_data_vector (locally_relevant_solution_phi, "phi");
    data_out.build_patches ();

    const std::string filename =
	("sol_vectors-" + Utilities::int_to_string (output_number, 3) + "."
	    + Utilities::int_to_string (
		triangulation.locally_owned_subdomain (), 4));
    std::ofstream output ((filename + ".vtu").c_str ());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      {
	std::vector < std::string > filenames;
	for (unsigned int i = 0;
	    i < Utilities::MPI::n_mpi_processes (mpi_communicator); ++i)
	  filenames.push_back (
	      "sol_vectors-" + Utilities::int_to_string (output_number, 3) + "."
		  + Utilities::int_to_string (i, 4) + ".vtu");

	std::ofstream master_output ((filename + ".pvtu").c_str ());
	data_out.write_pvtu_record (master_output, filenames);
      }
  }

template<int dim>
  void
  MultiPhase<dim>::output_rho ()
  {
    DataOut < dim > data_out;
    data_out.attach_dof_handler (dof_handler_LS);
    std::vector < std::string > solution_names;
    solution_names.push_back ("phi");
    data_out.add_data_vector (locally_relevant_solution_phi, solution_names);
    data_out.build_patches ();
    const std::string filename = ("sol_"
	+ Utilities::int_to_string (output_number, 3) + ".vtu");
    data_out.write_vtu_in_parallel (filename.c_str (), mpi_communicator);
// uses write_vtu_in_parallel to write output file parallelly.
    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      std::cout << filename.c_str () << "  is writtern" << std::endl;
  }

template<int dim>
  void
  MultiPhase<dim>::output_rho_mine ()
  {
// look step 35 for more info.
    const FESystem<dim> joint_fe (fe_U, 2, fe_P, 1, fe_LS, 1);
    DoFHandler < dim > joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Vector<double> joint_solution (joint_dof_handler.n_dofs ());
    std::vector<types::global_dof_index> loc_joint_dof_indices (
	joint_fe.dofs_per_cell), loc_vel_dof_indices (fe_U.dofs_per_cell),
	loc_pres_dof_indices (fe_P.dofs_per_cell), loc_phi_dof_indices (
	    fe_LS.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator joint_cell =
	joint_dof_handler.begin_active (), joint_endc =
	joint_dof_handler.end (), vel_cell = dof_handler_U.begin_active (),
	pres_cell = dof_handler_P.begin_active (), phi_cell =
	    dof_handler_LS.begin_active ();
    for (; joint_cell != joint_endc;
	++joint_cell, ++vel_cell, ++pres_cell, ++phi_cell)
      if (joint_cell->is_locally_owned ())
	{
	  joint_cell->get_dof_indices (loc_joint_dof_indices);
	  vel_cell->get_dof_indices (loc_vel_dof_indices);
	  pres_cell->get_dof_indices (loc_pres_dof_indices);
	  phi_cell->get_dof_indices (loc_phi_dof_indices);
	  //		std::cout << "joint_fe.dofs_per_cell=" << joint_fe.dofs_per_cell<< std::endl;
	  for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
	    {
	      int first = joint_fe.system_to_base_index (i).first.first;
	      int second = joint_fe.system_to_base_index (i).first.second;
	      //			std::cout << first << "   " << second << std::endl;
	      switch (joint_fe.system_to_base_index (i).first.first)
		{
		case 0:
		  {
		    if (joint_fe.system_to_base_index (i).first.second == 0)
		      {
			joint_solution (loc_joint_dof_indices[i]) =
			    locally_relevant_solution_u[loc_vel_dof_indices[joint_fe.system_to_base_index (
				i).second]];
		      }
		    else
		      {
			joint_solution (loc_joint_dof_indices[i]) =
			    locally_relevant_solution_v[loc_vel_dof_indices[joint_fe.system_to_base_index (
				i).second]];
		      }
		  }
		  break;
		case 1:
		  joint_solution (loc_joint_dof_indices[i]) =
		      locally_relevant_solution_p (
			  loc_pres_dof_indices[joint_fe.system_to_base_index (i).second]);
		  break;
		case 2:
		  {
		    joint_solution (loc_joint_dof_indices[i]) =
			locally_relevant_solution_phi (
			    loc_phi_dof_indices[joint_fe.system_to_base_index (
				i).second]);
		  }
		  break;
		default:
		  Assert (false, ExcInternalError ());
		}
	    }
	}
//
    std::vector < std::string > joint_solution_names (dim, "velocity");
    joint_solution_names.push_back ("p");
    joint_solution_names.push_back ("phi");
    DataOut < dim > data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    std::vector < DataComponentInterpretation::DataComponentInterpretation
	> component_interpretation (
	    dim + 2, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] =
	DataComponentInterpretation::component_is_scalar;
    component_interpretation[dim + 1] =
	DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector (joint_solution, joint_solution_names,
			      DataOut < dim > ::type_dof_data,
			      component_interpretation);
    data_out.build_patches (2);

    const std::string name (
	"sol_" + Utilities::int_to_string (output_number, 5)
	    + Utilities::int_to_string (
		Utilities::MPI::this_mpi_process (mpi_communicator), 3)
	    + ".vtk");
    std::ofstream output (name);
    data_out.write_vtk (output);
    std::cout << name << " is written" << std::endl;
  }

template<int dim>
  void
  MultiPhase<dim>::run ()
  {
////////////////////////
// GENERAL PARAMETERS //
////////////////////////
    umax = 1;
    cfl = 0.01;
    verbose = true;
    get_output = true;
    output_number = 0;
//  n_refinement=8;
    output_time = 0.1;
    final_time = 10.0;
//////////////////////////////////////////////
// PARAMETERS FOR THE NAVIER STOKES PROBLEM //
//////////////////////////////////////////////
    rho_fluid = 1000.;
    nu_fluid = 1.0;
    rho_air = 1.0;
    nu_air = 1.8e-2;
//PROBLEM=BREAKING_DAM;
    PROBLEM = FILLING_TANK;
//PROBLEM=SMALL_WAVE_PERTURBATION;
//PROBLEM=FALLING_DROP;

    ForceTerms<dim> force_function (std::vector<double>
      { -1.0, 0.0 }); // this is the gravity term, change it if you want to change the gravity direction.

//////////////////////////////////////
// PARAMETERS FOR TRANSPORT PROBLEM //
//////////////////////////////////////
    cK = 1.0;
    cE = 1.0;
    sharpness_integer = 10; //this will be multipled by min_h
//TRANSPORT_TIME_INTEGRATION=FORWARD_EULER;
    TRANSPORT_TIME_INTEGRATION = SSP33;
//ALGORITHM = "MPP_u1";
//ALGORITHM = "NMPP_uH";
    ALGORITHM = "MPP_uH";

// ADJUST PARAMETERS ACCORDING TO PROBLEM
    if (PROBLEM == FALLING_DROP)
      n_refinement = 3;

//////////////
// GEOMETRY //
//////////////
    if (PROBLEM == FILLING_TANK)
      {
	std::vector<unsigned int> repetitions;
	repetitions.push_back (3);
	repetitions.push_back (3);
	GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions,
						   Point < dim > (0.0, 0.0),
						   Point < dim > (1, 1), true);
      }
    else if (PROBLEM == BREAKING_DAM || PROBLEM == SMALL_WAVE_PERTURBATION)
      {
	std::vector<unsigned int> repetitions;
	repetitions.push_back (2);
	repetitions.push_back (1);
	GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions,
						   Point < dim > (0.0, 0.0),
						   Point < dim > (1.0, 0.5),
						   true);
      }
    else if (PROBLEM == FALLING_DROP)
      {
	std::vector<unsigned int> repetitions;
	repetitions.push_back (1);
	repetitions.push_back (4);
	GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions,
						   Point < dim > (0.0, 0.0),
						   Point < dim > (0.3, 0.9),
						   true);
      }
    triangulation.refine_global (4);

//  std::ofstream out ("grid-1.eps");
//  GridOut grid_out;
//  grid_out.write_eps (triangulation, out);
//  std::cout << "Grid written to grid-1.eps" << std::endl;

// uncomment above lines to witre output files for the mesh.

    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      std::cout << "Number of active cells: " << triangulation.n_active_cells ()
	  << "   using processors:"
	  << Utilities::MPI::n_mpi_processes (mpi_communicator) << std::endl;
// SETUP
    setup ();

// PARAMETERS FOR TIME STEPPING
    min_h = GridTools::minimal_cell_diameter (triangulation) / std::sqrt (2);

    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      printf ("cfl * min_h / umax * 5=%f  \n", cfl * min_h / umax * 5);
    time_step = 0.0001; // set the initial time_step size as a small number since it will change based on the largest velocity;
    sharpness = 0.01; // this velue is used to initialize phi profile.
    std::cout << "sharpness =  " << sharpness << std::endl;
// INITIAL CONDITIONS
    initial_condition ();

// NAVIER STOKES SOLVER
    NavierStokesSolver<dim> navier_stokes (degree_LS, degree_U, time_step, eps,
					   rho_air, nu_air, rho_fluid, nu_fluid,
					   force_function, verbose,
					   triangulation, mpi_communicator);
// BOUNDARY CONDITIONS FOR NAVIER STOKES
    get_boundary_values_U ();

    navier_stokes.set_boundary_conditions (boundary_values_id_u,
					   boundary_values_id_v,
					   boundary_values_u,
					   boundary_values_v);

//set INITIAL CONDITION within NAVIER STOKES
    navier_stokes.initial_condition (locally_relevant_solution_phi,
				     locally_relevant_solution_u,
				     locally_relevant_solution_v,
				     locally_relevant_solution_p);

// TRANSPORT SOLVER
    LevelSetSolver<dim> transport_solver (degree_LS, degree_U, time_step, cK,
					  cE, verbose, ALGORITHM,
					  TRANSPORT_TIME_INTEGRATION,
					  triangulation, mpi_communicator);
// BOUNDARY CONDITIONS FOR PHI
    get_boundary_values_phi (boundary_values_id_phi, boundary_values_phi);

    transport_solver.set_boundary_conditions (boundary_values_id_phi,
					      boundary_values_phi);

//    completely_distributed_solution_u = locally_relevant_solution_u;
//    completely_distributed_solution_u.set (boundary_values_id_u,
//					   boundary_values_u);
//    completely_distributed_solution_u.compress (VectorOperation::insert);
//    locally_relevant_solution_u = completely_distributed_solution_u;
//
//    completely_distributed_solution_v = locally_relevant_solution_v;
//    completely_distributed_solution_v.set (boundary_values_id_v,
//					   boundary_values_v);
//    completely_distributed_solution_v.compress (VectorOperation::insert);
//    locally_relevant_solution_v = completely_distributed_solution_v;
//
//    completely_distributed_solution_phi = locally_relevant_solution_phi;
//    completely_distributed_solution_phi.set (boundary_values_id_phi,
//					     boundary_values_phi);
//    completely_distributed_solution_phi.compress (VectorOperation::insert);
//    locally_relevant_solution_phi = completely_distributed_solution_phi;

//	 std::ofstream haha ("locally_relevant_solution_u.txt");
//	locally_relevant_solution_u.print(haha);

//    completely_distributed_solution_phi.print(std::cout);

// enable above lines if you want to output the velocity vectors. I comment them to speed up the simulation.
    output_results ();

//set INITIAL CONDITION within TRANSPORT PROBLEM
    transport_solver.initial_condition (locally_relevant_solution_phi,
					locally_relevant_solution_u,
					locally_relevant_solution_v);
    int dofs_U = 2 * dof_handler_U.n_dofs ();
    int dofs_P = 2 * dof_handler_P.n_dofs ();
    int dofs_LS = dof_handler_LS.n_dofs ();
    int dofs_TOTAL = dofs_U + dofs_P + dofs_LS;

// NO BOUNDARY CONDITIONS for LEVEL SET

    if (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)
      {
	pcout << "Cfl: " << cfl << "; umax: " << umax << "; min h: " << min_h
	    << "; time step: " << time_0 << std::endl;
	pcout << "   Number of active cells:       "
	    << triangulation.n_global_active_cells () << std::endl
	    << "   Number of degrees of freedom: " << std::endl << "      U: "
	    << dofs_U << std::endl << "      P: " << dofs_P << std::endl
	    << "      LS: " << dofs_LS << std::endl << "      TOTAL: "
	    << dofs_TOTAL << std::endl;

      }

// TIME STEPPING
    int every = 1;
    int P, P_n;
    MPI_Comm_rank (mpi_communicator, &P);
    MPI_Comm_size (mpi_communicator, &P_n);
    MPI_Status status;
    double umax_0;
    double umax_max_0 = 0;
    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);

    cfl = 0.15 * 0.008 / min_h; // change the values of cfl to adjust the time_step_size
    int N = triangulation.n_global_active_cells ();
    for (timestep_number = 1, time_0 = time_step; timestep_number <= 100000;
	time_0 += time_step, ++timestep_number)
      {
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	if (P == 0)
	  printf (
	      "N= %d; current step is: %d;  at t= %f;  time_step=%f;  umax= %f; current time is %s \n",
	      N, timestep_number, time_0, time_step, umax_max_0,
	      asctime (timeinfo));

	double local_umax_i = get_u_max ();

	if (P != 0)
	  {
	    MPI_Send (&local_umax_i, 1, MPI_DOUBLE, 0, 0, mpi_communicator);

	  }
	if (P == 0)
	  {

	    for (int i = 1; i < P_n; i++)
	      {
		MPI_Recv (&umax_0, 1, MPI_DOUBLE, i, 0, mpi_communicator,
			  &status);
		umax_max_0 = (umax_max_0 > umax_0 ? umax_max_0 : umax_0);
	      }
	    umax_max_0 = (umax_max_0 > 0.15 ? umax_max_0 : 0.15);
	    double time_step_0 = cfl * min_h / umax_max_0;
	    for (int i = 1; i < P_n; i++)
	      {
		MPI_Send (&time_step_0, 1, MPI_DOUBLE, i, 0, mpi_communicator);
	      }
	    time_step = time_step_0;
	    printf ("infomation send from P0 done! \n");
	  }
	if (P != 0)
	  {
	    MPI_Recv (&time_step, 1, MPI_DOUBLE, 0, 0, mpi_communicator,
		      &status);
	  }

	/* these lines transfer the local u_max to processor 0 and transfer the
	 * new time step size back to all processors.
	 */

	// GET NAVIER STOKES VELOCITY
	navier_stokes.set_time_step (time_step);
	navier_stokes.set_phi (locally_relevant_solution_phi);

	for (int i = 0; i < P_n; i++)
	  {
	    if (P == i)
	      {
		time (&rawtime);
		timeinfo = localtime (&rawtime);
		printf ("P %d ready to solve linear system time = %s\n", P,
			asctime (timeinfo));

	      }
	    MPI_Barrier (mpi_communicator);

	  }

	navier_stokes.nth_time_step (); // solve the linear system here.
	navier_stokes.get_velocity (locally_relevant_solution_u,
				    locally_relevant_solution_v);

	for (int i = 0; i < P_n; i++)
	  {
	    if (P == i)
	      {
		time (&rawtime);
		timeinfo = localtime (&rawtime);
		printf (
		    "P %d solved linear system and copied data to main; time = %s\n",
		    P, asctime (timeinfo));
	      }
	    MPI_Barrier (mpi_communicator);

	  }

	transport_solver.set_time_step (time_step);
	transport_solver.set_velocity (locally_relevant_solution_u,
				       locally_relevant_solution_v);
	// GET LEVEL SET SOLUTION
	transport_solver.nth_time_step (); // solve the linear system of phi here
	transport_solver.get_unp1 (locally_relevant_solution_phi);
	if (every % 5 == 0)   // write the output file every  time step.
	  output_results ();
	every++;
      }
    navier_stokes.get_velocity (locally_relevant_solution_u,
				locally_relevant_solution_v);
    transport_solver.get_unp1 (locally_relevant_solution_phi);
    if (get_output)
      output_results ();
  }

int
main (int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);
      PetscInitialize (&argc, &argv, PETSC_NULL, PETSC_NULL);
      deallog.depth_console (0);
	{
	  unsigned int degree_LS = 1;
	  unsigned int degree_U = 2;
	  MultiPhase<2> multi_phase (degree_LS, degree_U);
	  multi_phase.run ();
	  std::cout << "calculation done" << std::endl;
	}

      PetscFinalize ();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
	  << "----------------------------------------------------"
	  << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what ()
	  << std::endl << "Aborting!" << std::endl
	  << "----------------------------------------------------"
	  << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
	  << "----------------------------------------------------"
	  << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
	  << "----------------------------------------------------"
	  << std::endl;
      return 1;
    }
  return 0;
}
