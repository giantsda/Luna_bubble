template<int dim>
  void
  NavierStokesSolver<dim>::solve_U (
      const ConstraintMatrix &constraints,
      PETScWrappers::MPI::SparseMatrix &Matrix,
      std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
      PETScWrappers::MPI::Vector &completely_distributed_solution,
      const PETScWrappers::MPI::Vector &rhs)
  {
    SolverControl solver_control (dof_handler_U.n_dofs (), 1e-6);
    //  PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
    //  PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
    //  PETScWrappers::SolverChebychev solver(solver_control, mpi_communicator);
    printf ("A is %d by %d \n", dof_handler_U.n_dofs (),
	    dof_handler_U.n_dofs ());
    PETScWrappers::SolverBicgstab solver (solver_control, mpi_communicator);
    constraints.distribute (completely_distributed_solution);
    int de;

/*
    rhs.print(std::cout);
    std::ofstream out1 ("A.txt");
    Matrix.print (out1);
    FILE *file2;
    file2 = fopen ("b.txt", "w");
    for (int i=0;i< dof_handler_U.n_dofs ();i++)
    fprintf (file2, "%f \n",rhs[i]);
    fclose (file2);
*/
 
    ///////////////////////////////////////////////////////////////////////////

    KSP ksp;
    PC pc;
    Mat A, M;
    Vec X, B, D;
    KSPConvergedReason reason;
    PetscInt its;
    PetscErrorCode ierr;
    int N = dof_handler_U.n_dofs ();

    int rows_total = Matrix.m ();
    int columns_total = Matrix.n ();
    printf ("n_nonzero_elements is %d\n,", Matrix.n_nonzero_elements ());
    printf ("%d    %d  \n", rows_total, columns_total);
    int nonzeroguess = Matrix.n_nonzero_elements () / rows_total * 10;

    //
    //    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler_U.n_dofs(),
    //						    dof_handler_U.n_dofs());
    //    DoFTools::make_sparsity_pattern (dof_handler_U, dynamic_sparsity_pattern);
    //    SparsityPattern sparsity_pattern;
    //    sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    //    std::ofstream out ("sparsity_pattern1.svg");
    //    sparsity_pattern.print_svg (out);
    //
    //    typedef types::global_dof_index size_type;
    //    const std::vector<size_type> a{300};
    //
    //    PETScWrappers::MPI::SparseMatrix haha(mpi_communicator,sparsity_pattern,a,a,0);

    MatCreateSeqAIJ (mpi_communicator, N, N, nonzeroguess, NULL, &A);
    VecCreateSeq (mpi_communicator, N, &B);
    VecDuplicate (B, &X);
    typedef types::global_dof_index size_type;
    std::pair < PETScWrappers::MatrixBase::size_type, PETScWrappers::MatrixBase::size_type
	> loc_range = Matrix.local_range ();
    PetscInt ncols;
    const PetscInt *colnums;
    const PetscScalar *values;
    PETScWrappers::MatrixBase::size_type row;
    for (row = loc_range.first; row < loc_range.second; ++row)
      {
	PetscErrorCode ierr = MatGetRow (Matrix, row, &ncols, &colnums,
					 &values);
	AssertThrow (ierr == 0, ExcPETScError (ierr));
	for (PetscInt col = 0; col < ncols; ++col)
	  {
	    int x_row = row, x_col = colnums[col];
	    double va = values[col];
	    MatSetValues (A, 1, &x_row, 1, &x_col, &va, INSERT_VALUES);
	  }
	ierr = MatRestoreRow (Matrix, row, &ncols, &colnums, &values);
      }

    PetscScalar *val;
    PetscInt nlocal, istart, iend;
    VecGetArray (rhs, &val);
    VecGetLocalSize (rhs, &nlocal);
    VecGetOwnershipRange (rhs, &istart, &iend);
    for (unsigned int i = 0;
	i < Utilities::MPI::n_mpi_processes (mpi_communicator); i++)
      {
	const int mpi_ierr = MPI_Barrier (mpi_communicator);
	AssertThrowMPI (mpi_ierr);
	if (i == Utilities::MPI::this_mpi_process (mpi_communicator))
	  {
	    for (int i = 0; i < nlocal; ++i)
	      {
		double value = val[i];
		VecSetValues (B, 1, &i, &value, INSERT_VALUES);
	      }
	  }
      }

    // solveing
    MatAssemblyBegin (A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd (A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin (B);
    VecAssemblyEnd (B);
    VecAssemblyBegin (rhs);
    VecAssemblyEnd (rhs);
    PetscPrintf (PETSC_COMM_WORLD, "\nThe Kershaw matrix:\n\n");
    //    printf ("----------------\n");
    //    printf ("A=\n");
    //    MatView (A, PETSC_VIEWER_STDOUT_WORLD);
    //    printf ("X=\n");
    //    VecView (X, PETSC_VIEWER_STDOUT_WORLD);
    //    printf ("B=\n");
    //    VecView (B, PETSC_VIEWER_STDOUT_WORLD);
    //    printf ("----------------\n");
    KSPCreate (mpi_communicator, &ksp);
    KSPSetOperators (ksp, A, A);

    KSPSetType (ksp, KSPBCGS);
    KSPSetInitialGuessNonzero (ksp, PETSC_TRUE);
    KSPGetPC (ksp, &pc);
    PCSetType (pc, PCICC);

    KSPSetFromOptions (ksp);
    KSPSetUp (ksp);
    PCFactorGetMatrix (pc, &M);
    VecDuplicate (B, &D);
    MatGetDiagonal (M, D);

    KSPSolve (ksp, B, X);
    KSPGetConvergedReason (ksp, &reason);
    if (reason == KSP_DIVERGED_INDEFINITE_PC)
      {
	PetscPrintf (PETSC_COMM_WORLD,
		     "\nDivergence because of indefinite preconditioner;\n");
	PetscPrintf (
	    PETSC_COMM_WORLD,
	    "Run the executable again but with -pc_factor_shift_positive_definite option.\n");
      }
    else if (reason < 0)
      {
	ierr = PetscPrintf (
	    PETSC_COMM_WORLD,
	    "\nOther kind of divergence: this should not happen.\n");
      }
    else
      {
	KSPGetIterationNumber (ksp, &its);
	PetscPrintf (PETSC_COMM_WORLD, "\nConvergence in %d iterations.\n",
		     (int) its);
      }

    VecAssemblyBegin (completely_distributed_solution);
    VecAssemblyEnd (completely_distributed_solution);

    for (int i = 0; i < N; ++i)
      {
	double val;
	VecGetValues (X, 1, &i, &val);
	//	printf ("<<<%f \n", val);
	std::vector<size_type> haha1 =
	  { i };
	std::vector<PetscScalar> haha2 =
	  { val};
	completely_distributed_solution.set (haha1, haha2);

      }
    completely_distributed_solution.compress (VectorOperation::insert);
    KSPDestroy (&ksp);
    MatDestroy (&A);
    VecDestroy (&B);
    VecDestroy (&X);
    VecDestroy (&D);

    ///////////////////////
    constraints.distribute (completely_distributed_solution);
    printf ("completely_distributed_solution:::::::::::::::::::::::::::::\n");
    completely_distributed_solution.print (std::cout);
    std::cin >> de;
    if (solver_control.last_step () > MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
      rebuild_Matrix_U_preconditioners = true;
    if (verbose == true)
      pcout << "   Solved U in " << solver_control.last_step ()
	  << " iterations." << std::endl;
  }
