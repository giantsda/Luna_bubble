#include <iostream>
#include <petscksp.h>
#include <stdio.h>

// run it using   g++ debug_petsc.cc -I /home/chen/Desktop/software/petsc/petsc-3.5.4/include -I /home/chen/Desktop/software/petsc/petsc-3.5.4/x86_64/include -I /usr/lib/openmpi/include -L /home/chen/Desktop/software/petsc/petsc-3.5.4/x86_64/lib -lpetsc -g

// This is the program to debug the code that solves the linear system.

int
main (int argc, char **args)
{

  PetscInitialize (&argc, &args, PETSC_NULL, PETSC_NULL);
  KSP ksp;
  PC pc;
  Mat A, M;
  Vec X, B, D;
  KSPConvergedReason reason;
  PetscInt its, Istart, Iend;
  PetscErrorCode ierr;

  int de, N = 49, nonzeroguess = N;

  MatCreate (PETSC_COMM_WORLD, &A);

  MatSetSizes (A, PETSC_DECIDE, PETSC_DECIDE, N, N);

  MatSetFromOptions (A);
  MatMPIAIJSetPreallocation (A, nonzeroguess, NULL, nonzeroguess, NULL);
//  MatGetOwnershipRange(A,&Istart,&Iend);

  VecCreate (PETSC_COMM_WORLD, &X);
  VecSetSizes (X, PETSC_DECIDE, N);
  VecSetFromOptions (X);
  VecDuplicate (X, &B);

  FILE *file;
  file = fopen ("A.txt", "r");
  if (file == NULL)
    {
      fprintf (stderr, "Can't open input file in.list!\n");
      return 1;
    }
  char buff[255];
  int a, b, line = 0;
  double c;
  while (fgets (buff, 255, (FILE*) file))
    {
      //printf ("%s\n", buff);
      sscanf (buff, " (%d,%d) %lf", &a, &b, &c);
      //printf ("reading:    %d   %d   %5.20e  \n", a, b, c);
      MatSetValues (A, 1, &a, 1, &b, &c, INSERT_VALUES);
    }
  fclose (file);

  file = fopen ("b.txt", "r");
  if (file == NULL)
    {
      fprintf (stderr, "Can't open input file in.list!\n");
      return 1;
    }

  while (fgets (buff, 255, (FILE*) file))
    {
      //printf ("%s\n", buff);
      sscanf (buff, "%lf", &c);
      printf ("reading:%d:   %5.20e  \n", line, c);
      VecSetValues (B, 1, &line, &c, INSERT_VALUES);
      line++;
    }
  fclose (file);

  MatAssemblyBegin (A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd (A, MAT_FINAL_ASSEMBLY);
  VecAssemblyBegin (B);
  VecAssemblyEnd (B);
  // solveing
  KSPCreate (PETSC_COMM_WORLD, &ksp);
  KSPSetOperators (ksp, A, A);
  KSPSetType (ksp, KSPBCGS);
  KSPSetInitialGuessNonzero (ksp, PETSC_TRUE);
  KSPGetPC (ksp, &pc);
  PCSetType (pc, PCJACOBI);  // change it
  KSPSetFromOptions (ksp);
  KSPSetUp (ksp);

  printf ("----------------\n");
  printf ("A=\n");
  MatView (A, PETSC_VIEWER_STDOUT_WORLD);
  printf ("B=\n");
  VecView (B, PETSC_VIEWER_STDOUT_WORLD);
  printf ("----------------\n");
  std::cin >> de;
  KSPSolve (ksp, B, X);
  printf ("HEREIS THE SOLUTION=\n");
  VecView (X, PETSC_VIEWER_STDOUT_WORLD);
  KSPDestroy (&ksp);
  MatDestroy (&A);
  VecDestroy (&B);
  VecDestroy (&X);

  PetscFinalize ();

  return 0;
}

