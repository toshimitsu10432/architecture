#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <x86intrin.h>


#define BLOCKSIZE 32
#define UNROLL 4

#define FP_SINGLE     /* Data Size: float -> ON */

#if defined(FP_SINGLE)
#define REAL float		/* Data Size: float -> ON */
#else
#define REAL double		/* Data Size: double(default) -> ON */
#endif

#define BLOCKING		/* blocking -> ON */
#define OMP                           /* OpenMP -> ON */


/* Unoptimized */
void
dgemm_unopt (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;
  REAL cij;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	cij = C[i + j * n];	/* cij = C[i][j] */
	for (k = 0; k < n; k++)
	  cij += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
	C[i + j * n] = cij;	/* C[i][j] = cij */
      }
}


/* Blocking &  Blocking+AVX */
void
do_block (int n, int si, int sj, int sk, REAL * A, REAL * B, REAL * C)
{
  int i, j, k;
  REAL cij;
    #pragma omp parallel for
  for (i = si; i < si + BLOCKSIZE; ++i)
    {
      for (j = sj; j < sj + BLOCKSIZE; ++j)
	{
	  cij = C[i + j * n];	/* cij = C[i][j] */
	  for (k = sk; k < sk + BLOCKSIZE; k++)
	    cij += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
	  C[i + j * n] = cij;	/* C[i][j] = cij */
	}
    }
}


void
dgemm_blocking (REAL * A, REAL * B, REAL * C, int n)
{
  int sj, si, sk;
    #pragma omp parallel for
  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
	do_block (n, si, sj, sk, A, B, C);
}

/* Timer */
double
seconds ()
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  return (double) tv.tv_sec + ((double) tv.tv_usec) / 1000000.0;
}


/* init matrics */
void
int_mat (REAL * A, REAL * B, REAL * C, int N)
{
  int i, j;

  srand (1);

  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      {
	A[i + j * N] = (REAL) rand () / (10000 + i + j);
	B[i + j * N] = (REAL) rand () / (10000 + i + j);
	C[i + j * N] = (REAL) 0.0;
      }
}


/* Check calculation*/
int
check_mat (REAL * C, REAL * C_unopt, int N)
{
  int n, m;
  double max_err=1.0e-5;

  for (n = 0; n < N; n++)
    {
      for (m = 0; m < N; m++)
	{
	  if (fabs ((C[n + N * m] - C_unopt[n + N * m])/C_unopt[n + N * m]) > max_err)
	    {
	      printf("Error:   result is different in %d,%d  (%.5f, %.5f) delta %.5f > max_err %.5f \n",
		 n, m, C[n + N * m], C_unopt[n + N * m],
		 fabs (C[n + N * m] - C_unopt[n + N * m]), max_err);
	    }
	}
    }
}


/*** Main(Matrix calculation) ***/
int main (int argc, char *argv[]) {
  REAL *A, *B, *C, *C_unopt;
  int N;			/* N=matrix size */
  int itr;			/* Number of iterations */
  int i;
  double t;

  if (argc < 3)
    {
      fprintf (stderr, "Specify M, #ITER\n");
      /* Argument 1:Array size, Argument 2:Number of iterations */
      exit (1);
    }

  N = atoi (argv[1]);		/* Argument 1:Array size */
  if (N % 8 != 0)
    {
      printf ("Please specify N that is a multipe of 8 for AVX 256 bit\n");
      return 0;
    }

  if (N % BLOCKSIZE != 0)
    {
      printf
	("Please specify N that is a multipe of BLOCKSIZE(%d) for Blocking\n",
	 BLOCKSIZE);
      return 0;
    }


  itr = atoi (argv[2]);		/* Argument 2:Number of iterations */


	/** memory set **/
  A = (REAL *) malloc (N * N * sizeof (REAL));
  B = (REAL *) malloc (N * N * sizeof (REAL));
  C = (REAL *) malloc (N * N * sizeof (REAL));
  C_unopt = (REAL *) malloc (N * N * sizeof (REAL));

	/** calculation **/
  for (i = 0; i < itr; ++i)
    {
      /*unoptimized */
      int_mat (A, B, C_unopt, N);
      t = seconds ();
      dgemm_unopt (A, B, C_unopt, N);
      t = seconds () - t;
      printf ("\n%f [s]  GFLOPS %f  |unoptimized| \n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);


      /*blocking */
#ifdef BLOCKING
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif
    }





	/** memory free **/
  free (A);
  free (B);
  free (C);
  free (C_unopt);

  return 0;
}


