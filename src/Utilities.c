
#include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_heapsort.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_vector.h"

#include "R.h"
#include "Rmath.h"

#include "MBVSfa.h"
#include "MBVSus.h"

#define Pi 3.141592653589793238462643383280


/********* For mvnBVS-US model *************/

double sumCorus_j(gsl_matrix *Sigma,
                  gsl_matrix *gamma,
                  int j,
                  int k)
{
    int p = gamma -> size2;    
    double val = 0;
    int r;
    
    for(r = 0; r < p; r++)
    {
        if(r != j)
        {
            val += fabs(gsl_matrix_get(Sigma, j, r)) / sqrt(gsl_matrix_get(Sigma, j, j)) / sqrt(gsl_matrix_get(Sigma, r, r)) * gsl_matrix_get(gamma, k, r);
        }
    }
    return val;
}


/*
 Random generation from the Inverse Wishart distribution
 */

void c_riwishart(int v,
                 gsl_matrix *X_ori,
                 gsl_matrix *sample)
{
    int i, j, df;
    double normVal;
    
    int p = X_ori->size1;
    
    gsl_matrix *X = gsl_matrix_calloc(p, p);
    matrixInv(X_ori, X);
    
    gsl_matrix *cholX = gsl_matrix_calloc(p, p);
    gsl_matrix *ZZ = gsl_matrix_calloc(p, p);
    gsl_matrix *XX = gsl_matrix_calloc(p, p);
    gsl_matrix *KK = gsl_matrix_calloc(p, p);
    
    gsl_matrix_memcpy(cholX, X);
    gsl_linalg_cholesky_decomp(cholX);
    
    for(i = 0; i < p; i ++)
    {
        for(j = 0; j < i; j ++)
        {
            gsl_matrix_set(cholX, i, j, 0);
        }
    }
    
    for(i = 0; i < p; i++)
    {
        df = v - i;
        gsl_matrix_set(ZZ, i, i, sqrt(rchisq(df)));
    }
    
    for(i = 0; i < p; i++)
    {
        for(j = 0; j < i; j ++)
        {
            normVal = rnorm(0, 1);
            gsl_matrix_set(ZZ, i, j, normVal);
        }
    }
        
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, ZZ, cholX, 0, XX);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, XX, XX, 0, KK);
    matrixInv(KK, sample);
    
    gsl_matrix_free(X);
    gsl_matrix_free(cholX);
    gsl_matrix_free(XX);
    gsl_matrix_free(ZZ);
    gsl_matrix_free(KK);
}




/********* For mvnBVS-FA model *************/

double sumCorfa_j(gsl_vector *lambda,
                  gsl_matrix *gamma,
                  int j,
                  int k)
{
    int p = gamma -> size2;
    double val = 0;
    int r;
    
    for(r = 0; r < p; r++)
    {
        if(r != j)
        {
            val += fabs(gsl_vector_get(lambda, j)) * fabs(gsl_vector_get(lambda, r)) / sqrt(pow(gsl_vector_get(lambda, j), 2) + 1) / sqrt(pow(gsl_vector_get(lambda, r), 2) + 1) * gsl_matrix_get(gamma, k, r);
        }
    }
    return val;
}


/********* Common functions *************/


double c_det(gsl_matrix *A)
{
    int signum, K = A->size1;
    double value;
    
    gsl_matrix *ALU = gsl_matrix_alloc(K, K);
    gsl_permutation *p     = gsl_permutation_alloc(K);
    gsl_matrix_memcpy(ALU, A);
    gsl_linalg_LU_decomp(ALU, p, &signum);
    
    
    value = gsl_linalg_LU_det(ALU, signum);
    
    
    gsl_matrix_free(ALU);
    gsl_permutation_free(p);
    
    
    return value;
}


/*
 Evaluate the quadratic form: v^T M^{-1} v
 */
void c_quadform_vMv(gsl_vector *v,
                    gsl_matrix *Minv,
                    double     *value)
{
    int    d = v->size;
    gsl_vector *tempVec = gsl_vector_calloc(d);
    
    gsl_blas_dsymv(CblasUpper, 1, Minv, v, 0, tempVec);
    gsl_blas_ddot(v, tempVec, value);
    
    gsl_vector_free(tempVec);
    return;
}


/*
 Evaluate the quadratic form: v^T M^{-1} u
 - note v and u are assumed to be of the same length
 */
void c_quadform_vMu(gsl_vector *v,
                    gsl_matrix *Minv,
                    gsl_vector *u,
                    double     *value)
{
    int    d = v->size;
    gsl_vector *tempVec = gsl_vector_calloc(d);
    
    gsl_blas_dsymv(CblasUpper, 1, Minv, u, 0, tempVec);
    gsl_blas_ddot(v, tempVec, value);
    
    gsl_vector_free(tempVec);
    return;
}


/*
 Evaluate the inverse of the matrix X
 */
void matrixInv(gsl_matrix *X, gsl_matrix *Xinv)
{
    int signum;
    int d = X->size1;
    gsl_matrix      *XLU = gsl_matrix_calloc(d, d);
    gsl_permutation *p   = gsl_permutation_alloc(d);
    
    gsl_matrix_memcpy(XLU, X);
    gsl_linalg_LU_decomp(XLU, p, &signum);
    gsl_linalg_LU_invert(XLU, p, Xinv);
    
    gsl_matrix_free(XLU);
    gsl_permutation_free(p);
    return;
}


/*
 Calculating column sums of matrix X
 */
void c_colSums(gsl_matrix *X, gsl_vector *v)
{
    int numCol = X->size2;
    int numRow = X->size1;
    int i, j;
    double sum = 0;
    for(j = 0; j < numCol; j++)
    {
        i = 0;
        while(i < numRow)
        {
            sum = sum + gsl_matrix_get(X, i, j);
            i++;
        }
        gsl_vector_set(v, j, sum);
        sum = 0;
    }
    return;
}


/*
 Calculating row sums of matrix X
 */
void c_rowSums(gsl_matrix *X, gsl_vector *v)
{
    int numCol = X->size2;
    int numRow = X->size1;
    int i, j;
    double sum = 0;
    for(i = 0; i < numRow; i++)
    {
        j = 0;
        while(j < numCol)
        {
            sum = sum + gsl_matrix_get(X, i, j);
            j++;
        }
        gsl_vector_set(v, i, sum);
        sum = 0;
    }
    return;
}


/*
 Replicate a vector v into rows of a matrix X
 */
void c_repVec_Rowmat(gsl_vector *v, gsl_matrix *X)
{
    int length = v->size;
    int numRep = X->size1;
    int i, j;
    for(i = 0; i < numRep; i++)
    {
        for(j = 0; j < length; j++)
        {
            gsl_matrix_set(X, i, j, gsl_vector_get(v, j));
        }
    }
    return;
}

/*
 Replicate a vector v into columns of a matrix X
 */
void c_repVec_Colmat(gsl_vector *v, gsl_matrix *X)
{
    int length = v->size;
    int numRep = X->size2;
    int i, j;
    for(j = 0; j < numRep; j++)
    {
        for(i = 0; i < length; i++)
        {
            gsl_matrix_set(X, i, j, gsl_vector_get(v, i));
        }
    }
    return;
}

/*
 Minimum of two numbers
 */
double c_min(double value1,
             double value2)
{
    double min = (value1 <= value2) ? value1 : value2;
    return min;
}


/*
 Maximum of two numbers
 */
double c_max(double value1,
             double value2)
{
    double max = (value1 >= value2) ? value1 : value2;
    return max;
}


/*
 Evaluate the inverse of the matrix M
 */
void c_solve(gsl_matrix *M,
             gsl_matrix *Minv)
{
    int signum;
    int d = M->size1;
    gsl_matrix      *MLU = gsl_matrix_calloc(d, d);
    gsl_permutation *p   = gsl_permutation_alloc(d);
    
    gsl_matrix_memcpy(MLU, M);
    gsl_linalg_LU_decomp(MLU, p, &signum);
    gsl_linalg_LU_invert(MLU, p, Minv);
    
    gsl_matrix_free(MLU);
    gsl_permutation_free(p);
    return;
}

/*
 Random number generation for inverse gamma distribution
 alpha = shape, beta = rate
 
 */
void c_rigamma(double *temp,
               double alpha,
               double beta)
{
    double shape = alpha;
    double scale = (double) 1/beta;
    double gam=1;
    
    if(alpha > 0 && beta > 0){
        gam = rgamma(shape, scale);
    }
    *temp = (double) 1/gam;
    return;
}


/*
 Random number generation for multivariate normal distribution
 mean (n)
 Var (n x n)
 sample (numSpl x n)
 */
void c_rmvnorm(gsl_matrix *sample,
               gsl_vector *mean,
               gsl_matrix *Var)
{
    int n = sample->size2;
    int numSpl = sample->size1;
    int i, j;
    double spl;
    
    gsl_matrix *temp = gsl_matrix_alloc(n, n);
    
    gsl_matrix_memcpy(temp, Var);
    gsl_linalg_cholesky_decomp(temp);
    
    for(i = 0; i < n; i ++){
        for(j = 0; j < n; j++){
            if(i > j){
                gsl_matrix_set(temp, i, j, 0);
            }
        }
    }
    
    for(i = 0; i < numSpl; i ++){
        for(j = 0; j < n; j ++){
            spl = rnorm(0, 1);
            gsl_matrix_set(sample, i, j, spl);
        }
    }
    
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1, temp, sample);
    
    for(i = 0; i < numSpl; i++){
        gsl_vector_view sampleRow = gsl_matrix_row(sample, i);
        gsl_vector_add(&sampleRow.vector, mean);
    }
    
    gsl_matrix_free(temp);
    return;
}


/*
 Density calculation for a multivariate normal distribution
 */
void c_dmvnorm(gsl_vector *x,
               gsl_vector *mu,
               double     sigma,
               gsl_matrix *AInv,
               double     *value)
{
    int signum, K = x->size;
    double sigmaSqInv = pow(sigma, -2);
    
    gsl_vector *diff       = gsl_vector_alloc(K);
    gsl_matrix *SigmaInv   = gsl_matrix_alloc(K, K);
    gsl_matrix *SigmaInvLU = gsl_matrix_alloc(K, K);
    gsl_permutation *p     = gsl_permutation_alloc(K);
    
    gsl_vector_memcpy(diff, x);
    gsl_vector_sub(diff, mu);
    
    gsl_matrix_memcpy(SigmaInv, AInv);
    gsl_matrix_scale(SigmaInv, sigmaSqInv);
    gsl_matrix_memcpy(SigmaInvLU, SigmaInv);
    gsl_linalg_LU_decomp(SigmaInvLU, p, &signum);
    
    c_quadform_vMv(diff, SigmaInv, value);
    
    *value = (log(gsl_linalg_LU_det(SigmaInvLU, signum)) - log(pow(2*Pi, K)) - *value) / 2;
    
    gsl_vector_free(diff);
    gsl_matrix_free(SigmaInv);
    gsl_matrix_free(SigmaInvLU);
    gsl_permutation_free(p);
    return;
}

double logit(double val)
{
    double val2 = log(val) - log(1-val);
    return val2;
}

double invLogit(double val)
{
    double val2 = (double) exp(val)/(1+exp(val));
    return val2;
}
double one_invLogit(double val)
{
    if(val > 700) val = 700;
    double val2 = (double) 1/(1+exp(val));
    return val2;
}




