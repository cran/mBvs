
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
#include "mzipBVS_general.h"
#include "mzip_restricted1.h"
#include "mzip_restricted2.h"
#include "MMZIP.h"

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

void c_riwishart(double v,
                 gsl_matrix *X_ori,
                 gsl_matrix *sample,
                 gsl_matrix *sampleInv)
{
    int i, j;
    double df;
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
        df = v - (double) i;
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
    gsl_matrix_memcpy(sampleInv, KK);
    
    gsl_matrix_free(X);
    gsl_matrix_free(cholX);
    gsl_matrix_free(XX);
    gsl_matrix_free(ZZ);
    gsl_matrix_free(KK);
}

void c_riwishart2(double v,
                  gsl_matrix *X_ori,
                  gsl_matrix *sample,
                  gsl_matrix *sampleInv)
{
    int i, j;
    double df;
    double normVal;
    
    int p = X_ori->size1;
    gsl_matrix *X = gsl_matrix_calloc(p, p);
    
    gsl_matrix *cholX = gsl_matrix_calloc(p, p);
    gsl_matrix *ZZ = gsl_matrix_calloc(p, p);
    gsl_matrix *XX = gsl_matrix_calloc(p, p);
    gsl_matrix *KK = gsl_matrix_calloc(p, p);
    
    gsl_vector *diag = gsl_vector_calloc(p);
    for(i = 0; i < p; i++) gsl_vector_set(diag, i, gsl_matrix_get(X_ori, i, i));
    double mod2 = c_min(0.01, fabs(gsl_vector_min(diag))*2);
    
    for(i = 0; i < p; i++) gsl_matrix_set(X_ori, i, i, gsl_matrix_get(X_ori, i, i)+mod2);
    matrixInv(X_ori, X);
    
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
        df = v - (double) i;
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
    gsl_matrix_memcpy(sampleInv, KK);
    
    gsl_matrix_free(X);
    gsl_matrix_free(cholX);
    gsl_matrix_free(XX);
    gsl_matrix_free(ZZ);
    gsl_matrix_free(KK);
    gsl_vector_free(diag);
}

void c_riwishart3(double v,
                  gsl_matrix *X_ori,
                  gsl_matrix *sample,
                  gsl_matrix *sampleInv,
                  gsl_matrix *cholX)
{
    int i, j;
    double df;
    double normVal;
    
    int p = X_ori->size1;
    
    gsl_matrix *X = gsl_matrix_calloc(p, p);
    matrixInv(X_ori, X);
    
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
        df = v - (double) i;
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
    gsl_matrix_memcpy(sampleInv, KK);
    
    gsl_matrix_free(X);
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

void c_rmvnorm2(gsl_matrix *sample,
                gsl_vector *mean,
                gsl_matrix *chol)
{
    int n = sample->size2;
    int numSpl = sample->size1;
    int i, j;
    double spl;
    
    for(i = 0; i < numSpl; i ++){
        for(j = 0; j < n; j ++){
            spl = rnorm(0, 1);
            gsl_matrix_set(sample, i, j, spl);
        }
    }
    
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1, chol, sample);
    
    for(i = 0; i < numSpl; i++){
        gsl_vector_view sampleRow = gsl_matrix_row(sample, i);
        gsl_vector_add(&sampleRow.vector, mean);
    }
    
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


/*
 Removing the i-th column and the i-th row from a matrix
 q X q -> (q-1) X (q-1)
 */
void removeRowColumn(gsl_matrix *R, int i, gsl_matrix *R_sub)
{
    int q = R->size1;
    int j, k;
    
    gsl_matrix_set_zero(R_sub);
    
    for(j = 0; j < q-1; j++)
    {
        for(k = 0; k < q-1; k++)
        {
            if(j < i && k < i)
            {
                gsl_matrix_set(R_sub, j, k, gsl_matrix_get(R, j, k));
            }else if(j < i && k >= i)
            {
                gsl_matrix_set(R_sub, j, k, gsl_matrix_get(R, j, k+1));
            }else if(j >= i && k < i)
            {
                gsl_matrix_set(R_sub, j, k, gsl_matrix_get(R, j+1, k));
            }else if(j >= i && k >= i)
            {
                gsl_matrix_set(R_sub, j, k, gsl_matrix_get(R, j+1, k+1));
            }
        }
    }
    
    return;
}



/*
 Getting the i-th column vector with i-th element removed
 q X q -> (q-1)
 */
void Get_subColumnVector(gsl_matrix *R, int i, gsl_vector *R_subvec)
{
    int q = R->size1;
    int j;
    
    gsl_vector_set_zero(R_subvec);
    
    for(j = 0; j < q-1; j++)
    {
        if(j < i)
        {
            gsl_vector_set(R_subvec, j, gsl_matrix_get(R, i, j));
        }else
        {
            gsl_vector_set(R_subvec, j, gsl_matrix_get(R, i, j+1));
        }
    }
    return;
}


/*
 Random generation from truncated normal distribution
 */
void c_rtnorm(double mean,
              double sd,
              double LL,
              double UL,
              int LL_neginf,
              int UL_posinf,
              double *value)
{
    int caseNO;
    int stop=0;
    double y, a, z, u, val, rho;
    
    LL = (LL-mean)/sd;
    UL = (UL-mean)/sd;
    
    if((LL_neginf == 1 && UL_posinf == 1) || (LL_neginf == 0 && LL < 0 && UL_posinf == 1) || (LL_neginf == 1 && UL >0 && UL_posinf == 0) || (UL_posinf == 0 && LL_neginf == 0 && LL <0 && UL > 0  && (UL-LL) > sqrt(2* Pi)))
    {
        caseNO = 1;
    }else if((LL >= 0 && UL > LL + 2*sqrt(exp(1))/(LL+sqrt(pow(LL,2)+4))*exp((2*LL-LL*sqrt(pow(LL, 2)+4))/4) && UL_posinf == 0 && LL_neginf == 0) || (LL >=0 && UL_posinf == 1 && LL_neginf == 0))
    {
        caseNO = 2;
    }else if((UL <= 0 && -LL > -UL + 2*sqrt(exp(1))/(-UL+sqrt(pow(UL, 2)+4))*exp((2*UL+UL*sqrt(pow(UL, 2)+4))/4) && LL_neginf == 0 && UL_posinf == 0) || (LL_neginf == 1 && UL <= 0 && UL_posinf == 0))
    {
        caseNO = 3;
    }else
    {
        caseNO = 4;
    }
    
    if(caseNO == 1)
    {
        while(stop == 0)
        {
            y = rnorm(0, 1);
            if(LL_neginf == 1 && UL_posinf == 1)
            {
                stop = 1;
                val = y;
            }else if(LL_neginf == 0 && UL_posinf == 1)
            {
                if(y > LL)
                {
                    stop = 1;
                    val = y;
                }
            }else if(LL_neginf == 1 && UL_posinf == 0)
            {
                if(y < UL)
                {
                    stop = 1;
                    val = y;
                }
            }else if(LL_neginf == 0 && UL_posinf == 0)
            {
                if(y > LL && y < UL)
                {
                    stop = 1;
                    val = y;
                }
            }
        }
    }
    if(caseNO == 2)
    {
        while(stop == 0)
        {
            a = (LL + sqrt(pow(LL, 2)+4))/2;
            z = rexp((double) 1/a) + LL;
            u = runif(0, 1);
            
            if(LL_neginf == 0 && UL_posinf == 0)
            {
                if(u <= exp(-pow(z-a, 2)/2) && z <= UL)
                {
                    stop = 1;
                    val = z;
                }
            }else if(LL_neginf == 0 && UL_posinf == 1)
            {
                if(u <= exp(-pow(z-a, 2)/2))
                {
                    stop = 1;
                    val = z;
                }
            }
        }
    }
    if(caseNO == 3)
    {
        while(stop == 0)
        {
            a = (-UL + sqrt(pow(UL, 2)+4))/2;
            z = rexp((double) 1/a) - UL;
            u = runif(0, 1);
            
            if(LL_neginf == 0 && UL_posinf == 0)
            {
                if(u <= exp(-pow(z-a, 2)/2) && z <= -LL)
                {
                    stop = 1;
                    val = -z;
                }
            }else if(LL_neginf == 1 && UL_posinf == 0)
            {
                if(u <= exp(-pow(z-a, 2)/2))
                {
                    stop = 1;
                    val = -z;
                }
            }
        }
    }
    if(caseNO == 4)
    {
        while(stop == 0)
        {
            if(LL_neginf == 0 && UL_posinf == 0)
            {
                z = runif(LL, UL);
                if(LL >0)
                {
                    rho = exp((pow(LL, 2) - pow(z, 2))/2);
                }else if(UL <0)
                {
                    rho = exp((pow(UL, 2) - pow(z, 2))/2);
                }else
                {
                    rho = exp(- pow(z, 2)/2);
                }
                u = runif(0, 1);
                
                if(u <= rho)
                {
                    stop = 1;
                    val = z;
                }
            }
        }
    }
    *value = mean + val * sd;
    return;
}


int c_multinom_sample(gsl_rng *rr,
                      gsl_vector *prob,
                      int length_prob)
{
    int ii, val;
    int KK = length_prob;
    double probK[KK];
    
    for(ii = 0; ii < KK; ii++)
    {
        probK[ii] = gsl_vector_get(prob, ii);
    }
    
    unsigned int samples[KK];
    
    gsl_ran_multinomial(rr, KK, 1, probK, samples);
    
    for(ii = 0; ii < KK; ii++)
    {
        if(samples[ii] == 1) val = ii + 1;
    }
    
    return val;
}

double logistic(double x)
{
    double value = exp(x)/(1+exp(x));
    return value;
}



/*
 Density calculation for a multivariate normal distribution
 */
void c_dmvnorm3(gsl_vector *x,
                gsl_vector *mu,
                double     sigma,
                gsl_matrix *AInv,
                double     *value)
{
    int K = x->size;
    double sigmaSqInv = pow(sigma, -2);
    
    gsl_vector *diff       = gsl_vector_alloc(K);
    gsl_matrix *SigmaInv   = gsl_matrix_alloc(K, K);
    
    gsl_vector_memcpy(diff, x);
    gsl_vector_sub(diff, mu);
    
    gsl_matrix_memcpy(SigmaInv, AInv);
    gsl_matrix_scale(SigmaInv, sigmaSqInv);
    c_quadform_vMv(diff, SigmaInv, value);
    
    *value = (c_ldet(SigmaInv) - log(pow(2*Pi, K)) - *value) / 2;
    
    gsl_vector_free(diff);
    gsl_matrix_free(SigmaInv);
    return;
    
}

double c_ldet(gsl_matrix *A)
{
    int signum;
    int K = A->size1;
    double value;
    
    gsl_matrix *ALU = gsl_matrix_alloc(K, K);
    gsl_permutation *p     = gsl_permutation_alloc(K);
    gsl_matrix_memcpy(ALU, A);
    gsl_linalg_LU_decomp(ALU, p, &signum);
    
    value = gsl_linalg_LU_lndet(ALU);
    
    gsl_matrix_free(ALU);
    gsl_permutation_free(p);
    
    return value;
}

int c_sgn_det(gsl_matrix *A)
{
    int signum;
    int K = A->size1;
    int value;
    
    gsl_matrix *ALU = gsl_matrix_alloc(K, K);
    gsl_permutation *p     = gsl_permutation_alloc(K);
    gsl_matrix_memcpy(ALU, A);
    gsl_linalg_LU_decomp(ALU, p, &signum);
    
    value = gsl_linalg_LU_sgndet(ALU, signum);
    
    gsl_matrix_free(ALU);
    gsl_permutation_free(p);
    
    return value;
}

double get_det(gsl_matrix * A)
{
    int sign=0;
    double det=0.0;
    int row_sq = A->size1;
    
    gsl_permutation * p = gsl_permutation_calloc(row_sq);
    gsl_matrix * tmp_ptr = gsl_matrix_calloc(row_sq, row_sq);
    
    int * signum = &sign;
    
    gsl_matrix_memcpy(tmp_ptr, A);
    gsl_linalg_LU_decomp(tmp_ptr, p, signum);
    det = gsl_linalg_LU_det(tmp_ptr, *signum);
    
    gsl_permutation_free(p);
    gsl_matrix_free(tmp_ptr);
    return det;
}

void psd_chk(gsl_matrix * A, gsl_vector * check)
{
    const size_t N = A->size2;
    
    size_t j;
    
    for (j = 0; j < N; ++j)
    {
        double ajj;
        
        gsl_vector_view v = gsl_matrix_subcolumn(A, j, j, N - j); /* A(j:n,j) */
        
        if (j > 0)
        {
            gsl_vector_view w = gsl_matrix_subrow(A, j, 0, j);           /* A(j,1:j-1)^T */
            gsl_matrix_view m = gsl_matrix_submatrix(A, j, 0, N - j, j); /* A(j:n,1:j-1) */
            
            gsl_blas_dgemv(CblasNoTrans, -1.0, &m.matrix, &w.vector, 1.0, &v.vector);
        }
        
        ajj = gsl_matrix_get(A, j, j);
        
        gsl_vector_set(check, j, ajj);
    }
    
    return ;
}




/********* For mmzip-BVS model *************/


void Get_subVector(gsl_vector *vec, int i, gsl_vector *subvec)
{
    int q = vec->size;
    int j;
    
    gsl_vector_set_zero(subvec);
    
    for(j = 0; j < q-1; j++)
    {
        if(j < i)
        {
            gsl_vector_set(subvec, j, gsl_vector_get(vec, j));
        }else
        {
            gsl_vector_set(subvec, j, gsl_vector_get(vec, j+1));
        }
    }
    
    return;
}



/*
 Evaluate the inverse of the matrix M
 */
void c_solve_covFA1(gsl_vector *lambda,
                 gsl_matrix *Minv)
{
    int d = Minv->size1;
    int i;
    double den;
    gsl_matrix *temp = gsl_matrix_calloc(d, d);
    
    gsl_blas_ddot(lambda, lambda, &den);
    den += 1;
    
    gsl_blas_dger(-1, lambda, lambda, temp);
    gsl_matrix_scale(temp, pow(den, -1));
    
    for(i = 0; i < d; i++)
    {
        gsl_matrix_set(temp, i, i, gsl_matrix_get(temp, i, i) + 1);
    }
    
    gsl_matrix_memcpy(Minv, temp);
    gsl_matrix_free(temp);
    
    return;
}



void c_solve_corFA1(gsl_vector *lambda,
                 gsl_matrix *Rinv)
{
    int d = Rinv->size1;
    int i, j;
    gsl_matrix *Minv = gsl_matrix_calloc(d, d);
    gsl_matrix *temp = gsl_matrix_calloc(d, d);
    gsl_vector *s = gsl_vector_calloc(d);

    c_solve_covFA1(lambda, Minv);
    
    for(i = 0; i < d; i++)
    {
        gsl_vector_set(s, i, pow(1 + pow(gsl_vector_get(lambda, i), 2), 0.5));
    }
    
    for(i = 0; i < d; i++)
    {
        for(j = 0; j < d; j++)
        {
            gsl_matrix_set(temp, j, i, gsl_vector_get(s, j) * gsl_matrix_get(Minv, j, i));
        }
    }
    
    for(i = 0; i < d; i++)
    {
        for(j = 0; j < d; j++)
        {
            gsl_matrix_set(Rinv, j, i, gsl_matrix_get(temp, j, i) * gsl_vector_get(s, i));
        }
    }
    
    gsl_matrix_free(Minv);
    gsl_matrix_free(temp);
    gsl_vector_free(s);
    
    return;
}





void cov_FA1(gsl_vector *lambda,
              gsl_matrix *M)
{
    int d = M->size1;
    int i;
    gsl_matrix *temp = gsl_matrix_calloc(d, d);

    gsl_blas_dger(1, lambda, lambda, temp);
    
    for(i = 0; i < d; i++)
    {
        gsl_matrix_set(temp, i, i, gsl_matrix_get(temp, i, i) + 1);
    }
    
    gsl_matrix_memcpy(M, temp);
    gsl_matrix_free(temp);
    
    return;
}



void c_cov2cor(gsl_matrix *Sigma,
               gsl_matrix *R)
{
    int d = Sigma->size1;
    int i, j;
    gsl_matrix *temp = gsl_matrix_calloc(d, d);
    gsl_vector *Is = gsl_vector_calloc(d);
    
    for(i = 0; i < d; i++)
    {
        gsl_vector_set(Is, i, pow(gsl_matrix_get(Sigma, i, i), -0.5));
    }
    
    for(i = 0; i < d; i++)
    {
        for(j = 0; j < d; j++)
        {
            gsl_matrix_set(temp, j, i, gsl_vector_get(Is, j) * gsl_matrix_get(Sigma, j, i));
        }
    }
    
    for(i = 0; i < d; i++)
    {
        for(j = 0; j < d; j++)
        {
            gsl_matrix_set(R, j, i, gsl_matrix_get(temp, j, i) * gsl_vector_get(Is, i));
        }
    }
    
    
    for(i = 0; i < d; i++)
    {
        gsl_matrix_set(R, i, i, 1);
    }
    
    gsl_matrix_free(temp);
    gsl_vector_free(Is);
    
    return;
}




/*
 Density calculation for a multivariate normal distribution
 */
void c_dmvnorm2_FA(gsl_vector *x,
                   gsl_vector *mu,
                   double     sigma,
                   gsl_matrix *AInv,
                   gsl_vector *m,
                   double     *value)
{
    int K = x->size;
    double sigmaSqInv = pow(sigma, -2);
    int j;
    double detR;
    
    gsl_vector *diff       = gsl_vector_alloc(K);
    gsl_matrix *SigmaInv   = gsl_matrix_alloc(K, K);
    
    gsl_vector_memcpy(diff, x);
    gsl_vector_sub(diff, mu);
    gsl_matrix_memcpy(SigmaInv, AInv);
    gsl_matrix_scale(SigmaInv, sigmaSqInv);
    
    gsl_blas_ddot(m, m, &detR);
    detR += 1;
    for(j = 0; j < K; j++)
    {
        detR /= pow(gsl_vector_get(m, j), 2) + 1;
    }
    
    c_quadform_vMv(diff, SigmaInv, value);
    
    *value = (-log(detR) - log(pow(2*Pi, K)) - *value) / 2;
    
    gsl_vector_free(diff);
    gsl_matrix_free(SigmaInv);
    return;
}



/*
 Density calculation for a multivariate normal distribution
 */
void c_dmvnorm2(gsl_vector *x,
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




void c_ldmvn_noDet(gsl_vector *x,
                gsl_vector *mu,
                double     sigma,
                gsl_matrix *AInv,
                double     *value)
{
    int K = x->size;
    double sigmaSqInv = pow(sigma, -2);
    
    gsl_vector *diff       = gsl_vector_alloc(K);
    gsl_matrix *SigmaInv   = gsl_matrix_alloc(K, K);
    
    gsl_vector_memcpy(diff, x);
    gsl_vector_sub(diff, mu);
    
    gsl_matrix_memcpy(SigmaInv, AInv);
    gsl_matrix_scale(SigmaInv, sigmaSqInv);
    c_quadform_vMv(diff, SigmaInv, value);

    *value *= -0.5;
    
    gsl_vector_free(diff);
    gsl_matrix_free(SigmaInv);
    return;
    
}



void new_mean_vec(gsl_vector *mean,
                  int n_old,
                  gsl_vector *newObs)
{
    gsl_vector_scale(mean, (double) n_old);
    gsl_vector_add(mean, newObs);
    gsl_vector_scale(mean, pow((double) n_old+1, -1));
    
    return;
}

void new_var_vec(gsl_vector *var,
                 gsl_vector *oldmean,
                  int n_old,
                  gsl_vector *newObs)
{
    int K = var->size;
    gsl_vector *temp       = gsl_vector_alloc(K);
    
    if(n_old == 1)
    {
        gsl_vector_set_zero(var);
    }else
    {
        gsl_vector_scale(var, (double) (n_old-1)/(n_old));
    }
    
    gsl_vector_memcpy(temp, oldmean);
    gsl_vector_sub(temp, newObs);
    gsl_vector_mul(temp, temp);    
    gsl_vector_scale(temp, (double) 1/(n_old+1));
    gsl_vector_add(var, temp);
    
    gsl_vector_free(temp);
    
    return;
}


void new_mean_mat(gsl_matrix *mean,
                  int n_old,
                  gsl_matrix *newObs)
{
    gsl_matrix_scale(mean, (double) n_old);
    gsl_matrix_add(mean, newObs);
    gsl_matrix_scale(mean, pow((double) n_old+1, -1));
    
    return;
}

void new_mean_mat2(gsl_matrix *mean,
                  gsl_matrix *s_Param,
                  gsl_matrix *newObs)
{
    int k, j;
    int p = mean->size1;
    int q = mean->size2;
    
    gsl_matrix_mul_elements(mean, s_Param);
    gsl_matrix_add(mean, newObs);
    
    for(k = 0; k < p; k++)
    {
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(mean, k, j, gsl_matrix_get(mean, k, j) * pow(gsl_matrix_get(s_Param, k, j)+1, -1));
        }
    }
    
    return;
}

void new_var_mat(gsl_matrix *var,
                 gsl_matrix *oldmean,
                  int n_old,
                  gsl_matrix *newObs)
{
    int n = var->size1;
    int p = var->size2;
    
    gsl_matrix *temp       = gsl_matrix_alloc(n, p);

    if(n_old == 1)
    {
        gsl_matrix_set_zero(var);
    }else
    {
        gsl_matrix_scale(var, (double) (n_old-1)/(n_old));
    }
        
    gsl_matrix_memcpy(temp, oldmean);
    gsl_matrix_sub(temp, newObs);
    gsl_matrix_mul_elements(temp, temp);
    gsl_matrix_scale(temp, (double) 1/(n_old+1));
    gsl_matrix_add(var, temp);
        
    gsl_matrix_free(temp);
    

    
    return;
}



void new_var_mat2(gsl_matrix *var,
                 gsl_matrix *oldmean,
                  gsl_matrix *s_Param,
                  gsl_matrix *newObs)
{
    int k, j;
    int p = var->size1;
    int q = var->size2;
    
    gsl_matrix *temp       = gsl_matrix_alloc(p, q);
    
    
    for(k = 0; k < p; k++)
    {
        for(j = 0; j < q; j++)
        {
            if(gsl_matrix_get(s_Param, k, j) == 1)
            {
                gsl_matrix_set(var, k, j, 0);
            }else
            {
                gsl_matrix_set(var, k, j, gsl_matrix_get(var, k, j)* (gsl_matrix_get(s_Param, k, j)-1)/gsl_matrix_get(s_Param, k, j) );
            }
        }
    }
        
    gsl_matrix_memcpy(temp, oldmean);
    gsl_matrix_sub(temp, newObs);
    gsl_matrix_mul_elements(temp, temp);
    
    for(k = 0; k < p; k++)
    {
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(temp, k, j, gsl_matrix_get(temp, k, j)/(gsl_matrix_get(s_Param, k, j)+1) );
        }
    }

    gsl_matrix_add(var, temp);
        
    gsl_matrix_free(temp);
    
    return;
}





