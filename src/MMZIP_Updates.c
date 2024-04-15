
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_heapsort.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"
#include "MMZIP.h"


/* Gradient of A (given gamma_kj = 1) */

void Grad_A_mmzip(gsl_matrix *YI_LamSI,
                  gsl_matrix *X0,
                  gsl_matrix *FI,
                  gsl_matrix *W_XAinvR,
                  gsl_vector *invRalpha0,
                  gsl_matrix *A,
                  gsl_matrix *gamma_alpha,
                  gsl_matrix *APriorV,
                  gsl_matrix *Delta)
{
    int n = YI_LamSI -> size1;
    int p0_all = A -> size1;
    int q_bin = A -> size2;
    
    gsl_matrix *Temp = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *Temp2 = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *GradPr = gsl_matrix_calloc(p0_all, q_bin);
    gsl_vector *ones_n = gsl_vector_calloc(n);
    
    gsl_vector_set_all(ones_n, 1);
    
    gsl_matrix_view YI_LamSI_sub = gsl_matrix_submatrix(YI_LamSI, 0, 0, n, q_bin);
    gsl_matrix_memcpy(Temp, FI);
    gsl_matrix_mul_elements(Temp, &YI_LamSI_sub.matrix);
    gsl_matrix_scale(Temp, -1);
    
    gsl_matrix_memcpy(Temp2, W_XAinvR);
    gsl_blas_dger(-1, ones_n, invRalpha0, Temp2);
    
    gsl_matrix_add(Temp, Temp2);
    
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, X0, Temp, 0, Delta);
    
    gsl_matrix_memcpy(GradPr, APriorV);
    gsl_matrix_mul_elements(GradPr, A);
    gsl_matrix_sub(Delta, GradPr);
    
    gsl_matrix_mul_elements(Delta, gamma_alpha);
    
    gsl_matrix_free(Temp);
    gsl_matrix_free(Temp2);
    gsl_matrix_free(GradPr);
    gsl_vector_free(ones_n);
    
    return;
}










/* Gradient of alpha0 */

void Grad_alpha0_mmzip(gsl_matrix *YI_LamSI,
                       gsl_matrix *FI,
                       gsl_vector *alpha0,
                       gsl_matrix *W_XAinvR,
                       gsl_vector *invRalpha0,
                       gsl_vector *mu_alpha0,
                       double sigSq_alpha0,
                       gsl_vector *Delta)
{
    int n = W_XAinvR -> size1;
    int q_bin = W_XAinvR -> size2;
    
    gsl_matrix *Temp = gsl_matrix_calloc(n, q_bin);
    gsl_vector *ones_n = gsl_vector_calloc(n);
    gsl_vector *GradPr = gsl_vector_calloc(q_bin);
    gsl_vector *n_invRalpha0 = gsl_vector_calloc(q_bin);
    
    gsl_vector_set_all(ones_n, 1);
    
    gsl_matrix_view YI_LamSI_sub = gsl_matrix_submatrix(YI_LamSI, 0, 0, n, q_bin);
    
    gsl_matrix_memcpy(Temp, FI);
    gsl_matrix_mul_elements(Temp, &YI_LamSI_sub.matrix);
    gsl_blas_dgemv(CblasTrans, -1, Temp, ones_n, 0, Delta);
    
    gsl_blas_dgemv(CblasTrans, 1, W_XAinvR, ones_n, 1, Delta);
    
    gsl_vector_memcpy(n_invRalpha0, invRalpha0);
    gsl_vector_scale(n_invRalpha0, (double) n);
    gsl_vector_sub(Delta, n_invRalpha0);
    
    gsl_vector_memcpy(GradPr, alpha0);
    gsl_vector_sub(GradPr, mu_alpha0);
    gsl_vector_scale(GradPr, pow(sigSq_alpha0, -1));
    
    gsl_vector_sub(Delta, GradPr);
    
    gsl_matrix_free(Temp);
    gsl_vector_free(ones_n);
    gsl_vector_free(GradPr);
    gsl_vector_free(n_invRalpha0);
    
    return;
}










void LH_all_mmzip(gsl_matrix *Y,
                  gsl_matrix *logLamSI,
                  gsl_matrix *LamSI,
                  gsl_matrix *XA,
                  gsl_vector *alpha0,
                  gsl_matrix *W,
                  gsl_matrix *invR,
                  double *logLH_val)
{
    int i, jj, jj_temp;
    double quad_val;
    
    int n = Y -> size1;
    int q = Y -> size2;
    int q_bin = W -> size2;
    
    gsl_matrix *W_XA = gsl_matrix_calloc(n, q_bin);
    gsl_vector *W_XA_alpha0 = gsl_vector_calloc(q_bin);
    
    *logLH_val = 0;
    
    for(jj = 0; jj < q; jj++)
    {
        jj_temp = (jj < q_bin) ? jj : (q_bin-1);
        for(i = 0; i < n; i++)
        {
            if((jj < q_bin && gsl_matrix_get(W, i, jj_temp) >= 0) || jj >= q_bin)
            {
                *logLH_val += gsl_matrix_get(Y, i, jj)*gsl_matrix_get(logLamSI, i, jj) - gsl_matrix_get(LamSI, i, jj);
            }
        }
    }
    
    gsl_matrix_memcpy(W_XA, W);
    gsl_matrix_sub(W_XA, XA);
    
    for(i=0; i<n; i++)
    {
        gsl_vector_view W_XA_i = gsl_matrix_row(W_XA, i);
        gsl_vector_memcpy(W_XA_alpha0, &W_XA_i.vector);
        gsl_vector_sub(W_XA_alpha0, alpha0);
        
        c_quadform_vMv(W_XA_alpha0, invR, &quad_val);
        *logLH_val += -0.5*quad_val;
    }
    
    gsl_matrix_free(W_XA);
    gsl_vector_free(W_XA_alpha0);
    
    return;
}



/* Calculate F x I_w */

void Cal_FI_mmzip(gsl_matrix *XA,
                  gsl_vector *alpha0,
                  gsl_matrix *W,
                  gsl_matrix *FI)
{
    int ii, j;
    double tempA;
    double PDFval, CDFval;
    
    int n = XA -> size1;
    int q_bin = XA -> size2;
    
    gsl_matrix_set_zero(FI);
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, ii, j) >= 0)
            {
                tempA = gsl_vector_get(alpha0, j) + gsl_matrix_get(XA, ii, j);
                
                PDFval = dnorm(tempA, 0, 1, 0);
                CDFval = pnorm(tempA, 0, 1, 1, 0);
                
                gsl_matrix_set(FI, ii, j, PDFval/CDFval);
            }
        }
    }
    
    return;
}


/* Calculate Lambda_star x I_w (Q x I_w) and its log value */

void Cal_LamSI_mmzip(gsl_matrix *XB,
                     gsl_matrix *XA,
                     gsl_vector *xi,
                     gsl_vector *beta0,
                     gsl_matrix *V,
                     gsl_vector *alpha0,
                     gsl_matrix *W,
                     gsl_matrix *LamSI,
                     gsl_matrix *logLamSI)
{
    int i, jj;
    double tempA, tempB;
    
    int n = XB -> size1;
    int q = XB -> size2;
    int q_bin = XA -> size2;
    
    gsl_matrix_set_zero(LamSI);
    gsl_matrix_set_zero(logLamSI);
    
    for(jj = 0; jj < q_bin; jj++)
    {
        for(i=0; i<n; i++)
        {
            if(gsl_matrix_get(W, i, jj) >= 0)
            {
                tempA = gsl_matrix_get(XA, i, jj) + gsl_vector_get(alpha0, jj);
                tempB = gsl_matrix_get(XB, i, jj)+gsl_vector_get(beta0, jj)+gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA, 0, 1, 1, 1);
                
                gsl_matrix_set(LamSI, i, jj, exp(tempB));
                gsl_matrix_set(logLamSI, i, jj, tempB);
            }
        }
    }
    
    if(q > q_bin)
    {
        for(jj = q_bin; jj < q; jj++)
        {
            for(i=0; i<n; i++)
            {
                tempB = gsl_matrix_get(XB, i, jj)+gsl_vector_get(beta0, jj)+gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                
                gsl_matrix_set(LamSI, i, jj, exp(tempB));
                gsl_matrix_set(logLamSI, i, jj, tempB);
            }
        }
    }
    
    return;
}




/* updating B, beta0, V, A, and alpha0 */

void update_group_mmzip(int *p_adj,
                        gsl_matrix *Y,
                        gsl_matrix *X1,
                        gsl_matrix *X0,
                        gsl_vector *sum_X0sq,
                        gsl_vector *xi,
                        gsl_vector *beta0,
                        gsl_matrix *B,
                        gsl_matrix *V,
                        gsl_matrix *gamma_beta,
                        gsl_vector *alpha0,
                        gsl_matrix *A,
                        gsl_matrix *W,
                        gsl_matrix *gamma_alpha,
                        gsl_matrix *updateNonzB,
                        gsl_vector *sigSq_beta,
                        gsl_vector *v_beta,
                        gsl_vector *mu_beta0,
                        double sigSq_beta0,
                        gsl_matrix *updateNonzA,
                        gsl_vector *sigSq_alpha,
                        gsl_vector *v_alpha,
                        gsl_vector *mu_alpha0,
                        double sigSq_alpha0,
                        gsl_matrix *invSigmaV,
                        gsl_matrix *cholSigmaV,
                        gsl_matrix *invR,
                        int *accept_group,
                        int *accept_V,
                        gsl_vector *accept_group100,
                        gsl_vector *accept_V100,
                        double *eps_group,
                        double *eps_V,
                        int L_group,
                        int L_V,
                        int *n_group,
                        int *numReps,
                        double *burninPerc,
                        int M,
                        gsl_matrix *M_B,
                        gsl_vector *M_beta0,
                        gsl_matrix *M_V,
                        gsl_matrix *M_A,
                        gsl_vector *M_alpha0,
                        double PtuneEps)
{
    int i, ii, j, k, l, u;
    double sumAccept, tempPrior, tempPrior_prop;
    double U_star, U_prop, K_star, K_prop, logR;
    int accept_ind;
    
    int n = Y -> size1;
    int p1_all = B -> size1;
    int q = B -> size2;
    int p0_all = A -> size1;
    int q_bin = A -> size2;
    
    gsl_matrix *XB = gsl_matrix_calloc(n, q);
    gsl_matrix *XB_prop = gsl_matrix_calloc(n, q);
    gsl_matrix *XA = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *XA_prop = gsl_matrix_calloc(n, q_bin);
    
    gsl_matrix *LamSI = gsl_matrix_calloc(n, q);
    gsl_matrix *LamSI_prop = gsl_matrix_calloc(n, q);
    gsl_matrix *logLamSI = gsl_matrix_calloc(n, q);
    gsl_matrix *logLamSI_prop = gsl_matrix_calloc(n, q);
    
    gsl_matrix *YI = gsl_matrix_calloc(n, q);
    gsl_matrix *YI_LamSI = gsl_matrix_calloc(n, q);
    gsl_matrix *YI_LamSI_prop = gsl_matrix_calloc(n, q);
    
    gsl_matrix *FI = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *FI_prop = gsl_matrix_calloc(n, q_bin);
    
    gsl_vector *zero_q = gsl_vector_calloc(q);
    
    gsl_matrix *B_ini = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *B_star = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *B_prop = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *p_B_ini = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *p_B_star = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *p_B_prop = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *Delta_B_star = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *Delta_B_prop = gsl_matrix_calloc(p1_all, q);
    
    gsl_matrix *V_ini = gsl_matrix_calloc(n, q);
    gsl_matrix *V_star = gsl_matrix_calloc(n, q);
    gsl_matrix *V_prop = gsl_matrix_calloc(n, q);
    gsl_matrix *p_V_ini = gsl_matrix_calloc(n, q);
    gsl_matrix *p_V_star = gsl_matrix_calloc(n, q);
    gsl_matrix *p_V_prop = gsl_matrix_calloc(n, q);
    gsl_matrix *Delta_V_star = gsl_matrix_calloc(n, q);
    gsl_matrix *Delta_V_prop = gsl_matrix_calloc(n, q);
    
    gsl_vector *beta0_ini = gsl_vector_calloc(q);
    gsl_vector *beta0_star = gsl_vector_calloc(q);
    gsl_vector *beta0_prop = gsl_vector_calloc(q);
    gsl_vector *p_beta0_ini = gsl_vector_calloc(q);
    gsl_vector *p_beta0_star = gsl_vector_calloc(q);
    gsl_vector *p_beta0_prop = gsl_vector_calloc(q);
    gsl_vector *Delta_beta0_star = gsl_vector_calloc(q);
    gsl_vector *Delta_beta0_prop = gsl_vector_calloc(q);
    
    gsl_matrix *A_ini = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *A_star = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *A_prop = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *p_A_ini = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *p_A_star = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *p_A_prop = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *Delta_A_star = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *Delta_A_prop = gsl_matrix_calloc(p0_all, q_bin);
    
    gsl_vector *alpha0_ini = gsl_vector_calloc(q_bin);
    gsl_vector *alpha0_star = gsl_vector_calloc(q_bin);
    gsl_vector *alpha0_prop = gsl_vector_calloc(q_bin);
    gsl_vector *p_alpha0_ini = gsl_vector_calloc(q_bin);
    gsl_vector *p_alpha0_star = gsl_vector_calloc(q_bin);
    gsl_vector *p_alpha0_prop = gsl_vector_calloc(q_bin);
    gsl_vector *Delta_alpha0_star = gsl_vector_calloc(q_bin);
    gsl_vector *Delta_alpha0_prop = gsl_vector_calloc(q_bin);
    
    gsl_matrix *BPriorV = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *APriorV = gsl_matrix_calloc(p0_all, q_bin);
    
    gsl_matrix *W_XAinvR = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *W_XAinvR_prop = gsl_matrix_calloc(n, q_bin);
    gsl_vector *invRalpha0 = gsl_vector_calloc(q_bin);
    gsl_vector *invRalpha0_prop = gsl_vector_calloc(q_bin);
    
    accept_ind = 0;
    *n_group += 1;
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X1, B, 0, XB);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X0, A, 0, XA);
    
    gsl_matrix_memcpy(B_ini, B);
    gsl_matrix_memcpy(B_star, B);
    
    gsl_matrix_memcpy(V_ini, V);
    gsl_matrix_memcpy(V_star, V);
    
    gsl_vector_memcpy(beta0_ini, beta0);
    gsl_vector_memcpy(beta0_star, beta0);
    
    gsl_matrix_memcpy(A_ini, A);
    gsl_matrix_memcpy(A_star, A);
    
    gsl_vector_memcpy(alpha0_ini, alpha0);
    gsl_vector_memcpy(alpha0_star, alpha0);
    
    Cal_LamSI_mmzip(XB, XA, xi, beta0, V, alpha0, W, LamSI, logLamSI);
    Cal_FI_mmzip(XA, alpha0, W, FI);
    
    for(j = 0; j < q; j++)
    {
        for(k = 0; k < p1_all; k++)
        {
            gsl_matrix_set(BPriorV, k, j, pow(gsl_vector_get(sigSq_beta, k)*pow(gsl_vector_get(v_beta, j),2), -1));
        }
    }
    
    for(j = 0; j < q_bin; j++)
    {
        for(k = 0; k < p0_all; k++)
        {
            gsl_matrix_set(APriorV, k, j, pow(gsl_vector_get(sigSq_alpha, k)*pow(gsl_vector_get(v_alpha, j),2), -1));
        }
    }
    
    Hessian_beta0_mmzip(LamSI, sigSq_beta0, M_beta0);
    Hessian_V_mmzip(LamSI, invSigmaV, M_V);
    Hessian_B_mmzip(LamSI, X1, BPriorV, M_B);
    Hessian_A_alpha0_mmzip(XA, X0, alpha0, sum_X0sq, A, W, Y, LamSI, invR, APriorV, sigSq_alpha0, M_alpha0, M_A);
    
    for(j = 0; j < q; j++)
    {
        for(k = 0; k < p1_all; k++)
        {
            if(gsl_matrix_get(gamma_beta, k, j) == 1)
            {
                gsl_matrix_set(p_B_star, k, j, rnorm(0, sqrt(gsl_matrix_get(M_B, k, j))));
            }
        }
    }
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(p_V_star, ii, j, rnorm(0, sqrt(gsl_matrix_get(M_V, ii, j))));
        }
    }
    
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(p_beta0_star, j, rnorm(0, sqrt(gsl_vector_get(M_beta0, j))));
    }
    
    for(j = 0; j < q_bin; j++)
    {
        for(k = 0; k < p0_all; k++)
        {
            if(gsl_matrix_get(gamma_alpha, k, j) == 1)
            {
                gsl_matrix_set(p_A_star, k, j, rnorm(0, sqrt(gsl_matrix_get(M_A, k, j))));
            }
        }
    }
    
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(p_alpha0_star, j, rnorm(0, sqrt(gsl_vector_get(M_alpha0, j))));
    }

    U_star=0;
    LH_all_mmzip(Y, logLamSI, LamSI, XA, alpha0, W, invR, &U_star);
    U_star *= -1;
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, ii, j) >= 0)
            {
                gsl_matrix_set(YI, ii, j, gsl_matrix_get(Y, ii, j));
            }
        }
        
        if(q > q_bin)
        {
            for(j = q_bin; j < q; j++)
            {
                gsl_matrix_set(YI, ii, j, gsl_matrix_get(Y, ii, j));
            }
        }
        
        gsl_vector_view V_star_ii = gsl_matrix_row(V_star, ii);
        
        c_ldmvn_noDet(&V_star_ii.vector, zero_q, 1, invSigmaV, &tempPrior);
        U_star -= tempPrior;
    }
    
    for(j = 0; j < q; j++)
    {
        U_star -= -pow(gsl_vector_get(beta0, j)-gsl_vector_get(mu_beta0, j), 2)/(2*sigSq_beta0);
        
        for(k = 0; k < p1_all; k++)
        {
            if(gsl_matrix_get(gamma_beta, k, j) == 1)
            {
                U_star -= -pow(gsl_matrix_get(B, k, j), 2)/(2*gsl_vector_get(sigSq_beta, k)*pow(gsl_vector_get(v_beta, j),2));
                
                gsl_matrix_set(BPriorV, k, j, pow(gsl_vector_get(sigSq_beta, k)*pow(gsl_vector_get(v_beta, j),2), -1));
            }
        }
    }
    
    for(j = 0; j < q_bin; j++)
    {
        U_star -= -pow(gsl_vector_get(alpha0, j)-gsl_vector_get(mu_alpha0, j), 2)/(2*sigSq_alpha0);
        
        for(k = 0; k < p0_all; k++)
        {
            if(gsl_matrix_get(gamma_alpha, k, j) == 1)
            {
                U_star -= -pow(gsl_matrix_get(A, k, j), 2)/(2*gsl_vector_get(sigSq_alpha, k)*pow(gsl_vector_get(v_alpha, j),2));
                
                gsl_matrix_set(APriorV, k, j, pow(gsl_vector_get(sigSq_alpha, k)*pow(gsl_vector_get(v_alpha, j),2), -1));
            }
        }
    }
    
    gsl_matrix_memcpy(YI_LamSI, YI);
    gsl_matrix_sub(YI_LamSI, LamSI);
    
    gsl_matrix_memcpy(W_XAinvR, W);
    gsl_matrix_sub(W_XAinvR, XA);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, W_XAinvR, invR, 0, W_XAinvR);
    
    gsl_blas_dgemv(CblasNoTrans, 1, invR, alpha0, 0, invRalpha0);
    
    Grad_B_mmzip(YI_LamSI, X1, B, gamma_beta, BPriorV, Delta_B_star);
    
    Grad_V_mmzip(YI_LamSI, V, W, invSigmaV, Delta_V_star);
    
    Grad_beta0_mmzip(YI_LamSI, beta0, mu_beta0, sigSq_beta0, Delta_beta0_star);
    
    Grad_A_mmzip(YI_LamSI, X0, FI, W_XAinvR, invRalpha0, A, gamma_alpha, APriorV, Delta_A_star);
    
    Grad_alpha0_mmzip(YI_LamSI, FI, alpha0, W_XAinvR, invRalpha0, mu_alpha0, sigSq_alpha0, Delta_alpha0_star);
    
    gsl_matrix_memcpy(p_B_ini, Delta_B_star);
    gsl_matrix_scale(p_B_ini, 0.5* *eps_group);
    gsl_matrix_add(p_B_ini, p_B_star);
    
    gsl_matrix_memcpy(p_V_ini, Delta_V_star);
    gsl_matrix_scale(p_V_ini, 0.5* *eps_group);
    gsl_matrix_add(p_V_ini, p_V_star);
    
    gsl_vector_memcpy(p_beta0_ini, Delta_beta0_star);
    gsl_vector_scale(p_beta0_ini, 0.5* *eps_group);
    gsl_vector_add(p_beta0_ini, p_beta0_star);
    
    gsl_matrix_memcpy(p_A_ini, Delta_A_star);
    gsl_matrix_scale(p_A_ini, 0.5* *eps_group);
    gsl_matrix_add(p_A_ini, p_A_star);
    
    gsl_vector_memcpy(p_alpha0_ini, Delta_alpha0_star);
    gsl_vector_scale(p_alpha0_ini, 0.5* *eps_group);
    gsl_vector_add(p_alpha0_ini, p_alpha0_star);
    
    U_prop = 0;
    for(l = 1; l <= L_group; l++)
    {
        gsl_matrix_memcpy(B_prop, p_B_ini);
        gsl_matrix_scale(B_prop, *eps_group);
        gsl_matrix_div_elements(B_prop, M_B);
        gsl_matrix_add(B_prop, B_ini);
        gsl_matrix_mul_elements(B_prop, gamma_beta);
        
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X1, B_prop, 0, XB_prop);
        
        gsl_matrix_memcpy(V_prop, p_V_ini);
        gsl_matrix_scale(V_prop, *eps_group);
        gsl_matrix_div_elements(V_prop, M_V);
        gsl_matrix_add(V_prop, V_ini);
        
        for(ii = 0; ii < n; ii++)
        {
            for(j = 0; j < q_bin; j++)
            {
                if(gsl_matrix_get(W, ii, j) < 0)
                {
                    gsl_matrix_set(V_prop, ii, j, gsl_matrix_get(V_star, ii, j));
                }
            }
        }
        
        gsl_vector_memcpy(beta0_prop, p_beta0_ini);
        gsl_vector_scale(beta0_prop, *eps_group);
        gsl_vector_div(beta0_prop, M_beta0);
        gsl_vector_add(beta0_prop, beta0_ini);
        
        gsl_matrix_memcpy(A_prop, p_A_ini);
        gsl_matrix_scale(A_prop, *eps_group);
        gsl_matrix_div_elements(A_prop, M_A);
        gsl_matrix_add(A_prop, A_ini);
        gsl_matrix_mul_elements(A_prop, gamma_alpha);
        
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X0, A_prop, 0, XA_prop);
        
        gsl_vector_memcpy(alpha0_prop, p_alpha0_ini);
        gsl_vector_scale(alpha0_prop, *eps_group);
        gsl_vector_div(alpha0_prop, M_alpha0);
        gsl_vector_add(alpha0_prop, alpha0_ini);
        
        gsl_matrix_set_zero(Delta_B_prop);
        gsl_matrix_set_zero(Delta_V_prop);
        gsl_vector_set_zero(Delta_beta0_prop);
        gsl_matrix_set_zero(Delta_A_prop);
        gsl_vector_set_zero(Delta_alpha0_prop);
        
        Cal_LamSI_mmzip(XB_prop, XA_prop, xi, beta0_prop, V_prop, alpha0_prop, W, LamSI_prop, logLamSI_prop);
        
        Cal_FI_mmzip(XA_prop, alpha0_prop, W, FI_prop);
        
        gsl_matrix_memcpy(YI_LamSI_prop, YI);
        gsl_matrix_sub(YI_LamSI_prop, LamSI_prop);
        
        gsl_matrix_memcpy(W_XAinvR_prop, W);
        gsl_matrix_sub(W_XAinvR_prop, XA_prop);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, W_XAinvR_prop, invR, 0, W_XAinvR_prop);
        
        gsl_blas_dgemv(CblasNoTrans, 1, invR, alpha0_prop, 0, invRalpha0_prop);
        
        Grad_B_mmzip(YI_LamSI_prop, X1, B_prop, gamma_beta, BPriorV, Delta_B_prop);
        
        Grad_V_mmzip(YI_LamSI_prop, V_prop, W, invSigmaV, Delta_V_prop);
        
        Grad_beta0_mmzip(YI_LamSI_prop, beta0_prop, mu_beta0, sigSq_beta0, Delta_beta0_prop);
        
        Grad_A_mmzip(YI_LamSI_prop, X0, FI_prop, W_XAinvR_prop, invRalpha0_prop, A_prop, gamma_alpha, APriorV, Delta_A_prop);
        
        Grad_alpha0_mmzip(YI_LamSI_prop, FI_prop, alpha0_prop, W_XAinvR_prop, invRalpha0_prop, mu_alpha0, sigSq_alpha0, Delta_alpha0_prop);
        
        if(l == L_group)
        {
            LH_all_mmzip(Y, logLamSI_prop, LamSI_prop, XA_prop, alpha0_prop, W, invR, &U_prop);
            U_prop *= -1;
            
            for(ii = 0; ii < n; ii++)
            {
                gsl_vector_view V_prop_ii = gsl_matrix_row(V_prop, ii);
                
                c_ldmvn_noDet(&V_prop_ii.vector, zero_q, 1, invSigmaV, &tempPrior_prop);
                U_prop -= tempPrior_prop;
            }
            
            for(j = 0; j < q; j++)
            {
                U_prop -= -pow(gsl_vector_get(beta0_prop, j)-gsl_vector_get(mu_beta0, j), 2)/(2*sigSq_beta0);
                
                for(k = 0; k < p1_all; k++)
                {
                    if(gsl_matrix_get(gamma_beta, k, j) == 1)
                    {
                        U_prop -= -pow(gsl_matrix_get(B_prop, k, j), 2)/(2*gsl_vector_get(sigSq_beta, k)*pow(gsl_vector_get(v_beta, j),2));
                    }
                }
            }
            
            for(j = 0; j < q_bin; j++)
            {
                U_prop -= -pow(gsl_vector_get(alpha0_prop, j)-gsl_vector_get(mu_alpha0, j), 2)/(2*sigSq_alpha0);
                
                for(k = 0; k < p0_all; k++)
                {
                    if(gsl_matrix_get(gamma_alpha, k, j) == 1)
                    {
                        U_prop -= -pow(gsl_matrix_get(A_prop, k, j), 2)/(2*gsl_vector_get(sigSq_alpha, k)*pow(gsl_vector_get(v_alpha, j),2));
                    }
                }
            }
        }
        
        if(l < L_group)
        {
            gsl_matrix_memcpy(p_B_prop, Delta_B_prop);
            gsl_matrix_scale(p_B_prop, *eps_group);
            gsl_matrix_add(p_B_prop, p_B_ini);
            
            gsl_matrix_memcpy(B_ini, B_prop);
            gsl_matrix_memcpy(p_B_ini, p_B_prop);
            
            gsl_matrix_memcpy(p_V_prop, Delta_V_prop);
            gsl_matrix_scale(p_V_prop, *eps_group);
            gsl_matrix_add(p_V_prop, p_V_ini);
            
            gsl_matrix_memcpy(V_ini, V_prop);
            gsl_matrix_memcpy(p_V_ini, p_V_prop);
            
            gsl_vector_memcpy(p_beta0_prop, Delta_beta0_prop);
            gsl_vector_scale(p_beta0_prop, *eps_group);
            gsl_vector_add(p_beta0_prop, p_beta0_ini);
            
            gsl_vector_memcpy(beta0_ini, beta0_prop);
            gsl_vector_memcpy(p_beta0_ini, p_beta0_prop);
            
            gsl_matrix_memcpy(p_A_prop, Delta_A_prop);
            gsl_matrix_scale(p_A_prop, *eps_group);
            gsl_matrix_add(p_A_prop, p_A_ini);
            
            gsl_matrix_memcpy(A_ini, A_prop);
            gsl_matrix_memcpy(p_A_ini, p_A_prop);
            
            gsl_vector_memcpy(p_alpha0_prop, Delta_alpha0_prop);
            gsl_vector_scale(p_alpha0_prop, *eps_group);
            gsl_vector_add(p_alpha0_prop, p_alpha0_ini);
            
            gsl_vector_memcpy(alpha0_ini, alpha0_prop);
            gsl_vector_memcpy(p_alpha0_ini, p_alpha0_prop);
        }else if(l == L_group)
        {
            gsl_matrix_memcpy(p_B_prop, Delta_B_prop);
            gsl_matrix_scale(p_B_prop, 0.5**eps_group);
            gsl_matrix_add(p_B_prop, p_B_ini);
            
            gsl_matrix_memcpy(p_V_prop, Delta_V_prop);
            gsl_matrix_scale(p_V_prop, 0.5**eps_group);
            gsl_matrix_add(p_V_prop, p_V_ini);
            
            gsl_vector_memcpy(p_beta0_prop, Delta_beta0_prop);
            gsl_vector_scale(p_beta0_prop, 0.5**eps_group);
            gsl_vector_add(p_beta0_prop, p_beta0_ini);
            
            gsl_matrix_memcpy(p_A_prop, Delta_A_prop);
            gsl_matrix_scale(p_A_prop, 0.5**eps_group);
            gsl_matrix_add(p_A_prop, p_A_ini);
            
            gsl_vector_memcpy(p_alpha0_prop, Delta_alpha0_prop);
            gsl_vector_scale(p_alpha0_prop, 0.5**eps_group);
            gsl_vector_add(p_alpha0_prop, p_alpha0_ini);
        }
    }
    
    K_star = 0;
    K_prop = 0;
    for(j = 0; j < q; j++)
    {
        for(k = 0; k < p1_all; k++)
        {
            if(gsl_matrix_get(gamma_beta, k, j) == 1)
            {
                K_star += pow(gsl_matrix_get(p_B_star, k, j), 2)* pow(gsl_matrix_get(M_B, k, j), -1) * 0.5;
                K_prop += pow(gsl_matrix_get(p_B_prop, k, j), 2)* pow(gsl_matrix_get(M_B, k, j), -1) * 0.5;
            }
        }
        
        K_star += pow(gsl_vector_get(p_beta0_star, j), 2) * pow(gsl_vector_get(M_beta0, j), -1) * 0.5;
        K_prop += pow(gsl_vector_get(p_beta0_prop, j), 2) * pow(gsl_vector_get(M_beta0, j), -1) * 0.5;
    }
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, ii, j) >=0)
            {
                K_star += pow(gsl_matrix_get(p_V_star, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
                K_prop += pow(gsl_matrix_get(p_V_prop, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
            }
        }
        if(q > q_bin)
        {
            for(j = q_bin; j < q; j++)
            {
                K_star += pow(gsl_matrix_get(p_V_star, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
                K_prop += pow(gsl_matrix_get(p_V_prop, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
            }
        }
    }
    
    for(j = 0; j < q_bin; j++)
    {
        for(k = 0; k < p0_all; k++)
        {
            if(gsl_matrix_get(gamma_alpha, k, j) == 1)
            {
                K_star += pow(gsl_matrix_get(p_A_star, k, j), 2)* pow(gsl_matrix_get(M_A, k, j), -1) * 0.5;
                K_prop += pow(gsl_matrix_get(p_A_prop, k, j), 2)* pow(gsl_matrix_get(M_A, k, j), -1) * 0.5;
            }
        }
        
        K_star += pow(gsl_vector_get(p_alpha0_star, j), 2) * pow(gsl_vector_get(M_alpha0, j), -1) * 0.5;
        K_prop += pow(gsl_vector_get(p_alpha0_prop, j), 2) * pow(gsl_vector_get(M_alpha0, j), -1) * 0.5;
    }
    
    logR = -(U_prop + K_prop) + U_star + K_star;
    
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        gsl_matrix_memcpy(B, B_prop);
        
        for(j = 0; j < q; j++)
        {
            for(k = 0; k < p1_all; k++)
            {
                if(gsl_matrix_get(gamma_beta, k, j) == 1)
                {
                    gsl_matrix_set(updateNonzB, k, j, gsl_matrix_get(B_prop, k, j));
                }
            }
        }
        
        gsl_matrix_memcpy(V, V_prop);
        gsl_vector_memcpy(beta0, beta0_prop);
        
        gsl_matrix_memcpy(A, A_prop);
        
        for(j = 0; j < q_bin; j++)
        {
            for(k = 0; k < p0_all; k++)
            {
                if(gsl_matrix_get(gamma_alpha, k, j) == 1)
                {
                    gsl_matrix_set(updateNonzA, k, j, gsl_matrix_get(A_prop, k, j));
                }
            }
        }
        
        gsl_vector_memcpy(alpha0, alpha0_prop);
        *accept_group += 1;
    }
    
    
    if(M <= *numReps* 0.5)
    {
        if(*n_group <= 100)
        {
            accept_ind = *n_group-1;
            gsl_vector_set(accept_group100, accept_ind, u);
        }else if(*n_group > 100)
        {
            accept_ind = *n_group % 10 + 90 - 1;
            gsl_vector_set(accept_group100, accept_ind, u);
        }
        
        if(*n_group % 10 == 0 && *n_group >= 100)
        {
            sumAccept = 0;
            for(i = 0; i < 99; i++)
            {
                sumAccept += gsl_vector_get(accept_group100, i);
            }
            
            if(sumAccept / 100 < 0.001)
            {
                *eps_group *= 0.1;
            }else if(sumAccept / 100 < 0.05)
            {
                *eps_group *= 0.5;
            }else if(sumAccept / 100 < 0.20)
            {
                *eps_group *= 0.7;
            }else if(sumAccept / 100 < 0.60)
            {
                *eps_group *= 0.9;
            }else if(sumAccept / 100 > 0.70)
            {
                *eps_group *= 1.1;
            }else if(sumAccept / 100 > 0.80)
            {
                *eps_group *= 2;
            }else if(sumAccept / 100 > 0.95)
            {
                *eps_group *= 10;
            }
            
            accept_ind = 90;
            
            for(i = 0; i < 90; i++)
            {
                gsl_vector_set(accept_group100, i, gsl_vector_get(accept_group100, i+10));
            }
            for(i = 90; i < 99; i++)
            {
                gsl_vector_set(accept_group100, i, 0);
            }
        }
    }
    
    

    
    
    gsl_matrix_memcpy(V_ini, V);
    gsl_matrix_memcpy(V_star, V);
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(p_V_star, ii, j, rnorm(0, sqrt(gsl_matrix_get(M_V, ii, j))));
        }
    }
    
    U_star=0;
    for(ii = 0; ii < n; ii++)
    {
        gsl_vector_view V_star_ii = gsl_matrix_row(V_star, ii);
        
        c_ldmvn_noDet(&V_star_ii.vector, zero_q, 1, invSigmaV, &tempPrior);
        U_star -= tempPrior;
    }
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1, V_star, invSigmaV, 0, Delta_V_star);
    
    gsl_matrix_memcpy(p_V_ini, Delta_V_star);
    gsl_matrix_scale(p_V_ini, 0.5* *eps_V);
    gsl_matrix_add(p_V_ini, p_V_star);
    
    U_prop = 0;
    for(l = 1; l <= L_V; l++)
    {
        gsl_matrix_memcpy(V_prop, p_V_ini);
        gsl_matrix_scale(V_prop, *eps_V);
        gsl_matrix_div_elements(V_prop, M_V);
        gsl_matrix_add(V_prop, V_ini);
        
        for(ii = 0; ii < n; ii++)
        {
            for(j = 0; j < q_bin; j++)
            {
                if(gsl_matrix_get(W, ii, j) >= 0)
                {
                    gsl_matrix_set(V_prop, ii, j, gsl_matrix_get(V_star, ii, j));
                }
            }
        }
        
        gsl_matrix_set_zero(Delta_V_prop);
        
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1, V_prop, invSigmaV, 0, Delta_V_prop);
        
        if(l == L_V)
        {
            for(ii = 0; ii < n; ii++)
            {
                gsl_vector_view V_prop_ii = gsl_matrix_row(V_prop, ii);
                
                c_ldmvn_noDet(&V_prop_ii.vector, zero_q, 1, invSigmaV, &tempPrior_prop);
                U_prop -= tempPrior_prop;
            }
        }
        
        if(l < L_V)
        {
            gsl_matrix_memcpy(p_V_prop, Delta_V_prop);
            gsl_matrix_scale(p_V_prop, *eps_V);
            gsl_matrix_add(p_V_prop, p_V_ini);
            
            gsl_matrix_memcpy(V_ini, V_prop);
            gsl_matrix_memcpy(p_V_ini, p_V_prop);
        }else if(l == L_V)
        {
            gsl_matrix_memcpy(p_V_prop, Delta_V_prop);
            gsl_matrix_scale(p_V_prop, 0.5**eps_V);
            gsl_matrix_add(p_V_prop, p_V_ini);
        }
    }
    
    K_star = 0;
    K_prop = 0;
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, ii, j) <0)
            {
                K_star += pow(gsl_matrix_get(p_V_star, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
                K_prop += pow(gsl_matrix_get(p_V_prop, ii, j), 2) * pow(gsl_matrix_get(M_V, ii, j), -1) * 0.5;
            }
        }
    }
    
    logR = -(U_prop + K_prop) + U_star + K_star;
    
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        gsl_matrix_memcpy(V, V_prop);
        *accept_V += 1;
    }
    
    if(M <= *numReps* 0.5)
    {
        if(*n_group <= 100)
        {
            accept_ind = *n_group-1;
            gsl_vector_set(accept_V100, accept_ind, u);
        }else if(*n_group > 100)
        {
            accept_ind = *n_group % 10 + 90 - 1;
            gsl_vector_set(accept_V100, accept_ind, u);
        }
        
        if(*n_group % 10 == 0 && *n_group >= 100)
        {
            sumAccept = 0;
            for(i = 0; i < 99; i++)
            {
                sumAccept += gsl_vector_get(accept_V100, i);
            }
            
            if(sumAccept / 100 < 0.001)
            {
                *eps_V *= 0.1;
            }else if(sumAccept / 100 < 0.05)
            {
                *eps_V *= 0.5;
            }else if(sumAccept / 100 < 0.20)
            {
                *eps_V *= 0.7;
            }else if(sumAccept / 100 < 0.60)
            {
                *eps_V *= 0.9;
            }else if(sumAccept / 100 > 0.70)
            {
                *eps_V *= 1.1;
            }else if(sumAccept / 100 > 0.80)
            {
                *eps_V *= 2;
            }else if(sumAccept / 100 > 0.95)
            {
                *eps_V *= 10;
            }
            
            accept_ind = 90;
            
            for(i = 0; i < 90; i++)
            {
                gsl_vector_set(accept_V100, i, gsl_vector_get(accept_V100, i+10));
            }
            for(i = 90; i < 99; i++)
            {
                gsl_vector_set(accept_V100, i, 0);
            }
        }
    }
    

    
    gsl_matrix_free(XB);
    gsl_matrix_free(XB_prop);
    gsl_matrix_free(XA);
    gsl_matrix_free(XA_prop);
    
    gsl_matrix_free(LamSI);
    gsl_matrix_free(LamSI_prop);
    gsl_matrix_free(logLamSI);
    gsl_matrix_free(logLamSI_prop);
    
    gsl_matrix_free(YI);
    gsl_matrix_free(YI_LamSI);
    gsl_matrix_free(YI_LamSI_prop);
    
    gsl_matrix_free(FI);
    gsl_matrix_free(FI_prop);
    
    gsl_vector_free(zero_q);
    
    gsl_matrix_free(B_ini);
    gsl_matrix_free(B_star);
    gsl_matrix_free(B_prop);
    gsl_matrix_free(p_B_ini);
    gsl_matrix_free(p_B_star);
    gsl_matrix_free(p_B_prop);
    gsl_matrix_free(Delta_B_star);
    gsl_matrix_free(Delta_B_prop);
    
    gsl_matrix_free(V_ini);
    gsl_matrix_free(V_star);
    gsl_matrix_free(V_prop);
    gsl_matrix_free(p_V_ini);
    gsl_matrix_free(p_V_star);
    gsl_matrix_free(p_V_prop);
    gsl_matrix_free(Delta_V_star);
    gsl_matrix_free(Delta_V_prop);
    
    gsl_vector_free(beta0_ini);
    gsl_vector_free(beta0_star);
    gsl_vector_free(beta0_prop);
    gsl_vector_free(p_beta0_ini);
    gsl_vector_free(p_beta0_star);
    gsl_vector_free(p_beta0_prop);
    gsl_vector_free(Delta_beta0_star);
    gsl_vector_free(Delta_beta0_prop);
    
    gsl_matrix_free(A_ini);
    gsl_matrix_free(A_star);
    gsl_matrix_free(A_prop);
    gsl_matrix_free(p_A_ini);
    gsl_matrix_free(p_A_star);
    gsl_matrix_free(p_A_prop);
    gsl_matrix_free(Delta_A_star);
    gsl_matrix_free(Delta_A_prop);
    
    gsl_vector_free(alpha0_ini);
    gsl_vector_free(alpha0_star);
    gsl_vector_free(alpha0_prop);
    gsl_vector_free(p_alpha0_ini);
    gsl_vector_free(p_alpha0_star);
    gsl_vector_free(p_alpha0_prop);
    gsl_vector_free(Delta_alpha0_star);
    gsl_vector_free(Delta_alpha0_prop);
    
    gsl_matrix_free(BPriorV);
    gsl_matrix_free(APriorV);
    
    gsl_matrix_free(W_XAinvR);
    gsl_matrix_free(W_XAinvR_prop);
    gsl_vector_free(invRalpha0);
    gsl_vector_free(invRalpha0_prop);
    
    return;
}





/* Gradient of B (given gamma_kj = 1) */

void Grad_B_mmzip(gsl_matrix *YI_LamSI,
                  gsl_matrix *X1,
                  gsl_matrix *B,
                  gsl_matrix *gamma_beta,
                  gsl_matrix *BPriorV,
                  gsl_matrix *Delta)
{
    int q = YI_LamSI -> size2;
    int p1_all = B -> size1;
    
    gsl_matrix *GradPr = gsl_matrix_calloc(p1_all, q);
    
    gsl_matrix_set_zero(Delta);
    
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, X1, YI_LamSI, 0, Delta);
    
    gsl_matrix_memcpy(GradPr, BPriorV);
    gsl_matrix_mul_elements(GradPr, B);
    gsl_matrix_sub(Delta, GradPr);
    
    gsl_matrix_mul_elements(Delta, gamma_beta);
    
    gsl_matrix_free(GradPr);
    
    return;
}






/* Gradient of V */

void Grad_V_mmzip(gsl_matrix *YI_LamSI,
                  gsl_matrix *V,
                  gsl_matrix *W,
                  gsl_matrix *invSigmaV,
                  gsl_matrix *Delta)

{
    int ii, j;
    
    int n = YI_LamSI -> size1;
    int q = YI_LamSI -> size2;
    int q_bin = W -> size2;
    
    gsl_matrix *GradPr = gsl_matrix_calloc(n, q);
    
    gsl_matrix_set_zero(Delta);
    
    gsl_matrix_memcpy(Delta, YI_LamSI);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, V, invSigmaV, 0, GradPr);
    gsl_matrix_sub(Delta, GradPr);
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, ii, j) < 0)
            {
                gsl_matrix_set(Delta, ii, j, 0);
            }
        }
    }
    
    gsl_matrix_free(GradPr);
    
    return;
}




/* log-likelihood for the poisson model part */

void LH_group_mmzip(gsl_matrix *Y,
                    gsl_matrix *XB,
                    gsl_matrix *Fixed,
                    gsl_vector *beta0,
                    gsl_matrix *V,
                    gsl_matrix *W,
                    double *logLH_val)
{
    int i, jj, jj_temp;
    double tempB;
    
    int n = XB -> size1;
    int q = XB -> size2;
    int q_bin = W -> size2;
    
    *logLH_val = 0;
    
    for(jj = 0; jj < q; jj++)
    {
        jj_temp = (jj < q_bin) ? jj : (q_bin-1);
        for(i=0; i<n; i++)
        {
            if((jj < q_bin && gsl_matrix_get(W, i, jj_temp) >= 0) || jj >= q_bin)
            {
                tempB = gsl_vector_get(beta0, jj)+gsl_matrix_get(XB, i, jj)+gsl_matrix_get(V, i, jj)+gsl_matrix_get(Fixed, i, jj);
                
                *logLH_val += gsl_matrix_get(Y, i, jj)*tempB - exp(tempB);
            }
        }
    }
    
    return;
}




/* Gradient of beta0 */

void Grad_beta0_mmzip(gsl_matrix *YI_LamSI,
                      gsl_vector *beta0,
                      gsl_vector *mu_beta0,
                      double sigSq_beta0,
                      gsl_vector *Delta)
{
    int n = YI_LamSI -> size1;
    int q = YI_LamSI -> size2;
    
    gsl_vector *ones_n = gsl_vector_calloc(n);
    gsl_vector *GradPr = gsl_vector_calloc(q);
    
    gsl_vector_set_zero(Delta);
    gsl_vector_set_all(ones_n, 1);
    
    gsl_blas_dgemv(CblasTrans, 1, YI_LamSI, ones_n, 0, Delta);
    
    gsl_vector_memcpy(GradPr, beta0);
    gsl_vector_sub(GradPr, mu_beta0);
    gsl_vector_scale(GradPr, pow(sigSq_beta0, -1));
    
    gsl_vector_sub(Delta, GradPr);
    
    gsl_vector_free(ones_n);
    gsl_vector_free(GradPr);
    
    return;
}


/* Hessian of A and alpha0 */

void Hessian_A_alpha0_mmzip(gsl_matrix *XA,
                            gsl_matrix *X0,
                            gsl_vector *alpha0,
                            gsl_vector *sum_X0sq,
                            gsl_matrix *A,
                            gsl_matrix *W,
                            gsl_matrix *Y,
                            gsl_matrix *LamSI,
                            gsl_matrix *invR,
                            gsl_matrix *APriorV,
                            double sigSq_alpha0,
                            gsl_vector *M_alpha0,
                            gsl_matrix *M_A)
{
    int n = XA -> size1;
    int p0_all = M_A -> size1;
    int q_bin = M_A -> size2;
    
    int i, jj, kk;
    double tempA, val_a0_1, val_A_1, val_a0_2, val_A_2, val, PDFval, CDFval;
    
    gsl_matrix *term1_A = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *term2_A = gsl_matrix_calloc(p0_all, q_bin);
    
    gsl_vector *term1_alpha0 = gsl_vector_calloc(q_bin);
    gsl_vector *term2_alpha0 = gsl_vector_calloc(q_bin);
    
    for(kk = 0; kk < p0_all; kk++)
    {
        for(jj = 0; jj < q_bin; jj++)
        {
            for(i=0; i<n; i++)
            {
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    tempA = gsl_vector_get(alpha0, jj) + gsl_matrix_get(XA, i, jj);
                    PDFval = dnorm(tempA, 0, 1, 0);
                    CDFval = pnorm(tempA, 0, 1, 1, 0);
                    
                    val_a0_1 = -gsl_matrix_get(Y, i, jj)* PDFval * (tempA * CDFval + PDFval)/pow(CDFval, 2);
                    val_A_1 = val_a0_1 * pow(gsl_matrix_get(X0, i, kk), 2);
                                        
                    val_a0_2 = gsl_matrix_get(LamSI, i, jj) * PDFval * (tempA*CDFval + 2*PDFval) / pow(CDFval, 3);
                    val_A_2 = val_a0_2 * pow(gsl_matrix_get(X0, i, kk), 2);
                    
                    gsl_matrix_set(term1_A, kk, jj, gsl_matrix_get(term1_A, kk, jj) + val_A_1+val_A_2);
                    
                    if(kk == 0)
                    {
                        gsl_vector_set(term1_alpha0, jj, gsl_vector_get(term1_alpha0, jj)+val_a0_1+val_a0_2);
                    }
                }
            }
        }
    }
    
    for(jj = 0; jj < q_bin; jj++)
    {
        val = n*gsl_matrix_get(invR, jj, jj) + pow(sigSq_alpha0, -1);
        gsl_vector_set(term2_alpha0, jj, val);
    }
    
    for(jj = 0; jj < q_bin; jj++)
    {
        for(kk = 0; kk < p0_all; kk++)
        {
            val = gsl_matrix_get(invR, jj, jj) * gsl_vector_get(sum_X0sq, kk);
            gsl_matrix_set(term2_A, kk, jj, val);
        }
    }
    gsl_matrix_add(term2_A, APriorV);
    
    
    for(jj = 0; jj < q_bin; jj++)
    {
        val = gsl_vector_get(term1_alpha0, jj) + gsl_vector_get(term2_alpha0, jj);
        if(val <= 0)
        {
            gsl_vector_set(M_alpha0, jj, gsl_vector_get(term2_alpha0, jj));
        }else
        {
            gsl_vector_set(M_alpha0, jj, val);
        }
        
        for(kk = 0; kk < p0_all; kk++)
        {
            val = gsl_matrix_get(term1_A, kk, jj) + gsl_matrix_get(term2_A, kk, jj);
            if(val <= 0)
            {
                gsl_matrix_set(M_A, kk, jj, gsl_matrix_get(term2_A, kk, jj));
            }else
            {
                gsl_matrix_set(M_A, kk, jj, val);
            }
        }
    }
    
    gsl_matrix_free(term1_A);
    gsl_matrix_free(term2_A);
    gsl_vector_free(term1_alpha0);
    gsl_vector_free(term2_alpha0);
    
    return;
}




/* Hessian of beta0 */
void Hessian_beta0_mmzip(gsl_matrix *LamSI,
                         double sigSq_beta0,
                         gsl_vector *M_beta0)
{
    int n = LamSI -> size1;
    int q = LamSI -> size2;
    
    gsl_vector *ones_n = gsl_vector_calloc(n);
    gsl_vector *priVec = gsl_vector_calloc(q);
    
    gsl_vector_set_zero(M_beta0);
    gsl_vector_set_all(ones_n, 1);
    gsl_vector_set_all(priVec, 1/sigSq_beta0);
    
    gsl_blas_dgemv(CblasTrans, 1, LamSI, ones_n, 0, M_beta0);
    gsl_vector_add(M_beta0, priVec);
    
    gsl_vector_free(ones_n);
    gsl_vector_free(priVec);
    
    return;
}

/* Hessian of V */

void Hessian_V_mmzip(gsl_matrix *LamSI,
                  gsl_matrix *invSigmaV,
                  gsl_matrix *M_V)
{
    int ii, j;
    
    int n = LamSI -> size1;
    int q = LamSI -> size2;
    
    gsl_matrix *priMat = gsl_matrix_calloc(n, q);
    gsl_matrix_memcpy(M_V, LamSI);
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(M_V, ii, j, gsl_matrix_get(M_V, ii, j)+gsl_matrix_get(invSigmaV, j, j));
        }
    }
    gsl_matrix_free(priMat);
    
    return;
}


/* Hessian of B */

void Hessian_B_mmzip(gsl_matrix *LamSI,
                  gsl_matrix *X1,
                  gsl_matrix *BPriorV,
                  gsl_matrix *M_B)
{
    int n = LamSI -> size1;
    int p1_all = M_B -> size1;
    int ii, j;
    
    gsl_matrix *Xsq = gsl_matrix_calloc(n, p1_all);
    gsl_matrix_set_zero(M_B);
    
    for(ii = 0; ii < n; ii++)
    {
        for(j = 0; j < p1_all; j++)
        {
            gsl_matrix_set(Xsq, ii, j, pow(gsl_matrix_get(X1, ii, j), 2));
        }
    }
    
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Xsq, LamSI, 0, M_B);
    gsl_matrix_add(M_B, BPriorV);
    
    gsl_matrix_free(Xsq);
    
    return;
}




/* updating B, gamma_beta -- SSVS */

void updateRP_beta_mmzip_SSVS(int *p_adj,
                              gsl_matrix *Y,
                              gsl_matrix *X1,
                              gsl_matrix *X0,
                              gsl_vector *xi,
                              gsl_vector *beta0,
                              gsl_matrix *B,
                              gsl_matrix *V,
                              gsl_matrix *gamma_beta,
                              gsl_vector *alpha0,
                              gsl_matrix *A,
                              gsl_matrix *W,
                              gsl_matrix *updateNonzB,
                              gsl_vector *sigSq_beta,
                              gsl_vector *v_beta,
                              gsl_vector *omega_beta,
                              gsl_matrix *beta_prop_var,
                              gsl_matrix *accept_B,
                              gsl_matrix *n_B,
                              double pU_B_ssvs)
{
    double beta_prop;
    double logR, choice;
    double logLH, logLH_prop, logPrior, logPrior_prop, logProp_new, logProp;
    double sumGam;
    double p_add, p_del;
    double tempB, tempA;
    double term1, term2, term1_prop, term2_prop;
    
    int u, i, jj, k, l, m, lInx, count, move;
    
    int n = Y -> size1;
    int q = B -> size2;
    int p1_all = B -> size1;
    int q_bin = A -> size2;
    
    p_add = (double) 1/3;
    p_del = (double) 1/3;
    
    gsl_matrix *gamma_prop = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *B_prop = gsl_matrix_calloc(p1_all, q);
    gsl_matrix *XA = gsl_matrix_calloc(n, q_bin);
    gsl_matrix *XB = gsl_matrix_calloc(n, q);
    gsl_matrix *Fixed = gsl_matrix_calloc(n, q);
    gsl_matrix *expXB = gsl_matrix_calloc(n, q);
    gsl_vector *expXB_prop_jj = gsl_vector_calloc(n);
    gsl_vector *XB_prop_jj = gsl_vector_calloc(n);
    gsl_matrix *YI = gsl_matrix_calloc(n, q);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X0, A, 0, XA);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X1, B, 0, XB);
    
    for(jj = 0; jj < q; jj++)
    {
        for(i = 0; i < n; i++)
        {
            if(jj < q_bin)
            {
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    tempA = gsl_matrix_get(XA, i, jj) + gsl_vector_get(alpha0, jj);
                    tempB = gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA, 0, 1, 1, 1);
                    gsl_matrix_set(YI, i, jj, gsl_matrix_get(Y, i, jj));
                    gsl_matrix_set(expXB, i, jj, exp(gsl_matrix_get(XB, i, jj)));
                    gsl_matrix_set(Fixed, i, jj, exp(tempB));
                }
            }else
            {
                tempB = gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                gsl_matrix_set(YI, i, jj, gsl_matrix_get(Y, i, jj));
                gsl_matrix_set(expXB, i, jj, exp(gsl_matrix_get(XB, i, jj)));
                gsl_matrix_set(Fixed, i, jj, exp(tempB));
            }
        }
    }
    
    for(jj = 0; jj < q; jj++)
    {
        if(runif(0, 1) < pU_B_ssvs)
        {
            logLH = 0;
            logLH_prop = 0;
            logPrior_prop = 0;
            logPrior = 0;
            logProp = 0;
            logProp_new = 0;
            
            gsl_matrix_memcpy(gamma_prop, gamma_beta);
            gsl_matrix_memcpy(B_prop, B);
            
            sumGam = 0;
            for(i = 0; i < p1_all - *p_adj; i++)
            {
                sumGam += gsl_matrix_get(gamma_beta, i, jj);
            }
            
            /* selecting a move */
            /* move: 1=add, 2=delete, 3=swap */
            
            if((p1_all - *p_adj) == 1)
            {
                if(gsl_matrix_get(gamma_beta, 0, jj) == 1) move = 2;
                if(gsl_matrix_get(gamma_beta, 0, jj) == 0) move = 1;
            }
            if((p1_all - *p_adj) > 1)
            {
                if(sumGam == (p1_all - *p_adj)) move = 2;
                if(sumGam == 0) move = 1;
                if(sumGam != (p1_all - *p_adj) && sumGam != 0)
                {
                    choice  = runif(0, 1);
                    move = 1;
                    if(choice > p_add) move = 2;
                    if(choice > p_add + p_del) move = 3;
                }
            }
            
            if(move == 1) /* add move */
            {
                if((p1_all - *p_adj) == 1)
                {
                    l = 0;
                }else
                {
                    choice  = runif(0, 1);
                    lInx = (int) (choice * ((double) p1_all - (double) *p_adj - (double) sumGam)) + 1;
                    count = 0;
                    k = -1;
                    while(lInx != count)
                    {
                        k += 1;
                        if(gsl_matrix_get(gamma_beta, k, jj) == 0)
                        {
                            count += 1;
                        }
                    }
                    l = k;
                }
                gsl_matrix_set(n_B, l, jj, gsl_matrix_get(n_B, l, jj)+1);
                
                gsl_matrix_set(gamma_prop, l, jj, 1);
                beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, jj), sqrt(gsl_matrix_get(beta_prop_var, l, jj)));
                
                gsl_matrix_set(B_prop, l, jj, beta_prop);
                
            }else if(move == 2) /* delete move */
            {
                if((p1_all - *p_adj) == 1)
                {
                    l = 0;
                }else
                {
                    choice  = runif(0, 1);
                    
                    lInx = (int) (choice * ((double) sumGam)) + 1;
                    count = 0;
                    k = -1;
                    while(lInx != count)
                    {
                        k += 1;
                        if(gsl_matrix_get(gamma_beta, k, jj) == 1)
                        {
                            count += 1;
                        }
                    }
                    l = k;
                }
                
                gsl_matrix_set(n_B, l, jj, gsl_matrix_get(n_B, l, jj)+1);
                
                gsl_matrix_set(gamma_prop, l, jj, 0);
                beta_prop = 0;
                
                gsl_matrix_set(B_prop, l, jj, beta_prop);
                
            }else if(move == 3) /* swap move*/
            {
                choice  = runif(0, 1);
                lInx = (int) (choice * ((double) p1_all - (double) *p_adj - (double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma_beta, k, jj) == 0)
                    {
                        count += 1;
                    }
                }
                l = k;
                
                choice  = runif(0, 1);
                lInx = (int) (choice * ((double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma_beta, k, jj) == 1)
                    {
                        count += 1;
                    }
                }
                m = k;
                
                gsl_matrix_set(n_B, l, jj, gsl_matrix_get(n_B, l, jj)+1);
                gsl_matrix_set(n_B, m, jj, gsl_matrix_get(n_B, m, jj)+1);
                
                gsl_matrix_set(gamma_prop, l, jj, 1);
                gsl_matrix_set(gamma_prop, m, jj, 0);
                
                beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, jj), sqrt(gsl_matrix_get(beta_prop_var, l, jj)));
                
                gsl_matrix_set(B_prop, l, jj, beta_prop);
                gsl_matrix_set(B_prop, m, jj, 0);
            }
            
            gsl_vector_view Fixed_jj = gsl_matrix_column(Fixed, jj);
            gsl_vector_view expXB_jj = gsl_matrix_column(expXB, jj);
            gsl_vector_view YI_jj = gsl_matrix_column(YI, jj);
            gsl_vector_view XB_jj = gsl_matrix_column(XB, jj);
            
            gsl_blas_ddot(&YI_jj.vector, &XB_jj.vector, &term1);
            gsl_blas_ddot(&Fixed_jj.vector, &expXB_jj.vector, &term2);
            
            gsl_vector_view B_prop_jj = gsl_matrix_column(B_prop, jj);
            gsl_blas_dgemv(CblasNoTrans, 1, X1, &B_prop_jj.vector, 0, XB_prop_jj);
            
            gsl_vector_set_zero(expXB_prop_jj);
            for(i = 0; i < n; i++)
            {
                if(jj < q_bin)
                {
                    if(gsl_matrix_get(W, i, jj) >= 0)
                    {
                        gsl_vector_set(expXB_prop_jj, i, exp(gsl_vector_get(XB_prop_jj, i)));
                    }
                }else
                {
                    gsl_vector_set(expXB_prop_jj, i, exp(gsl_vector_get(XB_prop_jj, i)));
                }
            }
            
            gsl_blas_ddot(&YI_jj.vector, XB_prop_jj, &term1_prop);
            gsl_blas_ddot(&Fixed_jj.vector, expXB_prop_jj, &term2_prop);
            
            logLH = term1 - term2;
            logLH_prop = term1_prop - term2_prop;
            
            if(move == 1)/* add move */
            {
                logPrior_prop = dnorm(beta_prop, 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v_beta, jj), 1);
                logPrior_prop += log(gsl_vector_get(omega_beta, l));
                logPrior = log(1-gsl_vector_get(omega_beta, l));
                
                logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, jj), sqrt(gsl_matrix_get(beta_prop_var, l, jj)), 1);
                logProp =  log(p1_all - *p_adj-sumGam) - log(sumGam + 1);
                
            }else if(move == 2) /* delete move */
            {
                logPrior_prop = log(1-gsl_vector_get(omega_beta, l));
                logPrior = dnorm(gsl_matrix_get(B, l, jj), 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v_beta, jj), 1);
                logPrior += log(gsl_vector_get(omega_beta, l));
                
                logProp_new =  0;
                logProp = dnorm(gsl_matrix_get(updateNonzB, l, jj), beta_prop, sqrt(gsl_matrix_get(beta_prop_var, l, jj)), 1);
                logProp += log(sumGam) - log(p1_all - *p_adj-sumGam+1);
                
            }else if(move == 3) /* swap move*/
            {
                logPrior_prop = dnorm(beta_prop, 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v_beta, jj), 1);
                logPrior_prop += log(gsl_vector_get(omega_beta, l)) + log(1-gsl_vector_get(omega_beta, m));
                
                logPrior = dnorm(gsl_matrix_get(B, m, jj), 0, sqrt(gsl_vector_get(sigSq_beta, m))*gsl_vector_get(v_beta, jj), 1);
                logPrior += log(gsl_vector_get(omega_beta, m)) + log(1-gsl_vector_get(omega_beta, l));
                
                logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, jj), sqrt(gsl_matrix_get(beta_prop_var, l, jj)), 1);
                logProp = dnorm(gsl_matrix_get(updateNonzB, m, jj), beta_prop, sqrt(gsl_matrix_get(beta_prop_var, m, jj)), 1);
            }
            /**/
            
            
            if(move == 1 || move == 2 || move == 3)
            {
                logR = logLH_prop - logLH + logPrior_prop - logPrior + logProp - logProp_new;
                
                u = log(runif(0, 1)) < logR;
                
                if(u == 1)
                {
                    gsl_matrix_memcpy(gamma_beta, gamma_prop);
                    
                    if(move == 1 || move == 3)
                    {
                        gsl_matrix_set(updateNonzB, l, jj, beta_prop);
                    }else if(move == 2)
                    {
                        gsl_matrix_set(updateNonzB, l, jj, 0);
                    }
                    if(move == 1 || move == 2)
                    {
                        gsl_matrix_set(B, l, jj, beta_prop);
                        gsl_matrix_set(accept_B, l, jj, gsl_matrix_get(accept_B, l, jj)+1);
                    }
                    if(move == 3)
                    {
                        gsl_matrix_set(B, l, jj, beta_prop);
                        gsl_matrix_set(B, m, jj, 0);
                        gsl_matrix_set(accept_B, l, jj, gsl_matrix_get(accept_B, l, jj)+1);
                        gsl_matrix_set(accept_B, m, jj, gsl_matrix_get(accept_B, m, jj)+1);
                    }
                }
            }
        }
        
    }
    
    gsl_matrix_free(gamma_prop);
    gsl_matrix_free(B_prop);
    gsl_matrix_free(XA);
    gsl_matrix_free(XB);
    gsl_matrix_free(Fixed);
    gsl_matrix_free(expXB);
    gsl_vector_free(expXB_prop_jj);
    gsl_vector_free(XB_prop_jj);
    
    gsl_matrix_free(YI);
    
    return;
}



/* updating A, gamma_alpha -- SSVS */

void updateRP_alpha_mmzip_SSVS(int *p_adj,
                               gsl_matrix *Y,
                               gsl_matrix *X1,
                               gsl_matrix *X0,
                               gsl_vector *xi,
                               gsl_vector *beta0,
                               gsl_matrix *B,
                               gsl_matrix *V,
                               gsl_vector *alpha0,
                               gsl_matrix *A,
                               gsl_matrix *W,
                               gsl_matrix *invR,
                               gsl_matrix *gamma_alpha,
                               gsl_matrix *updateNonzA,
                               gsl_vector *sigSq_alpha,
                               gsl_vector *v_alpha,
                               gsl_vector *omega_alpha,
                               gsl_matrix *alpha_prop_var,
                               gsl_matrix *accept_A,
                               gsl_matrix *n_A,
                               double pU_A_ssvs)
{
    double alpha_prop;
    double logR, choice;
    double logLH, logLH_prop, logPrior, logPrior_prop, logProp_new, logProp;
    double sumGam, zalphaj, zalphaj_prop, zalphal, sum_zalphal, wrj, zalp0;
    double p_add, p_del;
    double tempA, tempA_prop, tempB, tempB_prop, xbeta;
    
    int u, i, jj, k, l, m, ll, lInx, count, move;
    
    int q_bin = A -> size2;
    int p0_all = A -> size1;
    int p0 = p0_all - *p_adj;
    int n = Y -> size1;
    
    p_add = (double) 1/3;
    p_del = (double) 1/3;
    
    gsl_matrix *gamma_prop = gsl_matrix_calloc(p0_all, q_bin);
    gsl_matrix *A_prop = gsl_matrix_calloc(p0_all, q_bin);
    
    gsl_vector *Wrow = gsl_vector_calloc(q_bin);
    gsl_vector *alphaj = gsl_vector_calloc(p0_all);
    gsl_vector *alphaj_prop = gsl_vector_calloc(p0_all);
    gsl_vector *alphal = gsl_vector_calloc(p0_all);
    
    gsl_vector *invRrow = gsl_vector_calloc(q_bin);
    gsl_vector *meanW = gsl_vector_calloc(q_bin);
    gsl_vector *meanW_prop = gsl_vector_calloc(q_bin);
    
    gsl_matrix_set_zero(updateNonzA);
    
    jj = (int) runif(0, q_bin);
    
    if(runif(0, 1) < 1)
    {
        logLH = 0;
        logLH_prop = 0;
        logPrior_prop = 0;
        logPrior = 0;
        logProp = 0;
        logProp_new = 0;
        
        gsl_matrix_memcpy(gamma_prop, gamma_alpha);
        gsl_matrix_memcpy(A_prop, A);
        
        gsl_matrix_get_row(invRrow, invR, jj);
        
        sumGam = 0;
        for(i = 0; i < p0; i++)
        {
            sumGam += gsl_matrix_get(gamma_alpha, i, jj);
        }
        
        /* selecting a move */
        /* move: 1=add, 2=delete, 3=swap */
        
        if((p0) == 1)
        {
            if(gsl_matrix_get(gamma_alpha, 0, jj) == 1) move = 2;
            if(gsl_matrix_get(gamma_alpha, 0, jj) == 0) move = 1;
        }
        if((p0) > 1)
        {
            if(sumGam == (p0)) move = 2;
            if(sumGam == 0) move = 1;
            if(sumGam != (p0) && sumGam != 0)
            {
                choice  = runif(0, 1);
                move = 1;
                if(choice > p_add) move = 2;
                if(choice > p_add + p_del) move = 3;
            }
        }
        
        /* for add move */
        if(move == 1)
        {
            if(p0 == 1)
            {
                l = 0;
            }else
            {
                choice  = runif(0, 1);
                lInx = (int) (choice * ((double) p0_all - (double) *p_adj - (double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma_alpha, k, jj) == 0)
                    {
                        count += 1;
                    }
                }
                l = k;
            }
            
            gsl_matrix_set(n_A, l, jj, gsl_matrix_get(n_A, l, jj)+1);
            
            gsl_matrix_set(gamma_prop, l, jj, 1);
            alpha_prop = rnorm(gsl_matrix_get(updateNonzA, l, jj), sqrt(gsl_matrix_get(alpha_prop_var, l, jj)));
            
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            
            for(i = 0; i < n; i++)
            {
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_matrix_get_col(alphaj, A, jj);
                
                gsl_blas_ddot(&X0row.vector, alphaj, &zalphaj);
                logLH += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj, 2);
                
                sum_zalphal = 0;
                for(ll = 0; ll < q_bin; ll++)
                {
                    if(ll != jj)
                    {
                        gsl_matrix_get_col(alphal, A, ll);
                        gsl_blas_ddot(&X0row.vector, alphal, &zalphal);
                        sum_zalphal += zalphal * gsl_matrix_get(invR, jj, ll);
                    }
                }
                logLH += - zalphaj * sum_zalphal;
                
                gsl_blas_ddot(Wrow, invRrow, &wrj);
                logLH += zalphaj * wrj;
                
                gsl_blas_ddot(alpha0, invRrow, &zalp0);
                logLH += -zalphaj * zalp0;
                
                gsl_vector_memcpy(alphaj_prop, alphaj);
                gsl_vector_set(alphaj_prop, l, alpha_prop);
                gsl_blas_ddot(&X0row.vector, alphaj_prop, &zalphaj_prop);
                
                logLH_prop += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj_prop, 2);
                logLH_prop += -zalphaj_prop * sum_zalphal;
                logLH_prop += zalphaj_prop * wrj;
                logLH_prop += -zalphaj_prop * zalp0;
                
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    
                    tempA = gsl_vector_get(alpha0, jj) + zalphaj;
                    tempA_prop = gsl_vector_get(alpha0, jj) + zalphaj_prop;
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA, 0, 1, 1, 1);
                    tempB_prop =  xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA_prop, 0, 1, 1, 1);
                    
                    logLH       += -pnorm(tempA, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += -pnorm(tempA_prop, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
            }
            
            logPrior_prop = dnorm(alpha_prop, 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v_alpha, jj), 1);
            logPrior_prop += log(gsl_vector_get(omega_alpha, l));
            logPrior = log(1-gsl_vector_get(omega_alpha, l));
            
            logProp_new = dnorm(alpha_prop, gsl_matrix_get(updateNonzA, l, jj), sqrt(gsl_matrix_get(alpha_prop_var, l, jj)), 1);
            logProp =  log(p0-sumGam) - log(sumGam + 1);
            
        }
        
        /* for delete move */
        if(move == 2)
        {
            if((p0) == 1) l = 0;
            else
            {
                choice  = runif(0, 1);
                
                lInx = (int) (choice * ((double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma_alpha, k, jj) == 1)
                    {
                        count += 1;
                    }
                }
                l = k;
            }
            
            gsl_matrix_set(n_A, l, jj, gsl_matrix_get(n_A, l, jj)+1);
            
            gsl_matrix_set(gamma_prop, l, jj, 0);
            alpha_prop = 0;
            
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            
            for(i = 0; i < n; i++)
            {
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_matrix_get_col(alphaj, A, jj);
                
                gsl_blas_ddot(&X0row.vector, alphaj, &zalphaj);
                logLH += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj, 2);
                
                sum_zalphal = 0;
                for(ll = 0; ll < q_bin; ll++)
                {
                    if(ll != jj)
                    {
                        gsl_matrix_get_col(alphal, A, ll);
                        gsl_blas_ddot(&X0row.vector, alphal, &zalphal);
                        sum_zalphal += zalphal * gsl_matrix_get(invR, jj, ll);
                    }
                }
                logLH += - zalphaj * sum_zalphal;
                
                gsl_blas_ddot(Wrow, invRrow, &wrj);
                logLH += zalphaj * wrj;
                
                gsl_blas_ddot(alpha0, invRrow, &zalp0);
                logLH += -zalphaj * zalp0;
                
                gsl_vector_memcpy(alphaj_prop, alphaj);
                gsl_vector_set(alphaj_prop, l, alpha_prop);
                gsl_blas_ddot(&X0row.vector, alphaj_prop, &zalphaj_prop);
                
                logLH_prop += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj_prop, 2);
                logLH_prop += -zalphaj_prop * sum_zalphal;
                logLH_prop += zalphaj_prop * wrj;
                logLH_prop += -zalphaj_prop * zalp0;
                
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    
                    tempA = gsl_vector_get(alpha0, jj) + zalphaj;
                    tempA_prop = gsl_vector_get(alpha0, jj) + zalphaj_prop;
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA, 0, 1, 1, 1);
                    tempB_prop =  xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA_prop, 0, 1, 1, 1);
                    
                    logLH       += -pnorm(tempA, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += -pnorm(tempA_prop, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
            }
            
            logPrior_prop = log(1-gsl_vector_get(omega_alpha, l));
            logPrior = dnorm(gsl_matrix_get(A, l, jj), 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v_alpha, jj), 1);
            logPrior += log(gsl_vector_get(omega_alpha, l));
            
            logProp_new =  0;
            logProp = dnorm(gsl_matrix_get(updateNonzA, l, jj), alpha_prop, sqrt(gsl_matrix_get(alpha_prop_var, l, jj)), 1);
            logProp += log(sumGam) - log(p0-sumGam+1);
        }
        
        /* for swap move */
        if(move == 3)
        {
            choice  = runif(0, 1);
            lInx = (int) (choice * ((double) p0_all - (double) *p_adj - (double) sumGam)) + 1;
            count = 0;
            k = -1;
            while(lInx != count)
            {
                k += 1;
                if(gsl_matrix_get(gamma_alpha, k, jj) == 0)
                {
                    count += 1;
                }
            }
            l = k;
            
            choice  = runif(0, 1);
            lInx = (int) (choice * ((double) sumGam)) + 1;
            count = 0;
            k = -1;
            while(lInx != count)
            {
                k += 1;
                if(gsl_matrix_get(gamma_alpha, k, jj) == 1)
                {
                    count += 1;
                }
            }
            m = k;
            
            gsl_matrix_set(n_A, l, jj, gsl_matrix_get(n_A, l, jj)+1);
            gsl_matrix_set(n_A, m, jj, gsl_matrix_get(n_A, m, jj)+1);
            
            gsl_matrix_set(gamma_prop, l, jj, 1);
            gsl_matrix_set(gamma_prop, m, jj, 0);
            
            alpha_prop = rnorm(gsl_matrix_get(updateNonzA, l, jj), sqrt(gsl_matrix_get(alpha_prop_var, l, jj)));
            
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            gsl_matrix_set(A_prop, m, jj, 0);
            
            for(i = 0; i < n; i++)
            {
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_matrix_get_col(alphaj, A, jj);
                
                gsl_blas_ddot(&X0row.vector, alphaj, &zalphaj);
                logLH += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj, 2);
                
                sum_zalphal = 0;
                for(ll = 0; ll < q_bin; ll++)
                {
                    if(ll != jj)
                    {
                        gsl_matrix_get_col(alphal, A, ll);
                        gsl_blas_ddot(&X0row.vector, alphal, &zalphal);
                        sum_zalphal += zalphal * gsl_matrix_get(invR, jj, ll);
                    }
                }
                logLH += - zalphaj * sum_zalphal;
                
                gsl_blas_ddot(Wrow, invRrow, &wrj);
                logLH += zalphaj * wrj;
                
                gsl_blas_ddot(alpha0, invRrow, &zalp0);
                logLH += -zalphaj * zalp0;
                
                gsl_vector_memcpy(alphaj_prop, alphaj);
                gsl_vector_set(alphaj_prop, l, alpha_prop);
                gsl_vector_set(alphaj_prop, m, 0);
                gsl_blas_ddot(&X0row.vector, alphaj_prop, &zalphaj_prop);
                
                logLH_prop += -0.5 * gsl_matrix_get(invR, jj, jj) * pow(zalphaj_prop, 2);
                logLH_prop += -zalphaj_prop * sum_zalphal;
                logLH_prop += zalphaj_prop * wrj;
                logLH_prop += -zalphaj_prop * zalp0;
                
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    
                    tempA = gsl_vector_get(alpha0, jj) + zalphaj;
                    tempA_prop = gsl_vector_get(alpha0, jj) + zalphaj_prop;
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA, 0, 1, 1, 1);
                    tempB_prop =  xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(tempA_prop, 0, 1, 1, 1);
                    
                    logLH       += -pnorm(tempA, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += -pnorm(tempA_prop, 0, 1, 1, 1)*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
            }
            
            logPrior_prop = dnorm(alpha_prop, 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v_alpha, jj), 1);
            logPrior_prop += log(gsl_vector_get(omega_alpha, l)) + log(1-gsl_vector_get(omega_alpha, m));
            
            logPrior = dnorm(gsl_matrix_get(A, m, jj), 0, sqrt(gsl_vector_get(sigSq_alpha, m))*gsl_vector_get(v_alpha, jj), 1);
            logPrior += log(gsl_vector_get(omega_alpha, m)) + log(1-gsl_vector_get(omega_alpha, l));
            
            logProp_new = dnorm(alpha_prop, gsl_matrix_get(updateNonzA, l, jj), sqrt(gsl_matrix_get(alpha_prop_var, l, jj)), 1);
            logProp = dnorm(gsl_matrix_get(updateNonzA, m, jj), alpha_prop, sqrt(gsl_matrix_get(alpha_prop_var, m, jj)), 1);
            
        }
        
        if(move == 1 || move == 2 || move == 3)
        {
            logR = logLH_prop - logLH + logPrior_prop - logPrior + logProp - logProp_new;
            
            u = log(runif(0, 1)) < logR;
            
            if(u == 1)
            {
                gsl_matrix_swap(gamma_alpha, gamma_prop);
                
                if(move == 1 || move == 3)
                {
                    gsl_matrix_set(updateNonzA, l, jj, alpha_prop);
                }else if(move == 2)
                {
                    gsl_matrix_set(updateNonzA, l, jj, 0);
                }
                if(move == 1 || move == 2)
                {
                    gsl_matrix_set(A, l, jj, alpha_prop);
                    gsl_matrix_set(accept_A, l, jj, gsl_matrix_get(accept_A, l, jj)+1);
                }
                if(move == 3)
                {
                    gsl_matrix_set(A, l, jj, alpha_prop);
                    gsl_matrix_set(A, m, jj, 0);
                    gsl_matrix_set(accept_A, l, jj, gsl_matrix_get(accept_A, l, jj)+1);
                    gsl_matrix_set(accept_A, m, jj, gsl_matrix_get(accept_A, m, jj)+1);
                }
            }
        }
    }
    
    gsl_matrix_free(gamma_prop);
    gsl_matrix_free(A_prop);
    
    gsl_vector_free(Wrow);
    gsl_vector_free(alphaj);
    gsl_vector_free(alphaj_prop);
    gsl_vector_free(alphal);
    
    gsl_vector_free(invRrow);
    gsl_vector_free(meanW);
    gsl_vector_free(meanW_prop);
    
    
    return;
}








/* updating SigmaV */

void update_SigmaV_mmzip(gsl_matrix *V,
                         gsl_matrix *SigmaV,
                         gsl_matrix *invSigmaV,
                         gsl_matrix *cholSigmaV,
                         gsl_matrix *Psi0,
                         double rho0)
{
    int j;
    double df;
    int n = V -> size1;
    int q = V -> size2;
    
    gsl_vector *Vrow = gsl_vector_calloc(q);
    
    gsl_matrix *VV = gsl_matrix_calloc(q, q);
    gsl_matrix *Sum = gsl_matrix_calloc(q, q);
    gsl_matrix_memcpy(Sum, Psi0);
    
    for(j = 0; j < n; j++)
    {
        gsl_matrix_get_row(Vrow, V, j);
        
        gsl_blas_dger(1, Vrow, Vrow, VV);
        gsl_matrix_add(Sum, VV);
        gsl_matrix_set_zero(VV);
    }
    
    df = (int) rho0 + n;
    
    c_riwishart3(df, Sum, SigmaV, invSigmaV, cholSigmaV);
    
    gsl_matrix_free(VV);
    gsl_matrix_free(Sum);
    gsl_vector_free(Vrow);
    
    return;
}








void Cal_logPost_mmzip(int *p_adj,
                       gsl_matrix *Y,
                       gsl_matrix *X1,
                       gsl_matrix *X0,
                       gsl_vector *xi,
                       gsl_vector *beta0,
                       gsl_matrix *B,
                       gsl_matrix *V,
                       gsl_matrix *gamma_beta,
                       gsl_vector *alpha0,
                       gsl_matrix *A,
                       gsl_matrix *W,
                       gsl_matrix *invSigmaV,
                       gsl_matrix *invR,
                       gsl_matrix *gamma_alpha,
                       gsl_vector *sigSq_beta,
                       gsl_vector *v_beta,
                       gsl_vector *omega_beta,
                       gsl_vector *sigSq_alpha,
                       gsl_vector *v_alpha,
                       gsl_vector *omega_alpha,
                       gsl_vector *a_beta,
                       gsl_vector *b_beta,
                       gsl_vector *a_alpha,
                       gsl_vector *b_alpha,
                       gsl_vector *mu_beta0,
                       double sigSq_beta0,
                       double a_beta0,
                       double b_beta0,
                       gsl_vector *mu_alpha0,
                       double sigSq_alpha0,
                       double a_alpha0,
                       double b_alpha0,
                       gsl_vector *m_vec,
                       gsl_vector *mu_m,
                       double v_m,
                       double *logPost_val)
{
    int i, j, k;
    double xalpha, xbeta, tempA, tempB, val;
    double loglh, logprior;
    
    int n = V -> size1;
    int q = V -> size2;
    int p1_all = B -> size1;
    int p0_all = A -> size1;
    int q_bin = A -> size2;
    
    gsl_vector *Wrow = gsl_vector_calloc(q_bin);
    gsl_vector *hi = gsl_vector_calloc(q_bin);
    gsl_vector *zero_q = gsl_vector_calloc(q);
    gsl_vector *Vrow = gsl_vector_calloc(q);
    
    loglh = 0;
    logprior = 0;
    
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            if(gsl_matrix_get(W, i, j) >= 0)
            {
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_vector_view Acol = gsl_matrix_column(A, j);
                gsl_blas_ddot(&X0row.vector, &Acol.vector, &xalpha);
                
                gsl_vector_view X1row = gsl_matrix_row(X1, i);
                gsl_vector_view Bcol = gsl_matrix_column(B, j);
                gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                
                tempA = xalpha + gsl_vector_get(alpha0, j);
                tempB = xbeta + gsl_vector_get(beta0, j) + gsl_matrix_get(V, i, j)+log(gsl_vector_get(xi, i)) - pnorm(tempA, 0, 1, 1, 1);;
                loglh += tempB*gsl_matrix_get(Y, i, j) - exp(tempB);
            }
        }
        
        if(q > q_bin)
        {
            for(j = q_bin; j < q; j++)
            {
                gsl_vector_view X1row = gsl_matrix_row(X1, i);
                gsl_vector_view Bcol = gsl_matrix_column(B, j);
                gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                
                tempB = xbeta + gsl_vector_get(beta0, j) + gsl_matrix_get(V, i, j)+log(gsl_vector_get(xi, i));
                loglh += tempB*gsl_matrix_get(Y, i, j) - exp(tempB);
            }
        }
        
        gsl_matrix_get_row(Wrow, W, i);
        gsl_vector_memcpy(hi, alpha0);
        gsl_vector_view X0row = gsl_matrix_row(X0, i);
        gsl_blas_dgemv(CblasTrans, 1, A, &X0row.vector, 1, hi);
        c_dmvnorm2_FA(Wrow, hi, 1, invR, m_vec, &val);
        loglh += val;
        
        gsl_matrix_get_row(Vrow, V, i);
        c_ldmvn_noDet(Vrow, zero_q, 1, invSigmaV, &val);
        logprior += val;
    }
    
    for(j = 0; j < q; j++)
    {
        for(k = 0; k < p1_all; k++)
        {
            if(gsl_matrix_get(gamma_beta, k, j) ==1)
            {
                logprior += dnorm(gsl_matrix_get(B, k, j), 0, sqrt(gsl_vector_get(sigSq_beta, k))*gsl_vector_get(v_beta, j), 1);
                if(k <= p1_all - *p_adj-1)
                {
                    logprior += log(gsl_vector_get(omega_beta, k));
                }
            }else
            {
                if(k <= p1_all - *p_adj-1)
                {
                    logprior += log(1-gsl_vector_get(omega_beta, k));
                }
            }
        }
        
        logprior += dnorm(gsl_vector_get(beta0, j), gsl_vector_get(mu_beta0, j), sqrt(sigSq_beta0), 1);
    }
    
    for(j = 0; j < q_bin; j++)
    {
        for(k = 0; k < p0_all; k++)
        {
            if(gsl_matrix_get(gamma_alpha, k, j) ==1)
            {
                logprior += dnorm(gsl_matrix_get(A, k, j), 0, sqrt(gsl_vector_get(sigSq_alpha, k))*gsl_vector_get(v_alpha, j), 1);
                if(k <= p0_all - *p_adj-1)
                {
                    logprior += log(gsl_vector_get(omega_alpha, k));
                }
            }else
            {
                if(k <= p0_all - *p_adj-1)
                {
                    logprior += log(1-gsl_vector_get(omega_alpha, k));
                }
            }
        }
        
        logprior += dnorm(gsl_vector_get(alpha0, j), gsl_vector_get(mu_alpha0, j), sqrt(sigSq_alpha0), 1);
    }
    
    for(j = 0; j < q_bin; j++)
    {
        logprior += dnorm(gsl_vector_get(m_vec, j), gsl_vector_get(mu_m, j), sqrt(v_m), 1);
    }
    
    for(k = 0; k < p1_all; k++)
    {
        logprior += dgamma(pow(gsl_vector_get(sigSq_beta, k), -1), gsl_vector_get(a_beta, k), pow(gsl_vector_get(b_beta, k),-1), 1) - 2 * log(gsl_vector_get(sigSq_beta, k));
    }
    
    for(k = 0; k < p0_all; k++)
    {
        logprior += dgamma(pow(gsl_vector_get(sigSq_alpha, k), -1), gsl_vector_get(a_alpha, k), pow(gsl_vector_get(b_alpha, k),-1), 1) - 2 * log(gsl_vector_get(sigSq_alpha, k));
    }
    
    logprior += dgamma(pow(sigSq_beta0, -1), a_beta0, pow(b_beta0,-1), 1) - 2 * log(sigSq_beta0);
    logprior += dgamma(pow(sigSq_alpha0, -1), a_alpha0, pow(b_alpha0,-1), 1) - 2 * log(sigSq_alpha0);
    
    *logPost_val = loglh + logprior;
    
    gsl_vector_free(Wrow);
    gsl_vector_free(hi);
    gsl_vector_free(zero_q);
    gsl_vector_free(Vrow);
    
    return;
    
}







/* Updating sigSq_alpha */

void update_sigSq_alpha_mmzip(gsl_matrix *A,
                              gsl_matrix *gamma_alpha,
                              gsl_vector *sigSq_alpha,
                              gsl_vector *v_alpha,
                              gsl_vector *a_alpha,
                              gsl_vector *b_alpha)
{
    int j, kk;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int p0 = A -> size1;
    int q = A -> size2; /* q = q_bin here */
    int numNZ = 0;
    
    kk = (int) runif(0, p0);
    
    for(j = 0; j < q; j++)
    {
        if(gsl_matrix_get(gamma_alpha, kk, j) == 1)
        {
            numNZ += 1;
        }
    }
    
    gsl_vector *alpha = gsl_vector_calloc(q);
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(alpha, j, gsl_matrix_get(A, kk, j));
    }
    
    zeta_shape = gsl_vector_get(a_alpha, kk) + (double) numNZ*0.5;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(alpha, j), 2)/pow(gsl_vector_get(v_alpha, j),2);
    }
    zeta_rate /= 2;
    zeta_rate += gsl_vector_get(b_alpha, kk);
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    gsl_vector_set(sigSq_alpha, kk, pow(zeta, -1));
    
    gsl_vector_free(alpha);
    
    return;
}



/* Updating sigSq_beta */

void update_sigSq_beta_mmzip(gsl_matrix *B,
                             gsl_matrix *gamma_beta,
                             gsl_vector *sigSq_beta,
                             gsl_vector *v_beta,
                             gsl_vector *a_beta,
                             gsl_vector *b_beta)
{
    int j, kk;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int p1 = B -> size1;
    int q = B -> size2;
    int numNZ = 0;
    
    kk = (int) runif(0, p1);
    
    for(j = 0; j < q; j++)
    {
        if(gsl_matrix_get(gamma_beta, kk, j) == 1)
        {
            numNZ += 1;
        }
    }
    
    gsl_vector *beta = gsl_vector_calloc(q);
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(beta, j, gsl_matrix_get(B, kk, j));
    }
    
    zeta_shape = gsl_vector_get(a_beta, kk) + (double) numNZ*0.5;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        if(gsl_matrix_get(gamma_beta, kk, j) == 1)
        {
            zeta_rate += pow(gsl_vector_get(beta, j), 2)/pow(gsl_vector_get(v_beta, j),2);
        }
    }
    zeta_rate /= 2;
    zeta_rate += gsl_vector_get(b_beta, kk);
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    gsl_vector_set(sigSq_beta, kk, pow(zeta, -1));
    
    gsl_vector_free(beta);
    
    return;
}





/* Updating sigSq_alpha0 */

void update_sigSq_alpha0_mmzip(gsl_vector *alpha0,
                               double *sigSq_alpha0,
                               gsl_vector *mu_alpha0,
                               double a_alpha0,
                               double b_alpha0)
{
    int j;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int q = alpha0 -> size; /* q = q_bin here */
    
    zeta_shape = a_alpha0 + (double) q*0.5;
    
    zeta_rate = 0;
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(alpha0, j) - gsl_vector_get(mu_alpha0, j), 2);
    }
    zeta_rate /= 2;
    zeta_rate += b_alpha0;
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    *sigSq_alpha0 = pow(zeta, -1);
    
    return;
}



/* Updating sigSq_beta0 */

void update_sigSq_beta0_mmzip(gsl_vector *beta0,
                              double *sigSq_beta0,
                              gsl_vector *mu_beta0,
                              double a_beta0,
                              double b_beta0)
{
    int j;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int q = beta0 -> size;
    
    zeta_shape = a_beta0 + (double) q*0.5;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(beta0, j) - gsl_vector_get(mu_beta0, j), 2);
    }
    zeta_rate /= 2;
    zeta_rate += b_beta0;
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    *sigSq_beta0 = pow(zeta, -1);
    
    return;
}









/* updating m */

void update_m_mmzip(gsl_matrix *X0,
                    gsl_vector *alpha0,
                    gsl_matrix *A,
                    gsl_matrix *W,
                    gsl_vector *m_vec,
                    gsl_matrix *invR,
                    gsl_vector *mu_m,
                    double v_m,
                    int *accept_m,
                    gsl_vector *accept_m100,
                    double *eps_m,
                    int L_m,
                    int *n_m,
                    int *numReps,
                    double *burninPerc,
                    int M,
                    gsl_vector *M_m,
                    double PtuneEps)
{
    int i, j, l, u;
    double mm, mm_prop;
    double U_star, U_prop, K_star, K_prop, logR, sumAccept, temp, val;
    int accept_ind;
    
    int n = W -> size1;
    int q = W -> size2; /* q = q_bin here */
    
    gsl_vector *m_ini = gsl_vector_calloc(q);
    gsl_vector *m_star = gsl_vector_calloc(q);
    gsl_vector *m_prop = gsl_vector_calloc(q);
    gsl_vector *p_ini = gsl_vector_calloc(q);
    gsl_vector *p_star = gsl_vector_calloc(q);
    gsl_vector *p_prop = gsl_vector_calloc(q);
    gsl_vector *Delta_star = gsl_vector_calloc(q);
    gsl_vector *Delta_prop = gsl_vector_calloc(q);
    gsl_vector *hi = gsl_vector_calloc(q);
    gsl_vector *Wrow = gsl_vector_calloc(q);
    gsl_vector *mSqmh = gsl_vector_calloc(n);
    gsl_vector *mSqmh_prop = gsl_vector_calloc(n);
    gsl_vector *sum_hjSq = gsl_vector_calloc(q);
    gsl_matrix *h_mat = gsl_matrix_calloc(n, q);
    gsl_vector *zero_q = gsl_vector_calloc(q);
    gsl_vector *temp_vec = gsl_vector_calloc(q);
    
    accept_ind = 0;
    *n_m += 1;
    U_star = 0;
    U_prop = 0;
    
    for(i = 0; i < n; i++)
    {
        gsl_matrix_get_row(Wrow, W, i);
        gsl_vector_memcpy(hi, Wrow);
        gsl_vector_sub(hi, alpha0);
        gsl_vector_view X0row = gsl_matrix_row(X0, i);
        gsl_blas_dgemv(CblasTrans, -1, A, &X0row.vector, 1, hi);
        for(j = 0; j < q; j++)
        {
            gsl_matrix_set(h_mat, i, j, gsl_vector_get(hi, j));
        }
    }
    
    for(j = 0; j < q; j++)
    {
        val = 0;
        for(i = 0; i < n; i++)
        {
            val += pow(gsl_matrix_get(h_mat, i, j), 2);
        }
        gsl_vector_set(sum_hjSq, j, val);
    }
    
    gsl_vector_memcpy(m_ini, m_vec);
    gsl_vector_memcpy(m_star, m_vec);
    
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(p_star, j, rnorm(0, sqrt(gsl_vector_get(M_m, j))));
    }
    
    gsl_blas_ddot(m_vec, m_vec, &mm);
    
    U_star -= -0.5*n*log(1+mm);
    
    for(j = 0; j < q; j++)
    {
        U_star -= 0.5*n*log(pow(gsl_vector_get(m_vec, j), 2)+1);
    }
    
    for(j = 0; j < q; j++)
    {
        val = 0;
        for(i = 0; i < n; i++)
        {
            val += pow(gsl_matrix_get(h_mat, i, j), 2);
        }
        U_star -= -0.5*(pow(gsl_vector_get(m_vec, j), 2)+1)*val;
    }
    
    for(i = 0; i < n; i++)
    {
        val = 0;
        for(j = 0; j < q; j++)
        {
            val += gsl_vector_get(m_vec, j) * pow(pow(gsl_vector_get(m_vec, j),2)+1, 0.5) * gsl_matrix_get(h_mat, i, j);
        }
        gsl_vector_set(mSqmh, i, val);
    }
    
    temp = 0;
    for(i = 0; i < n; i++)
    {
        temp += pow(gsl_vector_get(mSqmh, i), 2);
    }
    U_star -= 0.5 * pow(1+mm, -1) * temp;
    
    gsl_blas_ddot(m_vec, mu_m, &val);
    U_star -= -0.5*pow(v_m, -1)*(mm-2*val);
    
    gsl_vector_memcpy(Delta_star, m_vec);
    temp = -n*pow(1+mm, -1);
    gsl_vector_scale(Delta_star, temp);
    
    for(j = 0; j < q; j++)
    {
        val = n * gsl_vector_get(m_vec, j) / (pow(gsl_vector_get(m_vec, j), 2)+1) - gsl_vector_get(m_vec, j) * gsl_vector_get(sum_hjSq, j);
        gsl_vector_set(Delta_star, j, gsl_vector_get(Delta_star, j) + val);
        
        temp = 0;
        for(i = 0; i < n; i++)
        {
            temp += pow(gsl_vector_get(mSqmh, i), 2);
        }
        temp *= -pow(1+mm, -2) * gsl_vector_get(m_vec, j);
        gsl_vector_set(Delta_star, j, gsl_vector_get(Delta_star, j) + temp);
        
        temp = 0;
        for(i = 0; i < n; i++)
        {
            temp += gsl_vector_get(mSqmh, i)* gsl_matrix_get(h_mat, i, j);
        }
        temp *= pow(1+mm, -1) * (1+2*pow(gsl_vector_get(m_vec, j),2)) * pow(1 + pow(gsl_vector_get(m_vec, j), 2), -0.5) ;
        temp += -pow(v_m, -1) * (gsl_vector_get(m_vec, j) - gsl_vector_get(mu_m, j));
        gsl_vector_set(Delta_star, j, gsl_vector_get(Delta_star, j) + temp);
    }
    
    
    gsl_vector_memcpy(p_ini, Delta_star);
    temp = 0.5 * *eps_m;
    gsl_vector_scale(p_ini, temp);
    gsl_vector_add(p_ini, p_star);
    
    for(l = 1; l <= L_m; l++)
    {
        gsl_vector_memcpy(m_prop, p_ini);
        gsl_vector_scale(m_prop, *eps_m);
        gsl_vector_div(m_prop, M_m);
        gsl_vector_add(m_prop, m_ini);
        
        gsl_vector_set_zero(Delta_prop);
        
        gsl_blas_ddot(m_prop, m_prop, &mm_prop);
        
        for(i = 0; i < n; i++)
        {
            val = 0;
            for(j = 0; j < q; j++)
            {
                val += gsl_vector_get(m_prop, j) * pow(pow(gsl_vector_get(m_prop, j),2)+1, 0.5) * gsl_matrix_get(h_mat, i, j);
            }
            gsl_vector_set(mSqmh_prop, i, val);
        }
        
        gsl_vector_memcpy(Delta_prop, m_prop);
        temp = -n*pow(1+mm_prop, -1);
        gsl_vector_scale(Delta_prop, temp);
        
        for(j = 0; j < q; j++)
        {
            val = n * gsl_vector_get(m_prop, j) / (pow(gsl_vector_get(m_prop, j), 2)+1) - gsl_vector_get(m_prop, j) * gsl_vector_get(sum_hjSq, j);
            gsl_vector_set(Delta_prop, j, gsl_vector_get(Delta_prop, j) + val);
            
            temp = 0;
            for(i = 0; i < n; i++)
            {
                temp += pow(gsl_vector_get(mSqmh_prop, i), 2);
            }
            temp *= -pow(1+mm_prop, -2) * gsl_vector_get(m_prop, j);
            gsl_vector_set(Delta_prop, j, gsl_vector_get(Delta_prop, j) + temp);
            
            temp = 0;
            for(i = 0; i < n; i++)
            {
                temp += gsl_vector_get(mSqmh_prop, i)* gsl_matrix_get(h_mat, i, j);
            }
            temp *= pow(1+mm_prop, -1) * (1+2*pow(gsl_vector_get(m_prop, j), 2))  * pow(1 + pow(gsl_vector_get(m_prop, j), 2), -0.5);
            temp += -pow(v_m, -1) * (gsl_vector_get(m_prop, j) - gsl_vector_get(mu_m, j));
            gsl_vector_set(Delta_prop, j, gsl_vector_get(Delta_prop, j) + temp);
        }
        
        if(l < L_m)
        {
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, (double) *eps_m);
            gsl_vector_add(p_prop, p_ini);
            
            gsl_vector_memcpy(m_ini, m_prop);
            gsl_vector_memcpy(p_ini, p_prop);
        }else if(l == L_m)
        {
            U_prop -= -0.5*n*log(1+mm_prop);
            
            for(j = 0; j < q; j++)
            {
                U_prop -= 0.5*n*log(pow(gsl_vector_get(m_prop, j), 2)+1);
            }
            
            for(j = 0; j < q; j++)
            {
                val = 0;
                for(i = 0; i < n; i++)
                {
                    val += pow(gsl_matrix_get(h_mat, i, j), 2);
                }
                U_prop -= -0.5*(pow(gsl_vector_get(m_prop, j), 2)+1)*val;
            }
            
            temp = 0;
            for(i = 0; i < n; i++)
            {
                temp += pow(gsl_vector_get(mSqmh_prop, i), 2);
            }
            U_prop -= 0.5 * pow(1+mm_prop, -1) * temp;
            
            gsl_blas_ddot(m_prop, mu_m, &val);
            U_prop -= -0.5*pow(v_m, -1)*(mm_prop-2*val);
            
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, (double) 0.5* *eps_m);
            gsl_vector_add(p_prop, p_ini);
        }
    }
    
    K_star = 0;
    K_prop = 0;
    for(j = 0; j < q; j++)
    {
        K_star += pow(gsl_vector_get(p_star, j), 2)* pow(gsl_vector_get(M_m, j), -1) * 0.5;
        K_prop += pow(gsl_vector_get(p_prop, j), 2)* pow(gsl_vector_get(M_m, j), -1) * 0.5;
    }
    
    logR = -(U_prop + K_prop) + U_star + K_star;
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        gsl_vector_memcpy(m_vec, m_prop);
        *accept_m += 1;
        
        c_solve_corFA1(m_vec, invR);
    }
    
    if(M <= *numReps* 0.5)
    {
        if(*n_m <= 100)
        {
            accept_ind = *n_m-1;
            gsl_vector_set(accept_m100, accept_ind, u);
        }else if(*n_m > 100)
        {
            accept_ind = (int) *n_m % 10 + 90 - 1;
            gsl_vector_set(accept_m100, accept_ind, u);
        }
        
        if((int) *n_m % 10 == 0 && (int) *n_m >= 100)
        {
            sumAccept = 0;
            for(i = 0; i < 99; i++)
            {
                sumAccept += gsl_vector_get(accept_m100, i);
            }
            
            if(sumAccept / 100 < 0.001)
            {
                *eps_m *= 0.1;
            }else if(sumAccept / 100 < 0.05)
            {
                *eps_m *= 0.5;
            }else if(sumAccept / 100 < 0.20)
            {
                *eps_m *= 0.7;
            }else if(sumAccept / 100 < 0.60)
            {
                *eps_m *= 0.9;
            }else if(sumAccept / 100 > 0.70)
            {
                *eps_m *= 1.1;
            }else if(sumAccept / 100 > 0.80)
            {
                *eps_m *= 2;
            }else if(sumAccept / 100 > 0.95)
            {
                *eps_m *= 10;
            }
            
            accept_ind = 90;
            
            for(i = 0; i < 90; i++)
            {
                gsl_vector_set(accept_m100, i, gsl_vector_get(accept_m100, i+10));
            }
            for(i = 90; i < 99; i++)
            {
                gsl_vector_set(accept_m100, i, 0);
            }
        }
    }
    

    
    gsl_vector_free(m_ini);
    gsl_vector_free(m_star);
    gsl_vector_free(m_prop);
    gsl_vector_free(p_ini);
    gsl_vector_free(p_star);
    gsl_vector_free(p_prop);
    gsl_vector_free(Delta_star);
    gsl_vector_free(Delta_prop);
    gsl_vector_free(hi);
    gsl_vector_free(Wrow);
    gsl_vector_free(mSqmh);
    gsl_vector_free(mSqmh_prop);
    gsl_vector_free(sum_hjSq);
    gsl_matrix_free(h_mat);
    gsl_vector_free(zero_q);
    gsl_vector_free(temp_vec);
    
    return;
}









/* updating W */

void update_W_mmzip(gsl_matrix *Y,
                    gsl_matrix *X1,
                    gsl_matrix *X0,
                    gsl_vector *xi,
                    gsl_vector *beta0,
                    gsl_matrix *B,
                    gsl_matrix *V,
                    gsl_vector *alpha0,
                    gsl_matrix *A,
                    gsl_matrix *W,
                    gsl_vector *m_vec,
                    gsl_matrix *invR,
                    double pU_W)
{
    int i, k, jj;
    double forVar, mean, sd, sample, lprob, tempA, tempB, eta, cumNorm, lcumNorm, lsel;
    
    int n = Y -> size1;
    int q_bin = A -> size2;
    
    
    gsl_vector *m_subvec =gsl_vector_calloc(q_bin-1);
    
    gsl_vector *R_subvec =gsl_vector_calloc(q_bin-1);
    gsl_matrix *invSub =gsl_matrix_calloc(q_bin-1, q_bin-1);
    gsl_vector *transR_invSub =gsl_vector_calloc(q_bin-1);
    
    gsl_vector *w_alp0_Az =gsl_vector_calloc(q_bin-1);
    
    gsl_matrix *R_temp =gsl_matrix_calloc(q_bin, q_bin);
    gsl_matrix *covR_temp =gsl_matrix_calloc(q_bin, q_bin);
    
    cov_FA1(m_vec, covR_temp);
    c_cov2cor(covR_temp, R_temp);
    
    for(jj = 0; jj < q_bin; jj++)
    {
        if(runif(0, 1) < pU_W)
        {
            Get_subVector(m_vec, jj, m_subvec);
            c_solve_corFA1(m_subvec, invSub);
            
            Get_subColumnVector(R_temp, jj, R_subvec);
            
            gsl_blas_dgemv(CblasNoTrans, 1, invSub, R_subvec, 0, transR_invSub);
            gsl_blas_ddot(transR_invSub, R_subvec, &forVar);
            
            for(i = 0; i < n; i++)
            {
                gsl_vector_view X0_row = gsl_matrix_row(X0, i);
                gsl_vector_view X1_row = gsl_matrix_row(X1, i);
                
                if(forVar < 1)
                {
                    sd = pow((1 - forVar), 0.5);
                }else
                {
                    sd = sqrt(0.000001);
                }
                
                
                for(k=0; k < q_bin-1; k++)
                {
                    if(k < jj)
                    {
                        gsl_vector_view A_col = gsl_matrix_column(A, k);
                        gsl_blas_ddot(&X0_row.vector, &A_col.vector, &tempA);
                        gsl_vector_set(w_alp0_Az, k, gsl_matrix_get(W, i, k) - (gsl_vector_get(alpha0, k)+tempA));
                    }else
                    {
                        gsl_vector_view A_col = gsl_matrix_column(A, k+1);
                        gsl_blas_ddot(&X0_row.vector, &A_col.vector, &tempA);
                        gsl_vector_set(w_alp0_Az, k, gsl_matrix_get(W, i, k+1) - (gsl_vector_get(alpha0, k+1)+tempA));
                    }
                }
                
                gsl_vector_view A_col = gsl_matrix_column(A, jj);
                gsl_blas_ddot(&X0_row.vector, &A_col.vector, &tempA);
                
                gsl_blas_ddot(w_alp0_Az, transR_invSub, &mean);
                mean += gsl_vector_get(alpha0, jj) + tempA;
                
                if(gsl_matrix_get(Y, i, jj) > 0)
                {
                    c_rtnorm(mean, sd, 0, 100000, 0, 1, &sample);
                }else if(gsl_matrix_get(Y, i, jj) == 0)
                {
                    gsl_vector_view B_col = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1_row.vector, &B_col.vector, &tempB);
                    eta = gsl_vector_get(beta0, jj) + tempB + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i))-pnorm(gsl_vector_get(alpha0, jj) + tempA, 0, 1, 1, 1);
                    cumNorm = pnorm(0, mean, sd, 1, 0);
                    lcumNorm = pnorm(0, mean, sd, 0, 1);
                    
                    lprob = -exp(eta)+ lcumNorm - log(exp(-exp(eta))*(1-cumNorm) + cumNorm);
                    
                    lsel = log(runif(0, 1));
                    if(lsel <= lprob)
                    {
                        c_rtnorm(mean, sd, 0, 100000, 0, 1, &sample);
                    }else
                    {
                        c_rtnorm(mean, sd, -100000, 0, 1, 0, &sample);
                    }
                }
                gsl_matrix_set(W, i, jj, sample);
                
                gsl_vector_set_zero(w_alp0_Az);
            }
            gsl_matrix_set_zero(invSub);
        }
    }
    
    
    gsl_matrix_free(invSub);
    gsl_matrix_free(R_temp);
    gsl_matrix_free(covR_temp);
    
    gsl_vector_free(m_subvec);
    gsl_vector_free(R_subvec);
    gsl_vector_free(transR_invSub);
    gsl_vector_free(w_alp0_Az);
    
    return;
}








