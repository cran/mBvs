

#include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_heapsort.h"

#include "R.h"
#include "Rmath.h"

#include "MBVSfa.h"



/* updating regression parameter: B, gamma */

/**/

void updateRPfa(int *q_adj,
              gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_matrix *gamma,
              gsl_matrix *updateNonzB,
              gsl_vector *beta0,
              gsl_vector *lambda,
              double sigmaSq,
              gsl_matrix *invSigmaLam,
              gsl_vector *mu0,
              gsl_vector *v,
              gsl_vector *omega,
              double h0,
              double eta,
              gsl_matrix *rwbetaVar,
              gsl_matrix *accept_B)
{
    
    double beta_prop, res, res_prop, expTerm, expTerm1, expTerm2;
    double logPrior1, logPrior1_prop, logPrior2, logPrior2_prop;
    double logProp, logProp_new, logPrior2_1, logPrior2_1prop;
    double logPrior2_2, logPrior2_2prop, logR, choice, sumG;
    double logLH;
    double logLH_prop;
    double sumGam = 0;
    double p_add, p_del;
    
    int u, i, j, k, l, m, ii, kk, lInx, count, count_rev, move, putInx;
    int refine_q, refine_p;
    
    int p = B -> size2;
    int q = B -> size1;    
    int n = Y -> size1;

    p_add = (double) 1/3;
    p_del = (double) 1/3;
    
    gsl_matrix *gamma_prop = gsl_matrix_calloc(q, p);
    gsl_matrix *B_prop = gsl_matrix_calloc(q, p);
    
    gsl_vector *Yrow = gsl_vector_calloc(p);
    gsl_vector *meanY = gsl_vector_calloc(p);
    gsl_vector *meanY_prop = gsl_vector_calloc(p);

    for(j = 0; j < p; j++)
    {
        
        logLH = 0;
        logLH_prop = 0;
        
        
        gsl_matrix_memcpy(gamma_prop, gamma);
        gsl_matrix_memcpy(B_prop, B);
        
        sumGam = 0;
        for(i = 0; i < q - *q_adj; i++)
        {
            sumGam += gsl_matrix_get(gamma, i, j);
        }
    
    
    /* selecting a move */
    /* move: 1=add, 2=delete, 3=swap */
    
    if((q - *q_adj) == 1)
    {
        if(gsl_matrix_get(gamma, 0, j) == 1) move = 2;
        if(gsl_matrix_get(gamma, 0, j) == 0) move = 1;
    }
    if((q - *q_adj) > 1)
    {
        if(sumGam == (q - *q_adj)) move = 2;
        if(sumGam == 0) move = 1;
        if(sumGam != (q - *q_adj) && sumGam != 0)
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
        if((q - *q_adj) == 1) l = 0;
        else
        {
            choice  = runif(0, 1);
            
            lInx = (int) (choice * ((double) q - (double) *q_adj - (double) sumGam)) + 1;
            count = 0;
            k = -1;
            while(lInx != count)
            {
                k += 1;
                if(gsl_matrix_get(gamma, k, j) == 0)
                {
                    count += 1;
                }
            }
            l = k;
        }
        
        gsl_matrix_set(gamma_prop, l, j, 1);
        
        beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)));
        
        gsl_matrix_set(B_prop, l, j, beta_prop);
        
        for(i = 0; i < n; i++)
        {
            gsl_matrix_get_row(Yrow, Y, i);
            gsl_vector_memcpy(meanY, beta0);
            gsl_vector_memcpy(meanY_prop, beta0);
            gsl_vector_view Xrow = gsl_matrix_row(X, i);
            gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
            gsl_blas_dgemv(CblasTrans, 1, B_prop, &Xrow.vector, 1, meanY_prop);
            
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
            c_dmvnorm(Yrow, meanY_prop, sqrt(sigmaSq), invSigmaLam, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        logPrior1_prop = dnorm(beta_prop, 0, sqrt(sigmaSq) * gsl_vector_get(v, j), 1);
        logPrior1 = 0;
        
        expTerm = gsl_vector_get(omega, l) + eta * sumCorfa_j(lambda, gamma, j, l);
        
        logPrior2 = log(one_invLogit(expTerm));
        logPrior2_prop = log(invLogit(expTerm));
        
        logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)), 1);
        logProp = log(q - *q_adj-sumGam) - log(sumGam + 1);
        
        
    } /* end of add move */
    
    
    
    /* for delete move */
    
    if(move == 2)
    {
        if((q - *q_adj) == 1) l = 0;
        else
        {
            choice  = runif(0, 1);
            
            lInx = (int) (choice * ((double) sumGam)) + 1;
            count = 0;
            k = -1;
            while(lInx != count)
            {
                k += 1;
                if(gsl_matrix_get(gamma, k, j) == 1)
                {
                    count += 1;
                }
            }
            l = k;
        }
        
        gsl_matrix_set(gamma_prop, l, j, 0);
        
        beta_prop = 0;
        gsl_matrix_set(B_prop, l, j, beta_prop);
        
        for(i = 0; i < n; i++)
        {
            gsl_matrix_get_row(Yrow, Y, i);
            gsl_vector_memcpy(meanY, beta0);
            gsl_vector_memcpy(meanY_prop, beta0);
            gsl_vector_view Xrow = gsl_matrix_row(X, i);
            gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
            gsl_blas_dgemv(CblasTrans, 1, B_prop, &Xrow.vector, 1, meanY_prop);
            
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
            c_dmvnorm(Yrow, meanY_prop, sqrt(sigmaSq), invSigmaLam, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        logPrior1_prop = 0;
        logPrior1 = dnorm(gsl_matrix_get(B, l, j), 0, sqrt(sigmaSq) * gsl_vector_get(v, j), 1);
        
        expTerm = gsl_vector_get(omega, l) + eta * sumCorfa_j(lambda, gamma, j, l);
        
        logPrior2 = log(invLogit(expTerm));
        logPrior2_prop = log(one_invLogit(expTerm));
        
        logProp_new = 0;
        logProp = dnorm(gsl_matrix_get(B, l, j), gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)), 1)+log(sumGam) - log(q - *q_adj-sumGam+1);
        
    } /* end of delete move */
    
    
    /* for swap move */
    
    if(move == 3)
    {
        choice  = runif(0, 1);
        
        lInx = (int) (choice * ((double) q - (double) *q_adj - (double) sumGam)) + 1;
        
        count = 0;
        k = -1;
        while(lInx != count)
        {
            k += 1;
            if(gsl_matrix_get(gamma, k, j) == 0)
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
            if(gsl_matrix_get(gamma, k, j) == 1)
            {
                count += 1;
            }
        }
        m = k;
        
        gsl_matrix_set(gamma_prop, l, j, 1);
        gsl_matrix_set(gamma_prop, m, j, 0);
        
        beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)));
        
        gsl_matrix_set(B_prop, l, j, beta_prop);
        gsl_matrix_set(B_prop, m, j, 0);
        
        for(i = 0; i < n; i++)
        {
            gsl_matrix_get_row(Yrow, Y, i);
            gsl_vector_memcpy(meanY, beta0);
            gsl_vector_memcpy(meanY_prop, beta0);
            gsl_vector_view Xrow = gsl_matrix_row(X, i);
            gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
            gsl_blas_dgemv(CblasTrans, 1, B_prop, &Xrow.vector, 1, meanY_prop);
            
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
            c_dmvnorm(Yrow, meanY_prop, sqrt(sigmaSq), invSigmaLam, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        logPrior1_prop = dnorm(beta_prop, 0, sqrt(sigmaSq) * gsl_vector_get(v, j), 1);
        logPrior1 = dnorm(gsl_matrix_get(B, m, j), 0, sqrt(sigmaSq) * gsl_vector_get(v, j), 1);
        
        expTerm1 = gsl_vector_get(omega, l) + eta * sumCorfa_j(lambda, gamma, j, l);
        
        logPrior2_1 = log(one_invLogit(expTerm1));
        logPrior2_1prop = log(invLogit(expTerm1));
        
        expTerm2 = gsl_vector_get(omega, m) + eta * sumCorfa_j(lambda, gamma, j, m);
        
        logPrior2_2 = log(invLogit(expTerm2));
        logPrior2_2prop = log(one_invLogit(expTerm2));
        
        logPrior2_prop = logPrior2_1prop + logPrior2_2prop;
        logPrior2 = logPrior2_1 + logPrior2_2;
        
        logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)), 1);
        logProp = dnorm(gsl_matrix_get(B, m, j), gsl_matrix_get(updateNonzB, m, j), sqrt(gsl_matrix_get(rwbetaVar, m, j)), 1);
 
        
    } /* end of swap move */
    
    /* acceptance probability */
    
    logR = logLH_prop - logLH + logPrior1_prop - logPrior1 + logPrior2_prop - logPrior2 + logProp - logProp_new;

    u = log(runif(0, 1)) < logR;
        
        if(u == 1)
        {
            gsl_matrix_swap(gamma, gamma_prop);
            
            if(move == 1 || move == 3)
            {
                gsl_matrix_set(updateNonzB, l, j, beta_prop);
            }
            if(move == 1 || move == 2)
            {
                gsl_matrix_set(B, l, j, beta_prop);
                gsl_matrix_set(accept_B, l, j, (gsl_matrix_get(accept_B, l, j) + u));
            }
            if(move == 3)
            {
                gsl_matrix_set(B, l, j, beta_prop);
                gsl_matrix_set(B, m, j, 0);
                gsl_matrix_set(accept_B, l, j, (gsl_matrix_get(accept_B, l, j) + u));
                gsl_matrix_set(accept_B, m, j, (gsl_matrix_get(accept_B, m, j) + u));
            }
        }
    }

    /* Refining step for beta using random walk M-H */
    
    gsl_vector *colSumsGam = gsl_vector_calloc(p);
    
    c_colSums(gamma, colSumsGam);
    
    sumG = 0;
    for(j = 0; j < p; j++) sumG += gsl_vector_get(colSumsGam, j);
    
    count = 0;
    for(j = 0; j < p; j++)
    {
        if(gsl_vector_get(colSumsGam, j) > 0) count += 1;
    }
    
    count_rev = count;
    if(count == 0) count_rev = 1;
    
    gsl_vector *colSumsGamNonzInx = gsl_vector_calloc(count_rev);
    putInx = 0;
    
    for(j = 0; j < p; j++)
    {
        if(gsl_vector_get(colSumsGam, j) > 0)
        {
            gsl_vector_set(colSumsGamNonzInx, putInx, j);
            putInx += 1;
        }
    }
    
    /*  */
    refine_p = (int) c_min(5, p);
    refine_q = (int) c_min(100, q);
    
    if(sumG > 0)
    {
        for(ii = 0; ii < refine_p; ii++)
        {
            choice  = (int) runif(0, count_rev);
            j = (int) gsl_vector_get(colSumsGamNonzInx, choice);
     
            for(kk = 0; kk < refine_q; kk ++)
            {
                k  = (int) runif(0, q);
                
                if(gsl_matrix_get(gamma, k, j) ==1)
                {
                    beta_prop = rnorm(gsl_matrix_get(updateNonzB, k, j), sqrt(gsl_matrix_get(rwbetaVar, k, j)));
                    gsl_matrix_memcpy(B_prop, B);
                    gsl_matrix_set(B_prop, k, j, beta_prop);
                    
                    logLH = 0;
                    logLH_prop = 0;
                    
                    for(i = 0; i < n; i++)
                    {
                        gsl_matrix_get_row(Yrow, Y, i);
                        gsl_vector_memcpy(meanY, beta0);
                        gsl_vector_memcpy(meanY_prop, beta0);
                        gsl_vector_view Xrow = gsl_matrix_row(X, i);
                        gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
                        gsl_blas_dgemv(CblasTrans, 1, B_prop, &Xrow.vector, 1, meanY_prop);
                        
                        c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
                        c_dmvnorm(Yrow, meanY_prop, sqrt(sigmaSq), invSigmaLam, &res_prop);
                        
                        logLH += res;
                        logLH_prop += res_prop;
                    }
                    
                    logPrior1_prop = dnorm(beta_prop, 0, gsl_vector_get(v, j) * sqrt(sigmaSq), 1);
                    logPrior1 = dnorm(gsl_matrix_get(B, k, j), 0, gsl_vector_get(v, j) * sqrt(sigmaSq), 1);
                    
                    logProp_new = dnorm(beta_prop, gsl_matrix_get(B, k, j), sqrt(gsl_matrix_get(rwbetaVar, k, j)), 1);
                    logProp = dnorm(gsl_matrix_get(B, k, j), beta_prop, sqrt(gsl_matrix_get(rwbetaVar, k, j)), 1);
                    
                    logR = logLH_prop - logLH + logPrior1_prop - logPrior1 + logProp - logProp_new;
                    
                    u = log(runif(0, 1)) < logR;
                    if(u == 1)
                    {
                        gsl_matrix_set(B, k, j, beta_prop);
                        gsl_matrix_set(updateNonzB, k, j, beta_prop);
                        gsl_matrix_set(accept_B, k, j, (gsl_matrix_get(accept_B, k, j) + u));
                        
                    }
                }
            }
        }
    }
    
    gsl_matrix_free(gamma_prop);
    gsl_matrix_free(B_prop);
    gsl_vector_free(Yrow);
    gsl_vector_free(meanY);    
    gsl_vector_free(meanY_prop);
    gsl_vector_free(colSumsGam);
    gsl_vector_free(colSumsGamNonzInx);
    
    return;
}



/* updating intercept: beta0 */

/**/

void updateIPfa(gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_vector *beta0,
              double sigmaSq,
              gsl_matrix *invSigmaLam,
              gsl_vector *mu0,
              double h0)
{
    int n = Y -> size1;
    int p = Y -> size2;
    int j;
    
    gsl_matrix *varComp1 = gsl_matrix_calloc(p, p);
    gsl_matrix *varComp2 = gsl_matrix_calloc(p, p);
    gsl_matrix *varComp = gsl_matrix_calloc(p, p);
    
    gsl_matrix_memcpy(varComp2, invSigmaLam);
    gsl_matrix_scale(varComp2, (double) n);

    c_solve(varComp2, varComp);
    
    gsl_vector *ones_n = gsl_vector_calloc(n);
    gsl_vector_set_all(ones_n, 1);
    
    gsl_matrix *XB = gsl_matrix_calloc(n, p);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, X, B, 0, XB);
    
    gsl_vector *meanBeta1 = gsl_vector_calloc(p);
    gsl_vector *meanBeta2 = gsl_vector_calloc(p);
    gsl_vector *meanBeta = gsl_vector_calloc(p);
    gsl_blas_dgemv(CblasTrans, 1, Y, ones_n, 0, meanBeta1);
    gsl_blas_dgemv(CblasTrans, -1, XB, ones_n, 1, meanBeta1);
    
    gsl_vector_memcpy(meanBeta2, mu0);
    gsl_blas_dgemv(CblasNoTrans, 1, invSigmaLam, meanBeta1, 1, meanBeta2);
    
    gsl_blas_dgemv(CblasNoTrans, 1, varComp, meanBeta2, 0, meanBeta);
    
    gsl_matrix *beta0spl = gsl_matrix_calloc(1, p);
    gsl_matrix_scale(varComp, sigmaSq);
    
    c_rmvnorm(beta0spl, meanBeta, varComp);
    
    for(j = 0; j < p; j++)
    {
        gsl_vector_set(beta0, j, gsl_matrix_get(beta0spl, 0, j));
    }
    
    gsl_matrix_free(varComp1);
    gsl_matrix_free(varComp2);
    gsl_matrix_free(varComp);
    gsl_matrix_free(XB);
    gsl_matrix_free(beta0spl);
    gsl_vector_free(ones_n);
    gsl_vector_free(meanBeta1);
    gsl_vector_free(meanBeta2);
    gsl_vector_free(meanBeta);
    
    return;
    
}


/* updating lambda */

/**/

void updateCPfa(int *q_adj,
              gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_matrix *gamma,
              gsl_vector *beta0,
              gsl_vector *lambda,
              double sigmaSq,
              gsl_matrix *SigmaLam,
              gsl_matrix *invSigmaLam,
              double hLam,
              double eta,
              gsl_vector *omega,
              gsl_vector *muLam,
              double rwlambdaVar,
              gsl_vector *accept_lambda)
{
    double temp_prop, res, res_prop, expTerm, expTerm_prop;
    double logPriorLam, logPriorLam_prop;
    double logProp, logProp_new;    
    double logR;
    double logLH = 0;
    double logLH_prop = 0;
    double logPriorGam_prop_ini = 0;
    
    int i, j, k, l, u;
    
    int p = beta0 -> size;
    int n = Y -> size1;
    int q = B -> size1;
    
    j = (int) runif(0, p);

    gsl_vector *lambda_prop = gsl_vector_calloc(p);
    gsl_matrix *SigmaLam_prop = gsl_matrix_calloc(p, p);
    gsl_matrix *invSigmaLam_prop = gsl_matrix_calloc(p, p);
    gsl_vector *Yrow = gsl_vector_calloc(p);
    gsl_vector *meanY = gsl_vector_calloc(p);
    gsl_matrix *iden = gsl_matrix_calloc(p, p);    
    
    if(j >= 1)
    {
        gsl_vector_memcpy(lambda_prop, lambda);        
        temp_prop = rnorm(gsl_vector_get(lambda, j), sqrt(rwlambdaVar));
        
        gsl_vector_set(lambda_prop, j, temp_prop);

        gsl_matrix_set_identity(SigmaLam_prop);
        gsl_blas_dger(1, lambda_prop, lambda_prop, SigmaLam_prop);
        
        c_solve(SigmaLam_prop, invSigmaLam_prop);        
        
        for(i = 0; i < n; i++)
        {
            gsl_matrix_get_row(Yrow, Y, i);
            gsl_vector_memcpy(meanY, beta0);
            gsl_vector_view Xrow = gsl_matrix_row(X, i);
            gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
            
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam_prop, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        for(l = 0; l < p; l++)
        {
            for(k = 0; k < (q - *q_adj); k++)
            {
                expTerm = gsl_vector_get(omega, k) + eta * sumCorfa_j(lambda, gamma, l, k);
                expTerm_prop = gsl_vector_get(omega, k) + eta * sumCorfa_j(lambda_prop, gamma, l, k);
                if(gsl_matrix_get(gamma, k, l) == 1)
                {
                    logPriorGam_prop_ini += log(invLogit(expTerm_prop)) - log(invLogit(expTerm));
                }
                if(gsl_matrix_get(gamma, k, l) == 0)
                {
                    logPriorGam_prop_ini += log(one_invLogit(expTerm_prop)) - log(one_invLogit(expTerm));
                }
            }
        }

        gsl_matrix_set_identity(iden);
        
        logPriorLam = dnorm(gsl_vector_get(lambda, j), 0, sqrt(hLam * sigmaSq), 1);
        logPriorLam_prop = dnorm(gsl_vector_get(lambda_prop, j), 0, sqrt(hLam * sigmaSq), 1);
        
        logProp = dnorm(gsl_vector_get(lambda, j), gsl_vector_get(lambda_prop, j), sqrt(rwlambdaVar), 1);
        logProp_new = dnorm(gsl_vector_get(lambda_prop, j), gsl_vector_get(lambda, j), sqrt(rwlambdaVar), 1);
        
        logR = logLH_prop - logLH + logPriorGam_prop_ini + logPriorLam_prop - logPriorLam + logProp - logProp_new;

        u = log(runif(0, 1)) < logR;
        if(u == 1)
        {
            gsl_vector_set(lambda, j, temp_prop);
            gsl_vector_set(accept_lambda, j, (gsl_vector_get(accept_lambda, j) + u));
            
            gsl_matrix_set_identity(SigmaLam);
            gsl_blas_dger(1, lambda, lambda, SigmaLam);
            c_solve(SigmaLam, invSigmaLam);
        }
    }
    if(j == 0)
    {
        gsl_vector_memcpy(lambda_prop, lambda);
        
        temp_prop = rnorm(gsl_vector_get(lambda, j), sqrt(rwlambdaVar));
        
        gsl_vector_set(lambda_prop, j, temp_prop);
        
        gsl_matrix_set_identity(SigmaLam_prop);
        gsl_blas_dger(1, lambda_prop, lambda_prop, SigmaLam_prop);
        
        c_solve(SigmaLam_prop, invSigmaLam_prop);
        
        for(i = 0; i < n; i++)
        {
            gsl_matrix_get_row(Yrow, Y, i);
            gsl_vector_memcpy(meanY, beta0);
            gsl_vector_view Xrow = gsl_matrix_row(X, i);
            gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
            
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam, &res);
            c_dmvnorm(Yrow, meanY, sqrt(sigmaSq), invSigmaLam_prop, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        for(l = 0; l < p; l++)
        {
            for(k = 0; k < (q - *q_adj); k++)
            {
                expTerm = gsl_vector_get(omega, k) + eta * sumCorfa_j(lambda, gamma, l, k);
                expTerm_prop = gsl_vector_get(omega, k) + eta * sumCorfa_j(lambda_prop, gamma, l, k);
                if(gsl_matrix_get(gamma, k, l) == 1)
                {
                    logPriorGam_prop_ini += log(invLogit(expTerm_prop)) - log(invLogit(expTerm));
                }
                if(gsl_matrix_get(gamma, k, l) == 0)
                {
                    logPriorGam_prop_ini += log(one_invLogit(expTerm_prop)) - log(one_invLogit(expTerm));
                }
            }
        }

        gsl_matrix_set_identity(iden);
        
        logPriorLam = dnorm(gsl_vector_get(lambda, j), 0, sqrt(hLam * sigmaSq), 1);
        logPriorLam_prop = dnorm(gsl_vector_get(lambda_prop, j), 0, sqrt(hLam * sigmaSq), 1);
        
        
        logProp = dnorm(gsl_vector_get(lambda, j), gsl_vector_get(lambda_prop, j), sqrt(rwlambdaVar), 1);
        logProp_new = dnorm(gsl_vector_get(lambda_prop, j), gsl_vector_get(lambda, j), sqrt(rwlambdaVar), 1);
        
        logR = logLH_prop - logLH + logPriorGam_prop_ini + logPriorLam_prop - logPriorLam + logProp - logProp_new;

        u = log(runif(0, 1)) < logR;
        if(u == 1)
        {
            gsl_vector_set(lambda, j, temp_prop);
            gsl_vector_set(accept_lambda, j, (gsl_vector_get(accept_lambda, j) + u));
            
            gsl_matrix_set_identity(SigmaLam);
            gsl_blas_dger(1, lambda, lambda, SigmaLam);
            c_solve(SigmaLam, invSigmaLam);
        }
    }

    gsl_matrix_free(SigmaLam_prop);
    gsl_matrix_free(invSigmaLam_prop);
    gsl_matrix_free(iden);
    gsl_vector_free(lambda_prop);
    gsl_vector_free(Yrow);
    gsl_vector_free(meanY);
    
    return;
}


/* updating residual variance : sigmaSq */
/**/

void updateVPfa(gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_vector *beta0,
              gsl_vector *lambda,
              double *sigmaSq,
              gsl_matrix *invSigmaLam,
              double h0,
              double hLam,
              double nu0,
              double sigSq0,
              gsl_vector *v,              
              gsl_vector *mu0,
              gsl_vector *muLam)
{
    int p = beta0 -> size;
    int n = Y -> size1;
    int q = B -> size1;
    int i, j;
    
    double sig_a, sig_b, val;
    double sig_b1 = 0;
    double sig_b3 = 0;

    double sig_b2 = 0;
    double sig_b4 = 0;
    
    sig_a = (double) 1/2*(n*p + p*q + p + nu0);
    
    gsl_vector *meanY = gsl_vector_calloc(p);
    gsl_vector *diff = gsl_vector_calloc(p);
    gsl_vector *temp = gsl_vector_calloc(q);
    
    for(i = 0; i < n; i++)
    {
        gsl_matrix_get_row(diff, Y, i);
        gsl_vector_memcpy(meanY, beta0);
        gsl_vector_view Xrow = gsl_matrix_row(X, i);
        gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
        gsl_vector_sub(diff, meanY);
        c_quadform_vMu(diff, invSigmaLam, diff, &val);
        sig_b1 += val;
    }
    
    for(j = 0; j < p; j++)
    {
        gsl_matrix_get_col(temp, B, j);
        gsl_blas_ddot(temp, temp, &val);
        val /= pow(gsl_vector_get(v, j), 2);
        sig_b3 += val;
    }
    
    gsl_vector_memcpy(diff, lambda);
    gsl_vector_sub(diff, muLam);
    gsl_blas_ddot(diff, diff, &sig_b4);
    sig_b4 /= hLam;

    
    sig_b = (double) 1/2*(sig_b1 + sig_b2 + sig_b3 + sig_b4 + nu0 * sigSq0);
    
    c_rigamma(sigmaSq, sig_a, sig_b);
    
    gsl_vector_free(meanY);
    gsl_vector_free(diff);
    gsl_vector_free(temp);
    
}



