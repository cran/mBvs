

#include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_heapsort.h"

#include "R.h"
#include "Rmath.h"

#include "MBVSus.h"



/* updating regression parameter: B, gamma */

/**/
void updateRPus(int *q_adj,
              gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_matrix *gamma,
              gsl_matrix *Sigma,
              gsl_matrix *invSigma,
              gsl_matrix *updateNonzB,
              gsl_vector *beta0,
              gsl_vector *v,
              gsl_vector *omega,
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
    double sumGam;
    double p_add, p_del, p_swap;
    
    int u, i, j, k, l, m, ii, kk, lInx, count, count_rev, move, putInx;
    int refine_q, refine_p;
    
    int p = B -> size2;
    int q = B -> size1;    
    int n = Y -> size1;

    p_add = (double) 1/3;
    p_del = (double) 1/3;
    p_swap = 1-(p_add+p_del);
    
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
            
            c_dmvnorm(Yrow, meanY, 1, invSigma, &res);
            c_dmvnorm(Yrow, meanY_prop, 1, invSigma, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        logPrior1_prop = dnorm(beta_prop, 0, sqrt(gsl_matrix_get(Sigma, j, j))*gsl_vector_get(v, j), 1);
        logPrior1 = 0;
        
        expTerm = gsl_vector_get(omega, l) + eta * sumCorus_j(Sigma, gamma, j, l);
        
        logPrior2 = log(one_invLogit(expTerm));
        logPrior2_prop = log(invLogit(expTerm));
        
        logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)), 1);
        logProp =  log(q - *q_adj-sumGam) - log(sumGam + 1);
        
        
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
            
            c_dmvnorm(Yrow, meanY, 1, invSigma, &res);
            c_dmvnorm(Yrow, meanY_prop, 1, invSigma, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        
        logPrior1_prop = 0;
        logPrior1 = dnorm(gsl_matrix_get(B, l, j), 0, sqrt(gsl_matrix_get(Sigma, j, j)) * gsl_vector_get(v, j), 1);
        
        expTerm = gsl_vector_get(omega, l) + eta * sumCorus_j(Sigma, gamma, j, l);
        
        logPrior2 = log(invLogit(expTerm));
        logPrior2_prop = log(one_invLogit(expTerm));
        
        logProp_new = 0;
        logProp = dnorm(gsl_matrix_get(B, l, j), gsl_matrix_get(updateNonzB, l, j), sqrt(gsl_matrix_get(rwbetaVar, l, j)), 1) +log(sumGam) - log(q - *q_adj-sumGam+1);
        
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
            
            c_dmvnorm(Yrow, meanY, 1, invSigma, &res);
            c_dmvnorm(Yrow, meanY_prop, 1, invSigma, &res_prop);
            
            logLH += res;
            logLH_prop += res_prop;
        }
        logPrior1_prop = dnorm(beta_prop, 0, sqrt(gsl_matrix_get(Sigma, j, j))*gsl_vector_get(v, j), 1);
        logPrior1 = dnorm(gsl_matrix_get(B, m, j), 0, sqrt(gsl_matrix_get(Sigma, j, j))*gsl_vector_get(v, j), 1);
        
        expTerm1 = gsl_vector_get(omega, l) + eta * sumCorus_j(Sigma, gamma, j, l);
        
        logPrior2_1 = log(one_invLogit(expTerm1));
        logPrior2_1prop = log(invLogit(expTerm1));
        
        expTerm2 = gsl_vector_get(omega, m) + eta * sumCorus_j(Sigma, gamma, j, m);
        
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
                        
                        c_dmvnorm(Yrow, meanY, 1, invSigma, &res);
                        c_dmvnorm(Yrow, meanY_prop, 1, invSigma, &res_prop);
                        
                        logLH += res;
                        logLH_prop += res_prop;
                    }
                    logPrior1_prop = dnorm(beta_prop, 0, gsl_vector_get(v, j) * sqrt(gsl_matrix_get(Sigma, j, j)), 1);
                    logPrior1 = dnorm(gsl_matrix_get(B, k, j), 0, gsl_vector_get(v, j) * sqrt(gsl_matrix_get(Sigma, j, j)), 1);
                    
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

void updateIPus(gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_matrix *Sigma,
              gsl_matrix *invSigma,
              gsl_vector *beta0,
              gsl_vector *mu0,
              double h0)
{
    int n = Y -> size1;
    int p = Y -> size2;
    int j;
    
    gsl_matrix *varComp1 = gsl_matrix_calloc(p, p);
    gsl_matrix *varComp2 = gsl_matrix_calloc(p, p);
    gsl_matrix *varComp = gsl_matrix_calloc(p, p);
    
    gsl_matrix_set_identity(varComp1);
    
    gsl_matrix_scale(varComp1, (double) 1/h0);

    gsl_matrix_memcpy(varComp2, invSigma);
    gsl_matrix_scale(varComp2, (double) n);

    gsl_matrix_add(varComp2, varComp1);

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
    gsl_vector_scale(meanBeta2, (double) 1/h0);
    
    gsl_blas_dgemv(CblasNoTrans, 1, invSigma, meanBeta1, 1, meanBeta2);
    
    gsl_blas_dgemv(CblasNoTrans, 1, varComp, meanBeta2, 0, meanBeta);
    
    gsl_matrix *beta0spl = gsl_matrix_calloc(1, p);
    
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


/* updating variance-covariance matrix */

void updateCPus(int *q_adj,
              gsl_matrix *Y,
              gsl_matrix *X,
              gsl_matrix *B,
              gsl_matrix *gamma,
              gsl_matrix *Sigma,
              gsl_matrix *invSigma,
              gsl_vector *beta0,
              gsl_vector *omega,
              double eta,
              gsl_matrix *Psi0,
              double rho0,
              gsl_matrix *Psi_prop,
              double rho_prop,
              int *accept_Sigma)
{
    double res, res_prop, expTerm, expTerm_prop;
    double logPrior, logPrior_prop;
    double logProp_iniToprop, logProp_propToini;
    double logR;
    double logLH = 0;
    double logLH_prop = 0;
    double logPriorGam_ini = 0;
    double logPriorGam_prop = 0;
    
    int p = B -> size2;
    int q = B -> size1;
    int n = Y -> size1;
    
    int i, j, l, k, u;
    
    gsl_matrix *Sigma_prop = gsl_matrix_calloc(p, p);
    gsl_matrix *invSigma_prop = gsl_matrix_calloc(p, p);
    gsl_vector *Yrow = gsl_vector_calloc(p);
    gsl_vector *meanY = gsl_vector_calloc(p);
    
    gsl_matrix *Sigma_scaled = gsl_matrix_calloc(p, p);
    gsl_matrix_memcpy(Sigma_scaled, Sigma);
    gsl_matrix_scale(Sigma_scaled, rho_prop);
    
    c_riwishart(rho_prop+3, Sigma_scaled, Sigma_prop);
    
    c_solve(Sigma_prop, invSigma_prop);
    
    for(i = 0; i < n; i++)
    {
        gsl_matrix_get_row(Yrow, Y, i);
        gsl_vector_memcpy(meanY, beta0);
        gsl_vector_view Xrow = gsl_matrix_row(X, i);
        gsl_blas_dgemv(CblasTrans, 1, B, &Xrow.vector, 1, meanY);
        
        c_dmvnorm(Yrow, meanY, 1, invSigma, &res);
        c_dmvnorm(Yrow, meanY, 1, invSigma_prop, &res_prop);
        
        logLH += res;
        logLH_prop += res_prop;
    }
    
    for(l = 0; l < p; l++)
    {
        for(k = 0; k < (q - *q_adj); k++)
        {
            expTerm = gsl_vector_get(omega, k) + eta * sumCorus_j(Sigma, gamma, l, k);
            expTerm_prop = gsl_vector_get(omega, k) + eta * sumCorus_j(Sigma_prop, gamma, l, k);
            if(gsl_matrix_get(gamma, k, l) == 1)
            {
                logPriorGam_prop += log(invLogit(expTerm_prop));
                logPriorGam_ini  += log(invLogit(expTerm));
            }
            if(gsl_matrix_get(gamma, k, l) == 0)
            {
                logPriorGam_prop += log(one_invLogit(expTerm_prop));
                logPriorGam_ini += log(one_invLogit(expTerm));
            }
        }
    }
    
    gsl_matrix *Psi0invSigma = gsl_matrix_calloc(p, p);
    gsl_matrix *Psi0invSigma_prop = gsl_matrix_calloc(p, p);
    
    gsl_matrix *Sigma_prop_invSigma = gsl_matrix_calloc(p, p);
    gsl_matrix *Sigma_invSigma_prop = gsl_matrix_calloc(p, p);
    
    logPrior = -(rho0 + p + 1)/2*log(c_det(Sigma));
    logPrior_prop = -(rho0 + p + 1)/2*log(c_det(Sigma_prop));
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Psi0, invSigma, 0, Psi0invSigma);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Psi0, invSigma_prop, 0, Psi0invSigma_prop);
    
    for(j = 0; j < p; j++)
    {
        logPrior += -1/2 * gsl_matrix_get(Psi0invSigma, j, j);
        logPrior_prop += -1/2 * gsl_matrix_get(Psi0invSigma_prop, j, j);
    }
    
    logProp_propToini = -(2*(rho_prop+3) + p + 1)/2*log(c_det(Sigma));
    logProp_iniToprop = -(2*(rho_prop+3) + p + 1)/2*log(c_det(Sigma_prop));

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Sigma_prop, invSigma, 0, Sigma_prop_invSigma);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Sigma, invSigma_prop, 0, Sigma_invSigma_prop);
    
    for(j = 0; j < p; j++)
    {
        logProp_propToini += -rho_prop/2 * gsl_matrix_get(Sigma_prop_invSigma, j, j);
        logProp_iniToprop += -rho_prop/2 * gsl_matrix_get(Sigma_invSigma_prop, j, j);        
    }
    
    logR = logLH_prop - logLH + logPrior_prop - logPrior + logPriorGam_prop - logPriorGam_ini + logProp_propToini - logProp_iniToprop;
    
    u = log(runif(0, 1)) < logR;
    if(u == 1)
    {
        gsl_matrix_memcpy(Sigma, Sigma_prop);
        gsl_matrix_memcpy(invSigma, invSigma_prop);
        
        *accept_Sigma += 1;
    }
    
    gsl_matrix_free(Sigma_prop);
    gsl_matrix_free(invSigma_prop);
    
    gsl_vector_free(meanY);
    gsl_vector_free(Yrow);
    
    gsl_matrix_free(Sigma_prop_invSigma);
    gsl_matrix_free(Sigma_invSigma_prop);
    gsl_matrix_free(Psi0invSigma);
    gsl_matrix_free(Psi0invSigma_prop);
}





