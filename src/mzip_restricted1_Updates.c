#include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_sf.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"

#include "mzip_restricted1.h"


/* updating A, B, gamma */

void mzipBVS_restricted1_updateRP(int *p_adj,
              gsl_matrix *Y,
              gsl_matrix *X0,
              gsl_matrix *X1,
              gsl_vector *xi,
              gsl_vector *alpha0,
              gsl_matrix *A,
              gsl_matrix *W,
              gsl_vector *beta0,
              gsl_matrix *B,
              gsl_matrix *V,
              gsl_matrix *gamma,
              gsl_matrix *updateNonzA,
              gsl_matrix *updateNonzB,
              gsl_matrix *invR,
              gsl_vector *sigSq_alpha,
              gsl_vector *sigSq_beta,
              gsl_vector *phi,
              double nu_t,
              double sigSq_t,
              gsl_vector *v,
              gsl_vector *omega,
              double alpha_prop_var,
              double beta_prop_var,
              gsl_matrix *accept_B)
{
    double alpha_prop, beta_prop, res, res_prop;
    double logR, choice, sumG;
    double logLH, logLH_prop, logPrior, logPrior_prop, logProp_new, logProp;
    double sumGam;
    double p_add, p_del;
    double tempB, tempB_prop, xbeta, xbeta_prop, scale;
    
    int u, i, j, jj, k, l, m, ii, kk, kkk, lInx, count, count_rev, move, putInx;
    int refine_q, refine_p;
    
    int q = B -> size2;
    int p1 = B -> size1;
    int p = p1;
    int n = Y -> size1;
    
    p_add = (double) 1/3;
    p_del = (double) 1/3;
    
    gsl_matrix *gamma_prop = gsl_matrix_calloc(p, q);
    gsl_matrix *B_prop = gsl_matrix_calloc(p, q);
    gsl_matrix *A_prop = gsl_matrix_calloc(p, q);
    
    gsl_vector *Wrow = gsl_vector_calloc(q);
    gsl_vector *meanW = gsl_vector_calloc(q);
    gsl_vector *meanW_prop = gsl_vector_calloc(q);

    
    int numUpdate = 5;
    if(numUpdate > q)
    {
        numUpdate = q;
    }
  
    for(j = 0; j < numUpdate; j++)
    {
        jj = (int) runif(0, q);
        
        logLH = 0;
        logLH_prop = 0;
        logPrior_prop = 0;
        logPrior = 0;
        logProp = 0;
        logProp_new = 0;
        
        gsl_matrix_memcpy(gamma_prop, gamma);
        gsl_matrix_memcpy(B_prop, B);
        gsl_matrix_memcpy(A_prop, A);
        
        sumGam = 0;
        for(i = 0; i < p - *p_adj; i++)
        {
            sumGam += gsl_matrix_get(gamma, i, jj);
        }
        
        /* selecting a move */
        /* move: 1=add, 2=delete, 3=swap */
        
        if((p - *p_adj) == 1)
        {
            if(gsl_matrix_get(gamma, 0, jj) == 1) move = 2;
            if(gsl_matrix_get(gamma, 0, jj) == 0) move = 1;
        }
        if((p - *p_adj) > 1)
        {
            if(sumGam == (p - *p_adj)) move = 2;
            if(sumGam == 0) move = 1;
            if(sumGam != (p - *p_adj) && sumGam != 0)
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
            if((p - *p_adj) == 1)
            {
                l = 0;
            }else
            {
                choice  = runif(0, 1);
                lInx = (int) (choice * ((double) p - (double) *p_adj - (double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma, k, jj) == 0)
                    {
                        count += 1;
                    }
                }
                l = k;
            }
            
            gsl_matrix_set(gamma_prop, l, jj, 1);
            beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, jj), sqrt(beta_prop_var));
            alpha_prop = rnorm(gsl_matrix_get(updateNonzA, l, jj), sqrt(alpha_prop_var));
            
            gsl_matrix_set(B_prop, l, jj, beta_prop);
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            
            for(i = 0; i < n; i++)
            {
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_vector_view Bcol_prop = gsl_matrix_column(B_prop, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    gsl_blas_ddot(&X1row.vector, &Bcol_prop.vector, &xbeta_prop);
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    tempB_prop =  xbeta_prop + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    
                    logLH       += xbeta*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += xbeta_prop*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
                
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_memcpy(meanW, alpha0);
                gsl_vector_memcpy(meanW_prop, alpha0);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_blas_dgemv(CblasTrans, 1, A, &X0row.vector, 1, meanW);
                gsl_blas_dgemv(CblasTrans, 1, A_prop, &X0row.vector, 1, meanW_prop);
                
                scale = pow((double) sigSq_t / gsl_vector_get(phi, i), 0.5);
                
                c_dmvnorm(Wrow, meanW, scale, invR, &res);
                c_dmvnorm(Wrow, meanW_prop, scale, invR, &res_prop);
                
                logLH += res;
                logLH_prop += res_prop;
            }
            
            logPrior_prop = dnorm(beta_prop, 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v, jj), 1);
            logPrior_prop += dnorm(alpha_prop, 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v, jj), 1);
            logPrior_prop += gsl_vector_get(omega, l);
            logPrior = 0;
            
            logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, jj), sqrt(beta_prop_var), 1);
            logProp_new += dnorm(alpha_prop, gsl_matrix_get(updateNonzA, l, jj), sqrt(alpha_prop_var), 1);
            logProp =  log(p - *p_adj-sumGam) - log(sumGam + 1);
            
        }
        
        /* for delete move */
        if(move == 2)
        {
            if((p - *p_adj) == 1) l = 0;
            else
            {
                choice  = runif(0, 1);
                
                lInx = (int) (choice * ((double) sumGam)) + 1;
                count = 0;
                k = -1;
                while(lInx != count)
                {
                    k += 1;
                    if(gsl_matrix_get(gamma, k, jj) == 1)
                    {
                        count += 1;
                    }
                }
                l = k;
            }
            
            gsl_matrix_set(gamma_prop, l, jj, 0);
            beta_prop = 0;
            alpha_prop = 0;
            
            gsl_matrix_set(B_prop, l, jj, beta_prop);
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            
            for(i = 0; i < n; i++)
            {
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_vector_view Bcol_prop = gsl_matrix_column(B_prop, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    gsl_blas_ddot(&X1row.vector, &Bcol_prop.vector, &xbeta_prop);
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    tempB_prop =  xbeta_prop + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    
                    logLH       += xbeta*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += xbeta_prop*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
                
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_memcpy(meanW, alpha0);
                gsl_vector_memcpy(meanW_prop, alpha0);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_blas_dgemv(CblasTrans, 1, A, &X0row.vector, 1, meanW);
                gsl_blas_dgemv(CblasTrans, 1, A_prop, &X0row.vector, 1, meanW_prop);
                
                scale = pow((double) sigSq_t / gsl_vector_get(phi, i), 0.5);
                
                c_dmvnorm(Wrow, meanW, scale, invR, &res);
                c_dmvnorm(Wrow, meanW_prop, scale, invR, &res_prop);
                
                logLH += res;
                logLH_prop += res_prop;
            }
            
            logPrior_prop = 0;
            logPrior = dnorm(gsl_matrix_get(B, l, jj), 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v, jj), 1);
            logPrior += dnorm(gsl_matrix_get(A, l, jj), 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v, jj), 1);
            logPrior += gsl_vector_get(omega, l);
            
            logProp_new =  0;
            logProp = dnorm(gsl_matrix_get(updateNonzB, l, jj), beta_prop, sqrt(beta_prop_var), 1);
            logProp += dnorm(gsl_matrix_get(updateNonzA, l, jj), alpha_prop, sqrt(alpha_prop_var), 1);
            logProp += log(sumGam) - log(p - *p_adj-sumGam+1);
        }
        
        /* for swap move */
        if(move == 3)
        {
            choice  = runif(0, 1);
            lInx = (int) (choice * ((double) p - (double) *p_adj - (double) sumGam)) + 1;
            count = 0;
            k = -1;
            while(lInx != count)
            {
                k += 1;
                if(gsl_matrix_get(gamma, k, jj) == 0)
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
                if(gsl_matrix_get(gamma, k, jj) == 1)
                {
                    count += 1;
                }
            }
            m = k;
            
            gsl_matrix_set(gamma_prop, l, jj, 1);
            gsl_matrix_set(gamma_prop, m, jj, 0);
            
            beta_prop = rnorm(gsl_matrix_get(updateNonzB, l, jj), sqrt(beta_prop_var));
            alpha_prop = rnorm(gsl_matrix_get(updateNonzA, l, jj), sqrt(alpha_prop_var));
            
            gsl_matrix_set(B_prop, l, jj, beta_prop);
            gsl_matrix_set(B_prop, m, jj, 0);
            gsl_matrix_set(A_prop, l, jj, alpha_prop);
            gsl_matrix_set(A_prop, m, jj, 0);
            
            
            for(i = 0; i < n; i++)
            {
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1row = gsl_matrix_row(X1, i);
                    gsl_vector_view Bcol = gsl_matrix_column(B, jj);
                    gsl_vector_view Bcol_prop = gsl_matrix_column(B_prop, jj);
                    gsl_blas_ddot(&X1row.vector, &Bcol.vector, &xbeta);
                    gsl_blas_ddot(&X1row.vector, &Bcol_prop.vector, &xbeta_prop);
                    
                    tempB = xbeta + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    tempB_prop =  xbeta_prop + gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    
                    logLH       += xbeta*gsl_matrix_get(Y, i, jj) - exp(tempB);
                    logLH_prop  += xbeta_prop*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
                
                gsl_matrix_get_row(Wrow, W, i);
                gsl_vector_memcpy(meanW, alpha0);
                gsl_vector_memcpy(meanW_prop, alpha0);
                gsl_vector_view X0row = gsl_matrix_row(X0, i);
                gsl_blas_dgemv(CblasTrans, 1, A, &X0row.vector, 1, meanW);
                gsl_blas_dgemv(CblasTrans, 1, A_prop, &X0row.vector, 1, meanW_prop);
                
                scale = pow((double) sigSq_t / gsl_vector_get(phi, i), 0.5);
                
                c_dmvnorm(Wrow, meanW, scale, invR, &res);
                c_dmvnorm(Wrow, meanW_prop, scale, invR, &res_prop);
                
                logLH += res;
                logLH_prop += res_prop;
            }
            
            logPrior_prop = dnorm(beta_prop, 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v, jj), 1);
            logPrior_prop += dnorm(alpha_prop, 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v, jj), 1);
            logPrior_prop += gsl_vector_get(omega, l);
            
            logPrior = dnorm(gsl_matrix_get(B, m, jj), 0, sqrt(gsl_vector_get(sigSq_beta, l))*gsl_vector_get(v, jj), 1);
            logPrior += dnorm(gsl_matrix_get(A, m, jj), 0, sqrt(gsl_vector_get(sigSq_alpha, l))*gsl_vector_get(v, jj), 1);
            logPrior += gsl_vector_get(omega, m);
            
            logProp_new = dnorm(beta_prop, gsl_matrix_get(updateNonzB, l, jj), sqrt(beta_prop_var), 1);
            logProp_new += dnorm(alpha_prop, gsl_matrix_get(updateNonzA, l, jj), sqrt(alpha_prop_var), 1);
            logProp = dnorm(gsl_matrix_get(updateNonzB, m, jj), beta_prop, sqrt(beta_prop_var), 1);
            logProp += dnorm(gsl_matrix_get(updateNonzA, m, jj), alpha_prop, sqrt(alpha_prop_var), 1);
            
        }
        
        logR = logLH_prop - logLH + logPrior_prop - logPrior + logProp - logProp_new;
        u = log(runif(0, 1)) < logR;
        
        if(u == 1)
        {
            gsl_matrix_swap(gamma, gamma_prop);
            
            if(move == 1 || move == 3)
            {
                gsl_matrix_set(updateNonzB, l, jj, beta_prop);
                gsl_matrix_set(updateNonzA, l, jj, alpha_prop);
            }
            if(move == 1 || move == 2)
            {
                gsl_matrix_set(B, l, jj, beta_prop);
                gsl_matrix_set(A, l, jj, alpha_prop);
                gsl_matrix_set(accept_B, l, jj, (gsl_matrix_get(accept_B, l, jj) + u));
            }
            if(move == 3)
            {
                gsl_matrix_set(B, l, jj, beta_prop);
                gsl_matrix_set(B, m, jj, 0);
                gsl_matrix_set(A, l, jj, alpha_prop);
                gsl_matrix_set(A, m, jj, 0);
                gsl_matrix_set(accept_B, l, jj, (gsl_matrix_get(accept_B, l, jj) + u));
                gsl_matrix_set(accept_B, m, jj, (gsl_matrix_get(accept_B, m, jj) + u));
            }
        }

    }
    
    /* Refining step for B and A using M-H */
    gsl_vector *colSumsGam = gsl_vector_calloc(q);
    c_colSums(gamma, colSumsGam);
    
    sumG = 0;
    for(j = 0; j < q; j++) sumG += gsl_vector_get(colSumsGam, j);
    
    count = 0;
    for(j = 0; j < q; j++)
    {
        if(gsl_vector_get(colSumsGam, j) > 0) count += 1;
    }
    
    count_rev = count;
    if(count == 0) count_rev = 1;
    
    gsl_vector *colSumsGamNonzInx = gsl_vector_calloc(count_rev);
    putInx = 0;
    
    for(j = 0; j < q; j++)
    {
        if(gsl_vector_get(colSumsGam, j) > 0)
        {
            gsl_vector_set(colSumsGamNonzInx, putInx, j);
            putInx += 1;
        }
    }
    
    refine_q = (int) c_min(5, q);
    refine_p = (int) c_min(100, p);
    
    double loglh, loglh_prop;
    double D1, D2, D1_prop, D2_prop;
    double temp_prop_me, temp_prop_var, temp_prop_me_prop, temp_prop_var_prop;
    double logProp_IniToProp, logProp_PropToIni;
    
    double sum_X_phi = 0;
    double alpha_mean, alpha_var, alpha_a, alpha_b, val_temp, sum_val_temp;
    int ll;
    
    gsl_vector *beta_vec = gsl_vector_calloc(p1);
    gsl_vector *beta_vec_prop = gsl_vector_calloc(p1);
    gsl_vector *Xbeta = gsl_vector_calloc(n);
    gsl_vector *Xbeta_prop = gsl_vector_calloc(n);
    
    if(sumG > 0)
    {
        for(ii = 0; ii < refine_q; ii++)
        {
            choice  = (int) runif(0, count_rev);
            jj = (int) gsl_vector_get(colSumsGamNonzInx, choice);

            for(kkk = 0; kkk < refine_p; kkk++)
            {
                kk  = (int) runif(0, p);

                if(gsl_matrix_get(gamma, kk, jj) ==1)
                {
                    /* updating B */
                    for(k = 0; k < p1; k++)
                    {
                        gsl_vector_set(beta_vec, k, gsl_matrix_get(B, k, jj));
                    }
                    
                    gsl_blas_dgemv(CblasNoTrans, 1, X1, beta_vec, 0, Xbeta);
                    
                    loglh = 0;
                    loglh_prop = 0;
                    D1 = 0; D2 = 0;
                    D1_prop = 0; D2_prop = 0;
                    
                    for(i = 0; i < n; i++)
                    {
                        tempB = 0;
                        if(gsl_matrix_get(W, i, jj) >= 0)
                        {
                            tempB = gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                            tempB += gsl_vector_get(Xbeta, i);
                            
                            loglh += gsl_vector_get(Xbeta, i)*gsl_matrix_get(Y, i, jj) - exp(tempB);
                            D1 += gsl_matrix_get(Y, i, jj)*gsl_matrix_get(X1, i, kk) - exp(tempB)*gsl_matrix_get(X1, i, kk);
                            D2 -= exp(tempB)*pow(gsl_matrix_get(X1, i, kk), 2);
                        }
                    }
                    
                    loglh -= pow(gsl_vector_get(beta_vec, kk), 2)/(2*gsl_vector_get(sigSq_beta, kk));
                    D1 -= gsl_vector_get(beta_vec, kk)/gsl_vector_get(sigSq_beta, kk);
                    D2 -= 1/gsl_vector_get(sigSq_beta, kk);
                    
                    if(D1/D2 > 1 || D1/D2 < -1 )
                    {
                        gsl_vector_memcpy(beta_vec_prop, beta_vec);
                        gsl_vector_set(beta_vec_prop, kk, rnorm(gsl_vector_get(beta_vec, kk), sqrt(beta_prop_var)));
                        
                        gsl_blas_dgemv(CblasNoTrans, 1, X1, beta_vec_prop, 0, Xbeta_prop);
                        
                        for(i = 0; i < n; i++)
                        {
                            tempB_prop = 0;
                            if(gsl_matrix_get(W, i, jj) >= 0)
                            {
                                tempB_prop = gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                                tempB_prop += gsl_vector_get(Xbeta_prop, i);
                                
                                loglh_prop += gsl_vector_get(Xbeta_prop, i)*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                            }
                        }
                        
                        loglh_prop -= pow(gsl_vector_get(beta_vec_prop, kk), 2)/(2*gsl_vector_get(sigSq_beta, kk));
                        
                        logR = loglh_prop - loglh;
                    }else
                    {
                        temp_prop_me    = gsl_vector_get(beta_vec, kk) - D1/D2;
                        temp_prop_var   = -pow(2.4, 2)/D2;
                        
                        gsl_vector_memcpy(beta_vec_prop, beta_vec);
                        gsl_vector_set(beta_vec_prop, kk, rnorm(temp_prop_me, sqrt(temp_prop_var)));
                        
                        gsl_blas_dgemv(CblasNoTrans, 1, X1, beta_vec_prop, 0, Xbeta_prop);
                        
                        for(i = 0; i < n; i++)
                        {
                            tempB_prop = 0;
                            if(gsl_matrix_get(W, i, jj) >= 0)
                            {
                                tempB_prop = gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                                tempB_prop += gsl_vector_get(Xbeta_prop, i);
                                
                                loglh_prop += gsl_vector_get(Xbeta_prop, i)*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                                D1_prop += gsl_matrix_get(Y, i, jj)*gsl_matrix_get(X1, i, kk) - exp(tempB_prop)*gsl_matrix_get(X1, i, kk);
                                D2_prop -= exp(tempB_prop)*pow(gsl_matrix_get(X1, i, kk), 2);
                            }
                        }
                        
                        loglh_prop -= pow(gsl_vector_get(beta_vec_prop, kk), 2)/(2*gsl_vector_get(sigSq_beta, kk));
                        
                        D1_prop -= gsl_vector_get(beta_vec_prop, kk)/gsl_vector_get(sigSq_beta, kk);
                        D2_prop -= 1/gsl_vector_get(sigSq_beta, kk);
                        
                        temp_prop_me_prop    = gsl_vector_get(beta_vec_prop, kk) - D1_prop/D2_prop;
                        temp_prop_var_prop   = -pow(2.4, 2)/D2_prop;
                        
                        logProp_IniToProp = dnorm(gsl_vector_get(beta_vec_prop, kk), temp_prop_me, sqrt(temp_prop_var), 1);
                        logProp_PropToIni = dnorm(gsl_vector_get(beta_vec, kk), temp_prop_me_prop, sqrt(temp_prop_var_prop), 1);
                        
                        logR = loglh_prop - loglh + logProp_PropToIni - logProp_IniToProp;
                    }
                    
                    u = log(runif(0, 1)) < logR;
                    
                    if(u == 1)
                    {
                        gsl_matrix_set(B, kk, jj, gsl_vector_get(beta_vec_prop, kk));
                        gsl_matrix_set(updateNonzB, kk, jj, gsl_vector_get(beta_vec_prop, kk));
                        gsl_matrix_set(accept_B, kk, jj, gsl_matrix_get(accept_B, kk, jj)+1);
                    }

                    /* updating A */
                    
                    ll = kk;
                    sum_X_phi = 0;
                    for(i = 0; i < n; i++)
                    {
                        sum_X_phi += gsl_vector_get(phi, i) * pow(gsl_matrix_get(X0, i, ll), 2);
                    }
                    
                    alpha_a = sum_X_phi * gsl_matrix_get(invR, jj, jj) / sigSq_t + 1/gsl_vector_get(sigSq_alpha, ll);
                    alpha_var = pow(alpha_a, -1);

                    alpha_b = 0;
                    for(k = 0; k < q; k++)
                    {
                        if(k != jj)
                        {
                            alpha_b -= gsl_matrix_get(A, ll, k) * gsl_matrix_get(invR, k, jj);
                        }
                    }
                    alpha_b *= sum_X_phi;
                    alpha_b /= sigSq_t;

                    for(k = 0; k < q; k++)
                    {
                        sum_val_temp = 0;
                        for(i = 0; i < n; i++)
                        {
                            val_temp = gsl_matrix_get(W, i, k)-gsl_vector_get(alpha0, k);
                            if(p > 1)
                            {
                                for(m = 0; m < p; m++)
                                {
                                    if(m != ll)
                                    {
                                        val_temp -= gsl_matrix_get(X0, i, m) * gsl_matrix_get(A, m, k);
                                    }
                                }
                            }
                            val_temp *= gsl_matrix_get(X0, i, ll) * gsl_vector_get(phi, i);
                            sum_val_temp += val_temp;
                        }
                        sum_val_temp *= gsl_matrix_get(invR, jj, k);
                        alpha_b += pow(sigSq_t, -1) *  sum_val_temp;
                    }

                    alpha_mean = (double) alpha_b/alpha_a;
                    gsl_matrix_set(A, ll, jj, rnorm(alpha_mean, sqrt(alpha_var)));
                    gsl_matrix_set(updateNonzA, ll, jj, gsl_matrix_get(A, ll, jj));
                }
            }
        }
    }
    gsl_matrix_free(gamma_prop);
    gsl_matrix_free(B_prop);
    gsl_matrix_free(A_prop);
    
    gsl_vector_free(Wrow);
    gsl_vector_free(meanW);
    gsl_vector_free(meanW_prop);
    
    gsl_vector_free(beta_vec);
    gsl_vector_free(beta_vec_prop);
    gsl_vector_free(Xbeta);
    gsl_vector_free(Xbeta_prop);
    
    return;
}




void mzipBVS_restricted1_update_Sigma(gsl_matrix *X0,
                    gsl_vector *alpha0,
                    gsl_matrix *A,
                    gsl_matrix *W,
                    gsl_matrix *V,
                    gsl_matrix *R,
                    gsl_matrix *invR,
                    gsl_vector *S,
                    gsl_matrix *diaginvS,
                    gsl_matrix *invSigma,
                    gsl_vector *phi,
                    gsl_matrix *Q,
                    double sigSq_t,
                    double nu_t,
                    gsl_matrix *Psi0,
                    double rho0,
                    double rho_s,
                    int *accept_R)
{
    int i, j, u;
    double res, res_prop, logR;
    double logPrior, logPrior_prop;
    double logProp_iniToprop, logProp_propToini;
    double logLH = 0;
    double logLH_prop = 0;
    
    int n = W -> size1;
    int q = W -> size2;
    
    gsl_matrix *R_prop = gsl_matrix_calloc(q, q);
    gsl_matrix *invR_prop = gsl_matrix_calloc(q, q);
    gsl_matrix *invSigma_prop = gsl_matrix_calloc(q, q);
    gsl_matrix *Sigma_prop = gsl_matrix_calloc(q, q);
    gsl_matrix *diagS_prop = gsl_matrix_calloc(q, q);
    
    gsl_matrix *Sigma_scaled = gsl_matrix_calloc(q, q);
    gsl_matrix *Sigma = gsl_matrix_calloc(q, q);
    gsl_vector *Vrow = gsl_vector_calloc(q);
    gsl_vector *Wrow = gsl_vector_calloc(q);
    gsl_vector *zero_q = gsl_vector_calloc(q);
    gsl_vector *meanW = gsl_vector_calloc(q);
    
    gsl_matrix *Sigma_prop_invSigma = gsl_matrix_calloc(q, q);
    gsl_matrix *Sigma_invSigma_prop = gsl_matrix_calloc(q, q);
    
    gsl_matrix *Psi0invSigma = gsl_matrix_calloc(q, q);
    gsl_matrix *Psi0invSigma_prop = gsl_matrix_calloc(q, q);
    gsl_matrix *invTemp = gsl_matrix_calloc(q, q);
    
    c_solve(invSigma, Sigma);
    
    gsl_matrix_memcpy(Sigma_scaled, Sigma);
    gsl_matrix_scale(Sigma_scaled, rho_s);
    
    c_riwishart(rho_s+ (double) q+1, Sigma_scaled, Sigma_prop, invSigma_prop);
    
    for(j = 0; j < q; j++)
    {
        gsl_matrix_set(diagS_prop, j, j, (double) pow((double) gsl_matrix_get(Sigma_prop, j, j), 0.5));
    }
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, invSigma_prop,diagS_prop, 0, invTemp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, diagS_prop, invTemp, 0, invR_prop);
    
    for(i = 0; i < n; i++)
    {
        gsl_matrix_get_row(Vrow, V, i);
        c_dmvnorm(Vrow, zero_q, 1, invSigma, &res);
        c_dmvnorm(Vrow, zero_q, 1, invSigma_prop, &res_prop);
        
        logLH += res;
        logLH_prop += res_prop;
        
        gsl_matrix_get_row(Wrow, W, i);
        gsl_vector_memcpy(meanW, alpha0);
        gsl_vector_view Xrow = gsl_matrix_row(X0, i);
        gsl_blas_dgemv(CblasTrans, 1, A, &Xrow.vector, 1, meanW);
        
        c_dmvnorm(Wrow, meanW, pow(sigSq_t/gsl_vector_get(phi, i), 0.5), invR, &res);
        c_dmvnorm(Wrow, meanW, pow(sigSq_t/gsl_vector_get(phi, i), 0.5), invR_prop, &res_prop);
        
        logLH += res;
        logLH_prop += res_prop;
    }
    
    logPrior = -(rho0 + q + 1)/2*log(c_det(Sigma));
    logPrior_prop = -(rho0 + q + 1)/2*log(c_det(Sigma_prop));
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Psi0, invSigma, 0, Psi0invSigma);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Psi0, invSigma_prop, 0, Psi0invSigma_prop);
    
    for(j = 0; j < q; j++)
    {
        logPrior += -1/2 * gsl_matrix_get(Psi0invSigma, j, j);
        logPrior_prop += -1/2 * gsl_matrix_get(Psi0invSigma_prop, j, j);
    }
    
    logProp_propToini = -(2*rho_s + 3*q + 3)/2*log(c_det(Sigma));
    logProp_iniToprop = -(2*rho_s + 3*q + 3)/2*log(c_det(Sigma_prop));
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Sigma_prop, invSigma, 0, Sigma_prop_invSigma);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, Sigma, invSigma_prop, 0, Sigma_invSigma_prop);
    
    for(j = 0; j < q; j++)
    {
        logProp_propToini += -rho_s/2 * gsl_matrix_get(Sigma_prop_invSigma, j, j);
        logProp_iniToprop += -rho_s/2 * gsl_matrix_get(Sigma_invSigma_prop, j, j);
    }
    
    logR = logLH_prop - logLH + logPrior_prop - logPrior + logProp_propToini - logProp_iniToprop;
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        gsl_matrix_memcpy(invSigma, invSigma_prop);
        gsl_matrix_memcpy(invR, invR_prop);
        c_solve(invR, R);
        for(j = 0; j < q; j++)
        {
            gsl_vector_set(S, j, gsl_matrix_get(diagS_prop, j, j));
        }
        
        gsl_matrix_memcpy(invSigma, invSigma_prop);
        *accept_R += 1;
    }
    gsl_matrix_free(R_prop);
    gsl_matrix_free(invR_prop);
    gsl_matrix_free(invSigma_prop);
    gsl_matrix_free(Sigma_prop);
    gsl_matrix_free(diagS_prop);
    
    gsl_matrix_free(Sigma_scaled);
    gsl_matrix_free(Sigma);
    gsl_vector_free(Vrow);
    gsl_vector_free(Wrow);
    gsl_vector_free(zero_q);
    gsl_vector_free(meanW);
    
    gsl_matrix_free(Sigma_prop_invSigma);
    gsl_matrix_free(Sigma_invSigma_prop);
    
    gsl_matrix_free(Psi0invSigma);
    gsl_matrix_free(Psi0invSigma_prop);
    gsl_matrix_free(invTemp);
    
    return;
}






/* updating V */

void mzipBVS_restricted1_update_V(gsl_matrix *Y,
              gsl_matrix *X1,
              gsl_vector *xi,
              gsl_matrix *W,
              gsl_vector *beta0,
              gsl_matrix *B,
              gsl_matrix *V,
              gsl_matrix *invSigma,
              gsl_matrix *accept_V,
              double V_prop_var)
{
    int i, jj, k, u;
    double temp_prop, tempB, tempB_prop, loglh, loglh_prop, logR;
    
    double D1, D2, D1_prop, D2_prop;
    double temp_prop_me, temp_prop_var, temp_prop_me_prop, temp_prop_var_prop;
    double logProp_IniToProp, logProp_PropToIni;
    
    int n = Y -> size1;
    int q = Y -> size2;
    int p1 = X1 -> size2;
    
    gsl_vector *beta = gsl_vector_calloc(p1);
    gsl_vector *Xbeta = gsl_vector_calloc(n);
    
    for(jj=0; jj<q; jj++)
    {
        for(k = 0; k < p1; k++)
        {
            gsl_vector_set(beta, k, gsl_matrix_get(B, k, jj));
        }
        gsl_blas_dgemv(CblasNoTrans, 1, X1, beta, 0, Xbeta);
        
        for(i=0; i<n; i++)
        {
            loglh = 0;
            loglh_prop = 0;
            D1 = 0; D2 = 0;
            D1_prop = 0; D2_prop = 0;
            
            tempB = 0;
            if(gsl_matrix_get(W, i, jj) >= 0)
            {
                tempB = gsl_vector_get(beta0, jj)+gsl_vector_get(Xbeta, i)+gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                
                loglh += gsl_matrix_get(V, i, jj)*gsl_matrix_get(Y, i, jj) - exp(tempB);
                D1 += gsl_matrix_get(Y, i, jj) - exp(tempB);
                D2 -= exp(tempB);
            }
            
            loglh -= 0.5 * pow(gsl_matrix_get(V, i, jj), 2) * gsl_matrix_get(invSigma, jj, jj);
            D1 -= gsl_matrix_get(V, i, jj)* gsl_matrix_get(invSigma, jj, jj);
            D2 -= gsl_matrix_get(invSigma, jj, jj);

            for(k = 0; k < q; k++)
            {
                if(k != jj)
                {
                    loglh -= gsl_matrix_get(V, i, jj) * gsl_matrix_get(V, i, k) * gsl_matrix_get(invSigma, jj, k);
                    D1 -= gsl_matrix_get(V, i, k) * gsl_matrix_get(invSigma, jj, k);
                }
            }
            
            if(D1/D2 > 1 || D1/D2 < -1 )
            {
                temp_prop =  rnorm(gsl_matrix_get(V, i, jj), sqrt(V_prop_var));
                
                tempB_prop = 0;
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    tempB_prop = gsl_vector_get(beta0, jj)+gsl_vector_get(Xbeta, i)+temp_prop+log(gsl_vector_get(xi, i));
                    
                    loglh_prop += temp_prop*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                }
                loglh_prop -= 0.5 * pow(temp_prop, 2) * gsl_matrix_get(invSigma, jj, jj);
                
                for(k = 0; k < q; k++)
                {
                    if(k != jj)
                    {
                        loglh_prop -= temp_prop * gsl_matrix_get(V, i, k) * gsl_matrix_get(invSigma, jj, k);
                    }
                }
                
                logR = loglh_prop - loglh;
            }else
            {
                temp_prop_me    = gsl_matrix_get(V, i, jj) - D1/D2;
                temp_prop_var   = -pow(2.4, 2)/D2;
                
                temp_prop =  rnorm(temp_prop_me, sqrt(temp_prop_var));
                
                tempB_prop = 0;
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    tempB_prop = gsl_vector_get(beta0, jj)+gsl_vector_get(Xbeta, i)+temp_prop+log(gsl_vector_get(xi, i));
                    
                    loglh_prop += temp_prop*gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                    D1_prop += gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                    D2_prop -= exp(tempB_prop);
                }
                loglh_prop -= 0.5 * pow(temp_prop, 2) * gsl_matrix_get(invSigma, jj, jj);
                D1_prop -= temp_prop * gsl_matrix_get(invSigma, jj, jj);
                D2_prop -= gsl_matrix_get(invSigma, jj, jj);
                
                for(k = 0; k < q; k++)
                {
                    if(k != jj)
                    {
                        loglh_prop -= temp_prop * gsl_matrix_get(V, i, k) * gsl_matrix_get(invSigma, jj, k);
                        D1_prop -= gsl_matrix_get(V, i, k) * gsl_matrix_get(invSigma, jj, k);
                    }
                }
                
                temp_prop_me_prop   = temp_prop - D1_prop/D2_prop;
                temp_prop_var_prop   = -pow(2.4, 2)/D2_prop;
                
                logProp_IniToProp = dnorm(temp_prop, temp_prop_me, sqrt(temp_prop_var), 1);
                logProp_PropToIni = dnorm(gsl_matrix_get(V, i, jj), temp_prop_me_prop, sqrt(temp_prop_var_prop), 1);
                
                logR = loglh_prop - loglh + logProp_PropToIni - logProp_IniToProp;
            }

            u = log(runif(0, 1)) < logR;

            if(u == 1)
            {
                gsl_matrix_set(V, i, jj, temp_prop);
                gsl_matrix_set(accept_V, i, jj, gsl_matrix_get(accept_V, i, jj)+1);
            }
        }
    }
    

    gsl_vector_free(beta);
    gsl_vector_free(Xbeta);
    
    return;
}




/* updating beta0 */

void mzipBVS_restricted1_update_beta0(gsl_matrix *Y,
                  gsl_matrix *X1,
                  gsl_vector *xi,
                  gsl_matrix *W,
                  gsl_vector *beta0,
                  gsl_matrix *B,
                  gsl_matrix *V,
                  double mu_beta0,
                  double sigSq_beta0,
                  double beta0_prop_var,
                  gsl_vector *accept_beta0)
{
    int i, j, jj, u;
    double temp_prop, tempB, tempB_prop, loglh, loglh_prop, logR;
    double D1, D2, D1_prop, D2_prop;
    double temp_prop_me, temp_prop_var, temp_prop_me_prop, temp_prop_var_prop;
    double logProp_IniToProp, logProp_PropToIni;
 
    int n = Y -> size1;
    int q = Y -> size2;
    
    int numUpdate = 5;
    if(numUpdate > q)
    {
        numUpdate = q;
    }
    
    for(j = 0; j < numUpdate; j++)
    {
        jj = (int) runif(0, q);
        
        loglh = 0;
        loglh_prop = 0;
        D1 = 0; D2 = 0;
        D1_prop = 0; D2_prop = 0;
         
        for(i=0; i<n; i++)
        {
            tempB = 0;
            if(gsl_matrix_get(W, i, jj) >= 0)
            {
                gsl_vector_view X1_row = gsl_matrix_row(X1, i);
                gsl_vector_view B_col = gsl_matrix_column(B, jj);
                gsl_blas_ddot(&X1_row.vector, &B_col.vector, &tempB);

                tempB += gsl_vector_get(beta0, jj) + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                
                loglh += gsl_matrix_get(Y, i, jj)*gsl_vector_get(beta0, jj) - exp(tempB);
                D1 += gsl_matrix_get(Y, i, jj) - exp(tempB);
                D2 -= exp(tempB);
            }
        }

        loglh -= pow(gsl_vector_get(beta0, jj)-mu_beta0, 2)/(2*sigSq_beta0);
        D1 -= gsl_vector_get(beta0, jj)/sigSq_beta0;
        D2 -= 1/sigSq_beta0;
        
        if(D1/D2 > 1 || D1/D2 < -1 )
        {
            temp_prop =  rnorm(gsl_vector_get(beta0, jj), sqrt(beta0_prop_var));
            
            for(i=0; i<n; i++)
            {
                tempB_prop = 0;
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1_row = gsl_matrix_row(X1, i);
                    gsl_vector_view B_col = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1_row.vector, &B_col.vector, &tempB_prop);
                    
                    tempB_prop += temp_prop + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    loglh_prop += gsl_matrix_get(Y, i, jj)*temp_prop - exp(tempB_prop);
                }
            }
            
            loglh_prop -= pow(temp_prop-mu_beta0, 2)/(2*sigSq_beta0);
            logR = loglh_prop - loglh;
        }else
        {
            temp_prop_me    = gsl_vector_get(beta0, jj) - D1/D2;
            temp_prop_var   = -pow(2.4, 2)/D2;
            temp_prop =  rnorm(temp_prop_me, sqrt(temp_prop_var));
            
            for(i=0; i<n; i++)
            {
                tempB_prop = 0;
                if(gsl_matrix_get(W, i, jj) >= 0)
                {
                    gsl_vector_view X1_row = gsl_matrix_row(X1, i);
                    gsl_vector_view B_col = gsl_matrix_column(B, jj);
                    gsl_blas_ddot(&X1_row.vector, &B_col.vector, &tempB_prop);
                    
                    tempB_prop += temp_prop + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
                    
                    loglh_prop += gsl_matrix_get(Y, i, jj)*temp_prop - exp(tempB_prop);
                    D1_prop += gsl_matrix_get(Y, i, jj) - exp(tempB_prop);
                    D2_prop -= exp(tempB_prop);
                }
            }
            
            loglh_prop -= pow(temp_prop-mu_beta0, 2)/(2*sigSq_beta0);
            D1_prop -= temp_prop/sigSq_beta0;
            D2_prop -= 1/sigSq_beta0;
            
            temp_prop_me_prop    = temp_prop - D1_prop/D2_prop;
            temp_prop_var_prop   = -pow(2.4, 2)/D2_prop;
            
            logProp_IniToProp = dnorm(temp_prop, temp_prop_me, sqrt(temp_prop_var), 1);
            logProp_PropToIni = dnorm(gsl_vector_get(beta0, jj), temp_prop_me_prop, sqrt(temp_prop_var_prop), 1);
            
            logR = loglh_prop - loglh + logProp_PropToIni - logProp_IniToProp;
        }
        u = log(runif(0, 1)) < logR;
        
        if(u == 1)
        {
            gsl_vector_set(beta0, jj, temp_prop);
            gsl_vector_set(accept_beta0, jj, gsl_vector_get(accept_beta0, jj)+1);
        }
    }
    return;
}



/* updating W */

void mzipBVS_restricted1_update_W(gsl_matrix *Y,
              gsl_matrix *X0,
              gsl_matrix *X1,
              gsl_vector *xi,
              gsl_vector *alpha0,
              gsl_matrix *A,
              gsl_matrix *W,
              gsl_vector *beta0,
              gsl_matrix *B,
              gsl_matrix *V,
              gsl_matrix *invR,
              gsl_vector *phi,
              double nu_t,
              double sigSq_t)
{
    int i, k, jj;
    
    double mean, sd, sample, sum, prob, tempA, tempB, eta, cumNorm, sel;
    
    int n = Y -> size1;
    int q = Y -> size2;
    
    jj = (int) runif(0, q);
    
    for(i = 0; i < n; i++)
    {
        gsl_vector_view X0_row = gsl_matrix_row(X0, i);
        gsl_vector_view X1_row = gsl_matrix_row(X1, i);
        
        sd = pow(sigSq_t/gsl_vector_get(phi, i)/gsl_matrix_get(invR, jj, jj), 0.5);
        sum = 0;
        for(k=0; k<q; k++)
        {
            if(k != jj)
            {
                gsl_vector_view A_col = gsl_matrix_column(A, k);
                gsl_blas_ddot(&X0_row.vector, &A_col.vector, &tempA);
                sum += (gsl_vector_get(alpha0, k)+tempA-gsl_matrix_get(W, i, k))*gsl_matrix_get(invR, jj, k);
            }
        }
        gsl_vector_view A_col = gsl_matrix_column(A, jj);
        gsl_blas_ddot(&X0_row.vector, &A_col.vector, &tempA);
        sum += (gsl_vector_get(alpha0, jj)+tempA)*gsl_matrix_get(invR, jj, jj);
        mean = sum / gsl_matrix_get(invR, jj, jj);
        
        if(gsl_matrix_get(Y, i, jj) > 0)
        {
            c_rtnorm(mean, sd, 0, 100000, 0, 1, &sample);
        }else if(gsl_matrix_get(Y, i, jj) == 0)
        {
            gsl_vector_view B_col = gsl_matrix_column(B, jj);
            gsl_blas_ddot(&X1_row.vector, &B_col.vector, &tempB);
            eta = gsl_vector_get(beta0, jj) + tempB + gsl_matrix_get(V, i, jj)+log(gsl_vector_get(xi, i));
            cumNorm = pnorm(0, mean, sd, 1, 0);
            prob = exp(-exp(eta))*(1-cumNorm)/(exp(-exp(eta))*(1-cumNorm) + cumNorm);
            sel = runif(0, 1);
            if(sel <= prob)
            {
                c_rtnorm(mean, sd, 0, 100000, 0, 1, &sample);
            }else
            {
                c_rtnorm(mean, sd, -100000, 0, 1, 0, &sample);
            }
        }
        gsl_matrix_set(W, i, jj, sample);
    }
    return;
}




/* updating alpha0 */

void mzipBVS_restricted1_update_alpha0(gsl_matrix *X0,
                   gsl_vector *alpha0,
                   gsl_matrix *A,
                   gsl_matrix *W,
                   gsl_matrix *invR,
                   gsl_vector *phi,
                   double nu_t,
                   double sigSq_t,
                   double mu_alpha0,
                   double sigSq_alpha0)
{
    int i, j;
    
    int n = W -> size1;
    int q = W -> size2;
    
    double sum_phi = 0;
    
    gsl_matrix *Sigma_alpha0 =gsl_matrix_calloc(q, q);
    gsl_matrix *invSigma_alpha0 =gsl_matrix_calloc(q, q);
    gsl_matrix *scaled_invR =gsl_matrix_calloc(q, q);
    gsl_matrix *scaled_invR_sum =gsl_matrix_calloc(q, q);
    
    gsl_vector *mean_alpha0 = gsl_vector_calloc(q);
    
    gsl_vector *w_Ax0 = gsl_vector_calloc(q);
    gsl_vector *w_Ax0_sum = gsl_vector_calloc(q);
    gsl_vector *invR_w_Ax0 = gsl_vector_calloc(q);
    
    gsl_matrix *alpha0spl = gsl_matrix_calloc(1, q);
    
    for(j = 0; j < q; j++)
    {
        gsl_matrix_set(invSigma_alpha0, j, j, pow(sigSq_alpha0, -1));
    }
    
    for(i = 0; i < n; i++)
    {
        sum_phi += gsl_vector_get(phi, i);
    }
    
    gsl_matrix_memcpy(scaled_invR, invR);
    gsl_matrix_scale(scaled_invR, sum_phi*pow(sigSq_t, -1));
    gsl_matrix_add(invSigma_alpha0, scaled_invR);
    c_solve(invSigma_alpha0, Sigma_alpha0);
    
    for(i = 0; i < n; i++)
    {
        gsl_vector_view X0_row = gsl_matrix_row(X0, i);
        gsl_vector_view W_row = gsl_matrix_row(W, i);
        gsl_vector_memcpy(w_Ax0, &W_row.vector);
        gsl_blas_dgemv(CblasTrans, -1, A, &X0_row.vector, 1, w_Ax0);
        gsl_vector_scale(w_Ax0, gsl_vector_get(phi, i));
        gsl_vector_add(w_Ax0_sum, w_Ax0);
    }
    
    gsl_blas_dgemv(CblasNoTrans, 1, invR, w_Ax0_sum, 0, invR_w_Ax0);
    gsl_vector_scale(invR_w_Ax0, pow(sigSq_t, -1));
    gsl_blas_dgemv(CblasNoTrans, 1, Sigma_alpha0, invR_w_Ax0, 0, mean_alpha0);
    
    c_rmvnorm(alpha0spl, mean_alpha0, Sigma_alpha0);
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(alpha0, j, gsl_matrix_get(alpha0spl, 0, j));
    }
    
    gsl_matrix_free(Sigma_alpha0);
    gsl_matrix_free(invSigma_alpha0);
    gsl_matrix_free(scaled_invR);
    gsl_matrix_free(scaled_invR_sum);
    gsl_vector_free(mean_alpha0);
    gsl_vector_free(w_Ax0);
    gsl_vector_free(w_Ax0_sum);
    gsl_vector_free(invR_w_Ax0);
    gsl_matrix_free(alpha0spl);
    
    return;
}



/* Updating sigSq_alpha0 */

void mzipBVS_restricted1_update_sigSq_alpha0(gsl_vector *alpha0,
                         double *sigSq_alpha0,
                         double a_alpha0,
                         double b_alpha0)
{
    int j;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int q = alpha0 -> size;
    
    zeta_shape = a_alpha0 + (double) q/2;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(alpha0, j), 2);
    }
    zeta_rate /= 2;
    zeta_rate += b_alpha0;
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    *sigSq_alpha0 = pow(zeta, -1);
    
    return;
}


/* Updating sigSq_beta0 */

void mzipBVS_restricted1_update_sigSq_beta0(gsl_vector *beta0,
                        double *sigSq_beta0,
                        double a_beta0,
                        double b_beta0)
{
    int j;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int q = beta0 -> size;
    
    zeta_shape = a_beta0 + (double) q/2;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(beta0, j), 2);
    }
    zeta_rate /= 2;
    zeta_rate += b_beta0;
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    *sigSq_beta0 = pow(zeta, -1);
    
    return;
}


/* Updating sigSq_alpha */

void mzipBVS_restricted1_update_sigSq_alpha(gsl_matrix *A,
                        gsl_vector *sigSq_alpha,
                        gsl_vector *a_alpha,
                        gsl_vector *b_alpha)
{
    int j, kk;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int p0 = A -> size1;
    int q = A -> size2;
    
    kk = (int) runif(0, p0);
    
    gsl_vector *alpha = gsl_vector_calloc(q);
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(alpha, j, gsl_matrix_get(A, kk, j));
    }
    
    zeta_shape = gsl_vector_get(a_alpha, kk) + (double) q/2;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(alpha, j), 2);
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

void mzipBVS_restricted1_update_sigSq_beta(gsl_matrix *B,
                       gsl_vector *sigSq_beta,
                       gsl_vector *a_beta,
                       gsl_vector *b_beta)
{
    int j, kk;
    double zeta, zeta_rate, zeta_scale, zeta_shape;
    int p1 = B -> size1;
    int q = B -> size2;
    
    kk = (int) runif(0, p1);
    
    gsl_vector *beta = gsl_vector_calloc(q);
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(beta, j, gsl_matrix_get(B, kk, j));
    }
    
    zeta_shape = gsl_vector_get(a_beta, kk) + (double) q/2;
    
    zeta_rate = 0;
    
    for(j = 0; j < q; j++)
    {
        zeta_rate += pow(gsl_vector_get(beta, j), 2);
    }
    zeta_rate /= 2;
    zeta_rate += gsl_vector_get(b_beta, kk);
    zeta_scale = 1/zeta_rate;
    
    zeta = rgamma(zeta_shape, zeta_scale);
    gsl_vector_set(sigSq_beta, kk, pow(zeta, -1));
    
    gsl_vector_free(beta);
    
    return;
}


/* Updating phi */

void mzipBVS_restricted1_update_phi(gsl_matrix *X0,
                gsl_vector *alpha0,
                gsl_matrix *A,
                gsl_matrix *W,
                gsl_matrix *invR,
                gsl_vector *phi,
                double nu_t,
                double sigSq_t)
{
    int i, kk;
    double phi_rate, phi_scale, phi_shape, sample;
    int n = X0 -> size1;
    int q = A -> size2;
    
    gsl_vector *A_trans_X = gsl_vector_calloc(q);
    gsl_vector *W_alp0_AX = gsl_vector_calloc(q);
    gsl_vector *temp = gsl_vector_calloc(q);
    
    for(kk = 0; kk < 50; kk++)
    {
        i = (int) runif(0, n);
        
        gsl_vector_view X_row = gsl_matrix_row(X0, i);
        gsl_vector_view W_row = gsl_matrix_row(W, i);
        gsl_blas_dgemv(CblasTrans, 1, A, &X_row.vector, 0, A_trans_X);
        gsl_vector_memcpy(W_alp0_AX, &W_row.vector);
        gsl_vector_sub(W_alp0_AX, alpha0);
        gsl_vector_sub(W_alp0_AX, A_trans_X);
        gsl_blas_dgemv(CblasTrans, 1, invR, W_alp0_AX, 0, temp);
        gsl_blas_ddot(W_alp0_AX, temp, &phi_rate);
        phi_rate /= sigSq_t;
        phi_rate += nu_t;
        phi_rate /= 2;
        phi_scale = pow(phi_rate, -1);
        
        phi_shape = (nu_t + q) * 0.5;
        
        sample = rgamma(phi_shape, phi_scale);
        gsl_vector_set(phi, i, sample);
    }
    
    gsl_vector_free(A_trans_X);
    gsl_vector_free(W_alp0_AX);
    gsl_vector_free(temp);
    
    return;
}












