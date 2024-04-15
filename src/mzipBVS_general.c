
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_eigen.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"
#include "mzipBVS_general.h"



/* */
void mzipBVS_general_mcmc(double Ymat[],
                 double Xmat0[],
                 double Xmat1[],
                 double offs[],
                 int *n,
                 int *q,
                 int *p0,
                 int *p1,
                 int *p_adj,
                 double hyperP[],
                 double mcmcP[],
                 double startValues[],
                 double startGamma_beta[],
                 double startGamma_alpha[],
                 int *numReps,
                 int *thin,
                 double *burninPerc,
                 double store[],
                 double samples_beta0[],
                 double samples_B[],
                 double samples_V[],
                 double samples_alpha0[],
                 double samples_A[],
                 double samples_gamma_beta[],
                 double samples_gamma_alpha[],
                 double samples_W[],
                 double samples_R[],
                 double samples_S[],
                 double samples_Sigma_V[],
                 double samples_sigSq_alpha0[],
                 double samples_sigSq_beta0[],
                 double samples_sigSq_alpha[],
                 double samples_sigSq_beta[],
                 double samples_misc[],
                          double nVupdate)
{
    GetRNGstate();
    time_t now;
    int i, j, M;
    
    const gsl_rng_type * TT;
    gsl_rng * rr;
    
    gsl_rng_env_setup();
    
    TT = gsl_rng_default;
    rr = gsl_rng_alloc(TT);
 
    /* Data */
    gsl_matrix *Y = gsl_matrix_calloc(*n, (*q));
    gsl_matrix *X0 = gsl_matrix_calloc(*n, (*p0));
    gsl_matrix *X1 = gsl_matrix_calloc(*n, (*p1));
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(Y, i, j, Ymat[(j* *n) + i]);
        }
        for(j = 0; j < *p0; j++)
        {
            gsl_matrix_set(X0, i, j, Xmat0[(j* *n) + i]);
        }
        for(j = 0; j < *p1; j++)
        {
            gsl_matrix_set(X1, i, j, Xmat1[(j* *n) + i]);
        }
    }
    
    gsl_vector *xi = gsl_vector_calloc(*n);
    
    for(i = 0; i < *n; i++)
    {
        gsl_vector_set(xi, i, offs[i]);
    }
    
    
    /* Hyperparameters */
    
    gsl_vector *muS = gsl_vector_calloc(*q);
    gsl_matrix *PsiS = gsl_matrix_calloc(*q, *q);
    
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(muS, j, hyperP[j]);
    }
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(PsiS, j, i, hyperP[*q + i* *q + j]);
        }
    }
    
    double mu_alpha0 = hyperP[*q + *q* *q];
    gsl_vector *mu_alpha = gsl_vector_calloc(*q);
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(mu_alpha, j, hyperP[*q + *q* *q + 1 + j]);
    }
    
    double mu_beta0 = hyperP[*q + *q* *q + 1 + *q];
    gsl_vector *mu_beta = gsl_vector_calloc(*q);
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(mu_beta, j, hyperP[*q + *q* *q + 1 + *q + 1 + j]);
    }
    
    double a_alpha0 = hyperP[*q + *q* *q + 1 + *q + 1 + *q];
    double b_alpha0 = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1];
    
    gsl_vector *a_alpha = gsl_vector_calloc(*p0);
    gsl_vector *b_alpha = gsl_vector_calloc(*p0);
    
    for(j = 0; j < *p0; j++)
    {
        gsl_vector_set(a_alpha, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + j]);
        gsl_vector_set(b_alpha, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + j]);
    }
    
    double a_beta0 = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0];
    double b_beta0 = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1];
    
    gsl_vector *a_beta = gsl_vector_calloc(*p1);
    gsl_vector *b_beta = gsl_vector_calloc(*p1);
    
    for(j = 0; j < *p1; j++)
    {
        gsl_vector_set(a_beta, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + j]);
        gsl_vector_set(b_beta, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + j]);
    }
    
    double nu_t     = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1];
    double sigSq_t  = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1];
    double rho0     = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1];
    
    gsl_matrix *Psi0 = gsl_matrix_calloc(*q, *q);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(Psi0, j, i, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + i* *q + j]);
        }
    }
    
    gsl_vector *v_beta = gsl_vector_calloc(*q);
    
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(v_beta, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + j]);
    }
    
    gsl_vector *omega_beta = gsl_vector_calloc(*p1-*p_adj);
    
    for(j = 0; j < (*p1-*p_adj); j++)
    {
        gsl_vector_set(omega_beta, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + *q + j]);
    }
    
    gsl_vector *v_alpha = gsl_vector_calloc(*q);
    
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(v_alpha, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + *q + *p1-*p_adj + j]);
    }
    
    gsl_vector *omega_alpha = gsl_vector_calloc(*p0-*p_adj);
    
    for(j = 0; j < (*p0-*p_adj); j++)
    {
        gsl_vector_set(omega_alpha, j, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + *q + *p1-*p_adj + *q + j]);
    }
    
    double rhoR     = hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + *q + *p1-*p_adj + *q + *p0-*p_adj];
    
    gsl_matrix *PsiR = gsl_matrix_calloc(*q, *q);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(PsiR, j, i, hyperP[*q + *q* *q + 1 + *q + 1 + *q + 1 + 1 + *p0 + *p0 + 1 + 1 + *p1 + *p1 + 1 + 1 + 1 + *q* *q + *q + *p1-*p_adj + *q + *p0-*p_adj + 1 + i* *q + j]);
        }
    }
    
    double beta0_prop_var = mcmcP[0];
    double beta_prop_var = mcmcP[1];
    double V_prop_var = mcmcP[2];
    double alpha_prop_var = mcmcP[4];


    /* Starting values */

    gsl_matrix *B = gsl_matrix_calloc(*p1, *q);
    gsl_matrix *A = gsl_matrix_calloc(*p0, *q);
    gsl_matrix *V = gsl_matrix_calloc(*n, *q);
    gsl_matrix *W = gsl_matrix_calloc(*n, *q);
    gsl_vector *beta0 = gsl_vector_calloc(*q);
    gsl_vector *alpha0 = gsl_vector_calloc(*q);
    gsl_matrix *R = gsl_matrix_calloc(*q, *q);
    gsl_matrix *invR = gsl_matrix_calloc(*q, *q);
    gsl_vector *S = gsl_vector_calloc(*q);
    gsl_vector *sigSq_alpha = gsl_vector_calloc(*p0);
    gsl_vector *sigSq_beta = gsl_vector_calloc(*p1);
    gsl_vector *phi = gsl_vector_calloc(*n);
    gsl_matrix *gamma_beta = gsl_matrix_calloc(*p1, *q);
    gsl_matrix *gamma_alpha = gsl_matrix_calloc(*p0, *q);
    
    gsl_matrix *SigmaV = gsl_matrix_calloc(*q, *q);
    gsl_matrix *invSigmaV = gsl_matrix_calloc(*q, *q);
    
    for(i = 0; i < *p1; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(B, i, j, startValues[(j* *p1) + i]);
        }
    }

    for(i = 0; i < *p0; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(A, i, j, startValues[(*p1 * *q + j* *p0) + i]);
        }
    }
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(V, i, j, startValues[(*p1 * *q+ *p0 * *q + j* *n) + i]);
        }
    }
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(W, i, j, startValues[(*p0 * *q+ *p1 * *q + *n * *q + j* *n) + i]);
        }
    }
    
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(beta0, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + j]);
        gsl_vector_set(alpha0, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + j]);
    }
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(R, i, j, startValues[(*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + j* *q) + i]);
        }
    }

    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(S, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + j]);
    }
    
    c_solve(R, invR);
    
    double sigSq_alpha0 = startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q];
    
    for(j = 0; j < *p0; j++)
    {
        gsl_vector_set(sigSq_alpha, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q + 1 + j]);
    }
    
    double sigSq_beta0 = startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q + 1 + *p0];
    
    for(j = 0; j < *p1; j++)
    {
        gsl_vector_set(sigSq_beta, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q + 1 + *p0 + 1 + j]);
    }
    for(j = 0; j < *n; j++)
    {
        gsl_vector_set(phi, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q + 1 + *p0 + 1 + *p1 + j]);
    }
    
    for(i = 0; i < *p1; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(gamma_beta, i, j, startGamma_beta[(j* *p1) + i]);
        }
    }
    
    for(i = 0; i < *p0; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(gamma_alpha, i, j, startGamma_alpha[(j* *p0) + i]);
        }
    }
    
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(SigmaV, i, j, startValues[*p1 * *q+ *p0 * *q + *n * *q + *n * *q + *q + *q + *q* *q + *q + 1 + *p0 + 1 + *p1 + *n + *p1* *q + *p0 * *q + j* *q + i]);
        }
    }
    
    c_solve(SigmaV, invSigmaV);
    
    /* Variables required for storage of samples */
    
    int StoreInx;
    gsl_matrix *accept_A = gsl_matrix_calloc(*p0, *q);
    gsl_matrix *accept_B = gsl_matrix_calloc(*p1, *q);
    gsl_matrix *accept_V = gsl_matrix_calloc(*n, *q);
    gsl_matrix *accept_W = gsl_matrix_calloc(*n, *q);
    gsl_vector *accept_alpha0 = gsl_vector_calloc(*q);
    gsl_vector *accept_beta0 = gsl_vector_calloc(*q);
    int accept_R = 0;
    int accept_SigmaV = 0;
    gsl_vector *accept_S = gsl_vector_calloc(*q);
    
    int storeV, storeW;
    
    storeV = (int) store[0];
    storeW = (int) store[1];
    
    gsl_matrix *updateNonzB = gsl_matrix_calloc(*p1, *q);
    gsl_matrix_memcpy(updateNonzB, B);
    gsl_matrix *updateNonzA = gsl_matrix_calloc(*p0, *q);
    gsl_matrix_memcpy(updateNonzA, A);
    


    for(M = 0; M < *numReps; M++)
    {
    
        /* updating beta0 */
         mzipBVS_general_update_beta0(Y, X1, xi, W, beta0, B, V, mu_beta0, sigSq_beta0, beta0_prop_var, accept_beta0);
        
        
        /* updating B and gamma_beta */
        mzipBVS_general_updateRP_beta(p_adj, Y, X1, xi, W, beta0, B, V, gamma_beta, updateNonzB, sigSq_beta, v_beta, omega_beta, beta_prop_var, accept_B);
        
        
        /* updating sigSq_beta0 */
        mzipBVS_general_update_sigSq_beta0(beta0, &sigSq_beta0, a_beta0, b_beta0);
        
        
        /* updating sigSq_beta */
        mzipBVS_general_update_sigSq_beta(B, gamma_beta, sigSq_beta, v_beta, a_beta, b_beta);
        
        
        /* updating V */
        mzipBVS_general_update_V(Y, X1, xi, W, beta0, B, V, invSigmaV, accept_V, V_prop_var);
        
        
        /* updating SigmaV */
        mzipBVS_general_update_SigmaV(V, SigmaV, invSigmaV, Psi0, rho0);
        
        
        /* updating alpha0*/
         mzipBVS_general_update_alpha0_new(X0, alpha0, A, W, R, phi, nu_t, sigSq_t, mu_alpha0, sigSq_alpha0);
         
        
        /* updating A and gamma_alpha*/
         mzipBVS_general_updateRP_alpha(p_adj, Y, X0, alpha0, A, W, gamma_alpha, updateNonzA, invR, sigSq_alpha, phi, nu_t, sigSq_t, v_alpha, omega_alpha, alpha_prop_var, accept_A);
        
        
        /* updating sigSq_alpha0 */
         mzipBVS_general_update_sigSq_alpha0_new(alpha0, invR, &sigSq_alpha0, a_alpha0, b_alpha0);
         
        
        /* updating sigSq_alpha*/
         mzipBVS_general_update_sigSq_alpha(A, gamma_alpha, sigSq_alpha, v_alpha, a_alpha, b_alpha);
         
        
        /* updating W */
         mzipBVS_general_update_W(Y, X0, X1, xi, alpha0, A, W, beta0, B, V, R, invR, phi, nu_t, sigSq_t);

        
        /* updating R */
        mzipBVS_general_update_R_Gibbs(X0, alpha0, A, W, R, invR, S, phi, sigSq_alpha0, sigSq_t, PsiR, rhoR);
        
         
        

        
        
        /* Storing posterior samples */
        
        if( ( (M+1) % *thin ) == 0 && (M+1) > (*numReps * *burninPerc))
        {
            StoreInx = (M+1)/(*thin)- (*numReps * *burninPerc)/(*thin);
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p0; j++)
                {
                    samples_A[(StoreInx - 1) * (*q * *p0) + i * (*p0) + j] = gsl_matrix_get(A, j, i);
                }
            }
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1; j++)
                {
                    samples_B[(StoreInx - 1) * (*q * *p1) + i * (*p1) + j] = gsl_matrix_get(B, j, i);
                }
            }
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1; j++)
                {
                    samples_gamma_beta[(StoreInx - 1) * (*q * *p1) + i * (*p1) + j] = gsl_matrix_get(gamma_beta, j, i);
                }
            }
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p0; j++)
                {
                    samples_gamma_alpha[(StoreInx - 1) * (*q * *p0) + i * (*p0) + j] = gsl_matrix_get(gamma_alpha, j, i);
                }
            }
            
            if(storeV == 1)
            {
                for(i = 0; i < *q; i++)
                {
                    for(j = 0; j < *n; j++)
                    {
                        samples_V[(StoreInx - 1) * (*q * *n) + i * (*n) + j] = gsl_matrix_get(V, j, i);
                    }
                }
            }
            
            if(storeW == 1)
            {
                for(i = 0; i < *q; i++)
                {
                    for(j = 0; j < *n; j++)
                    {
                        samples_W[(StoreInx - 1) * (*q * *n) + i * (*n) + j] = gsl_matrix_get(W, j, i);
                    }
                }
            }

            for(j = 0; j < *q; j++)
            {
                samples_beta0[(StoreInx - 1) * (*q) + j] = gsl_vector_get(beta0, j);
            }
            
            for(j = 0; j < *q; j++)
            {
                samples_alpha0[(StoreInx - 1) * (*q) + j] = gsl_vector_get(alpha0, j);
            }
            
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    samples_R[(StoreInx - 1) * (*q * *q) + i * (*q) + j] = gsl_matrix_get(R, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    samples_Sigma_V[(StoreInx - 1) * (*q * *q) + i * (*q) + j] = gsl_matrix_get(SigmaV, j, i);
                }
            }
            
            for(j = 0; j < *q; j++)
            {
                samples_S[(StoreInx - 1) * (*q) + j] = gsl_vector_get(S, j);
            }
            
            for(j = 0; j < *p0; j++)
            {
                samples_sigSq_alpha[(StoreInx - 1) * (*p0) + j] = gsl_vector_get(sigSq_alpha, j);
            }
            for(j = 0; j < *p1; j++)
            {
                samples_sigSq_beta[(StoreInx - 1) * (*p1) + j] = gsl_vector_get(sigSq_beta, j);
            }
            
            samples_sigSq_alpha0[StoreInx - 1] = sigSq_alpha0;
            samples_sigSq_beta0[StoreInx - 1] = sigSq_beta0;
        }
        
        if(M == (*numReps - 1))
        {
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p0; j++)
                {
                    samples_misc[i * (*p0) + j] = (int) gsl_matrix_get(accept_A, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1; j++)
                {
                    samples_misc[*q * *p0 + i * (*p1) + j] = (int) gsl_matrix_get(accept_B, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *n; j++)
                {
                    samples_misc[*q * *p0 + *q * *p1 + i * (*n) + j] = (int) gsl_matrix_get(accept_V, j, i); 
                }
            }
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *n; j++)
                {
                    samples_misc[*q * *p0 + *q * *p1 + *n * *q + i * (*n) + j] = (int) gsl_matrix_get(accept_W, j, i);
                }
            }
            for(i = 0; i < *q; i++)
            {
                samples_misc[*q * *p0 + *q * *p1 + *n * *q + (*n) * *q + i] = (int) gsl_vector_get(accept_alpha0, i);
            }
            for(i = 0; i < *q; i++)
            {
                samples_misc[*q * *p0 + *q * *p1 + *n * *q + (*n) * *q + *q + i] = (int) gsl_vector_get(accept_beta0, i);
            }

            samples_misc[*q * *p0 + *q * *p1 + *n * *q + (*n) * *q + *q + *q] = accept_R;
            
            for(i = 0; i < *q; i++)
            {
                samples_misc[*q * *p0 + *q * *p1 + *n * *q + (*n) * *q + *q + *q + 1 + i] = (int) gsl_vector_get(accept_S, i);
            }
            
            samples_misc[*q * *p0 + *q * *p1 + *n * *q + (*n) * *q + *q + *q + 1 + *q] = accept_SigmaV;
            
        }
        
        if( ( (M+1) % 1000 ) == 0)
        {
            time(&now);
            Rprintf("iteration: %d: %s\n", M+1, ctime(&now));
            R_FlushConsole();
            R_ProcessEvents();
        }
    }
    
    gsl_rng_free(rr);
    
    PutRNGstate();
    return;
}





















