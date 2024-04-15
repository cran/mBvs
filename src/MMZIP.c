
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


/* */
void MMZIPmcmc(double Ymat[],
               double Xmat0[],
               double Xmat1[],
               double offs[],
               int *n,
               int *q,
               int *q_allNZ,
               int *p0_all,
               int *p1_all,
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
               double VS[],
               double samples_B[],
               double samples_gamma_beta[],
               double samples_beta0[],
               double samples_A[],
               double samples_gamma_alpha[],
               double samples_alpha0[],
               double samples_V[],
               double samples_W[],
               double samples_SigmaV_diag[],
               double samples_m[],
               double samples_sigSq_beta[],
               double samples_sigSq_beta0[],
               double samples_sigSq_alpha[],
               double samples_sigSq_alpha0[],
               double samples_misc[],
               double logpost[],
               double SigV_mean[],
               double SigV_var[],
               double M_B_ini[],
               double M_beta0_ini[],
               double M_V_ini[],
               double M_A_ini[],
               double M_alpha0_ini[],
               double M_m_ini[],
               double final_SigmaV[],
               double final_V[],
               double final_W[],
               double final_eps[])
{
    GetRNGstate();
    time_t now;
    int i, j, k, M;
    
    const gsl_rng_type * TT;
    gsl_rng * rr;
    
    gsl_rng_env_setup();
    
    TT = gsl_rng_default;
    rr = gsl_rng_alloc(TT);
    
    /* Data */
    int q_bin = *q - *q_allNZ;
    
    gsl_matrix *Y = gsl_matrix_calloc(*n, *q);
    gsl_matrix *X0 = gsl_matrix_calloc(*n, (*p0_all));
    gsl_matrix *X1 = gsl_matrix_calloc(*n, (*p1_all));
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(Y, i, j, Ymat[(j* *n) + i]);
        }
        for(j = 0; j < *p0_all; j++)
        {
            gsl_matrix_set(X0, i, j, Xmat0[(j* *n) + i]);
        }
        for(j = 0; j < *p1_all; j++)
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
    
    gsl_vector *v_beta = gsl_vector_calloc(*q);
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(v_beta, j, hyperP[j]);
    }
    
    gsl_vector *omega_beta = gsl_vector_calloc(*p1_all-*p_adj);
    for(j = 0; j < (*p1_all-*p_adj); j++)
    {
        gsl_vector_set(omega_beta, j, hyperP[*q + j]);
    }
    
    gsl_vector *a_beta = gsl_vector_calloc(*p1_all);
    gsl_vector *b_beta = gsl_vector_calloc(*p1_all);
    for(j = 0; j < *p1_all; j++)
    {
        gsl_vector_set(a_beta, j, hyperP[*q + (*p1_all-*p_adj) + j]);
        gsl_vector_set(b_beta, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + j]);
    }
    
    gsl_vector *mu_beta0 = gsl_vector_calloc(*q);
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(mu_beta0, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + j]);
    }
    
    double a_beta0 = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q];
    double b_beta0 = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1];
    
    gsl_vector *v_alpha = gsl_vector_calloc(q_bin);
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(v_alpha, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + j]);
    }
    
    gsl_vector *omega_alpha = gsl_vector_calloc(*p0_all-*p_adj);
    for(j = 0; j < (*p0_all-*p_adj); j++)
    {
        gsl_vector_set(omega_alpha, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + j]);
    }
    
    gsl_vector *a_alpha = gsl_vector_calloc(*p0_all);
    gsl_vector *b_alpha = gsl_vector_calloc(*p0_all);
    for(j = 0; j < *p0_all; j++)
    {
        gsl_vector_set(a_alpha, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + j]);
        gsl_vector_set(b_alpha, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + j]);
    }
    
    gsl_vector *mu_alpha0 = gsl_vector_calloc(q_bin);
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(mu_alpha0, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + j]);
    }
    
    double a_alpha0 = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin];
    double b_alpha0 = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin + 1];
    
    double rho0 = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin + 1 + 1];
    
    gsl_matrix *Psi0 = gsl_matrix_calloc(*q, *q);
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(Psi0, j, i, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin + 1 + 1 + 1 + i* *q + j]);
        }
    }
    
    gsl_vector *mu_m = gsl_vector_calloc(q_bin);
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(mu_m, j, hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin + 1 + 1 + 1 + *q * *q + j]);
    }
    
    double v_m = hyperP[*q + (*p1_all-*p_adj) + *p1_all + *p1_all + *q + 1 + 1 + q_bin + (*p0_all-*p_adj) + *p0_all + *p0_all + q_bin + 1 + 1 + 1 + *q * *q + q_bin];
    
    
    /* MCMC parameters */
    
    int L_group = (int) mcmcP[0];
    double eps_group = mcmcP[1];
    /*
     double Mvar_group = mcmcP[2];
     */
    
    int L_V = (int) mcmcP[0];
    
    gsl_matrix *beta_prop_var = gsl_matrix_calloc(*p1_all, *q);
    for(i = 0; i < *p1_all; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(beta_prop_var, i, j, mcmcP[3]);
        }
    }
    /*
     int L_A = (int) mcmcP[4];
     */
    double eps_A = mcmcP[5];
    /*
     double Mvar_A = mcmcP[6];
     */
    
    gsl_matrix *alpha_prop_var = gsl_matrix_calloc(*p0_all, q_bin);
    for(i = 0; i < *p0_all; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            gsl_matrix_set(alpha_prop_var, i, j, mcmcP[7]);
        }
    }
    /*
     int L_alpha0 = (int) mcmcP[8];
     */
    double eps_alpha0 = mcmcP[9];
    /*
     double Mvar_alpha0 = mcmcP[10];
     */
    
    int L_m = (int) mcmcP[11];
    double eps_m = mcmcP[12];
    /*
     double Mvar_m = mcmcP[13];
     */
    double eps_V = mcmcP[14];
    double PtuneEps = mcmcP[15];
    
    
    
    /*
     double PtuneM = mcmcP[16];
     */
    
    int tuneM = (int) mcmcP[17];
    
    
    
    /*
     Rprintf("tuneM = %d\n", tuneM);
     Rprintf("P of tuning P_M = %0.7f\n", PtuneM);
     Rprintf("P of tuning P_Eps = %0.7f\n", PtuneEps);
     
     Rprintf("Num of tuning M = %d\n", (int) (*numReps * PtuneM));
     Rprintf("Num of tuning Eps = %d\n", (int) (*numReps * PtuneEps));
     
     Rprintf("start of tuning M = %d\n", (int) (*numReps * PtuneM / 2));
     */
    
    
    double pU_B_ssvs = 1;
    double pU_A_ssvs = 1;
    double pU_W = 0.2;
    
    gsl_matrix *M_B = gsl_matrix_calloc(*p1_all, *q);
    for(i = 0; i < *p1_all; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(M_B, i, j, M_B_ini[j* *p1_all + i]);
        }
    }
    
    gsl_vector *M_beta0 = gsl_vector_calloc(*q);
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(M_beta0, j, M_beta0_ini[j]);
    }
    
    gsl_matrix *M_V = gsl_matrix_calloc(*n, *q);
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(M_V, i, j, M_V_ini[j* *n + i]);
        }
    }
    
    gsl_matrix *M_A = gsl_matrix_calloc(*p0_all, q_bin);
    for(i = 0; i < *p0_all; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            gsl_matrix_set(M_A, i, j, M_A_ini[j* *p0_all + i]);
        }
    }
    
    gsl_vector *M_alpha0 = gsl_vector_calloc(q_bin);
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(M_alpha0, j, M_alpha0_ini[j]);
    }
    
    gsl_vector *M_m = gsl_vector_calloc(q_bin);
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(M_m, j, M_m_ini[j]);
    }
    
    /* Starting values */
    
    gsl_matrix *B = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *gamma_beta = gsl_matrix_calloc(*p1_all, *q);
    gsl_vector *beta0 = gsl_vector_calloc(*q);
    gsl_matrix *V = gsl_matrix_calloc(*n, *q);
    gsl_matrix *SigmaV = gsl_matrix_calloc(*q, *q);
    gsl_matrix *invSigmaV = gsl_matrix_calloc(*q, *q);
    gsl_matrix *cholSigmaV = gsl_matrix_calloc(*q, *q);
    
    gsl_matrix *A = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix *gamma_alpha = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_vector *alpha0 = gsl_vector_calloc(q_bin);
    gsl_matrix *W = gsl_matrix_calloc(*n, q_bin);
    gsl_vector *m_vec = gsl_vector_calloc(q_bin);
    gsl_matrix *invR = gsl_matrix_calloc(q_bin, q_bin);
    
    gsl_vector *sigSq_alpha = gsl_vector_calloc(*p0_all);
    gsl_vector *sigSq_beta = gsl_vector_calloc(*p1_all);
    
    for(i = 0; i < *p1_all; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(gamma_beta, i, j, startGamma_beta[(j* *p1_all) + i]);
        }
    }
    
    for(i = 0; i < *p0_all; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            gsl_matrix_set(gamma_alpha, i, j, startGamma_alpha[(j* *p0_all) + i]);
        }
    }
    
    for(i = 0; i < *p1_all; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(B, i, j, startValues[(j* *p1_all) + i]);
        }
    }
    
    for(i = 0; i < *p0_all; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            gsl_matrix_set(A, i, j, startValues[*p1_all**q + j* *p0_all + i]);
        }
    }
    
    for(j = 0; j < *q; j++)
    {
        gsl_vector_set(beta0, j, startValues[*p1_all**q + *p0_all*q_bin  + j]);
    }
    
    for(j = 0; j < q_bin; j++)
    {
        gsl_vector_set(alpha0, j, startValues[*p1_all**q + *p0_all*q_bin + *q + j]);
        gsl_vector_set(m_vec, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + j]);
    }
    
    double sigSq_alpha0 = startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin];
    
    for(j = 0; j < *p0_all; j++)
    {
        gsl_vector_set(sigSq_alpha, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + j]);
    }
    
    double sigSq_beta0 = startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + *p0_all];
    
    for(j = 0; j < *p1_all; j++)
    {
        gsl_vector_set(sigSq_beta, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + *p0_all + 1 + j]);
    }
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(V, i, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + *p0_all + 1 + *p1_all + j* *n + i]);
        }
    }
    
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < q_bin; j++)
        {
            gsl_matrix_set(W, i, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + *p0_all + 1 + *p1_all + *n**q + j**n + i]);
        }
    }
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(SigmaV, i, j, startValues[*p1_all**q + *p0_all*q_bin + *q + q_bin + q_bin + 1 + *p0_all + 1 + *p1_all + *n**q + q_bin**n + j* *q + i]);
        }
    }
    
    
    /* Variables required for storage of samples */
    
    int StoreInx;
    
    int accept_group = 0;
    int accept_V = 0;
    gsl_matrix *accept_B_ssvs = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *accept_A_ssvs = gsl_matrix_calloc(*p0_all, q_bin);
    /*
     gsl_matrix *accept_A_hmc = gsl_matrix_calloc(*p0_all, q_bin);
     int accept_alpha0 = 0;
     gsl_vector *accept_A_100 = gsl_vector_calloc(100);
     gsl_vector *accept_alpha0_100 = gsl_vector_calloc(100);
     */
    gsl_vector *accept_m100 = gsl_vector_calloc(100);
    gsl_vector *accept_group100 = gsl_vector_calloc(100);
    gsl_vector *accept_V100 = gsl_vector_calloc(100);
    int accept_m = 0;
    
    int n_group = 0;
    /*
     int n_A_total = 0;
     
     int n_alpha0 = 0;
     */
    int n_m = 0;
    gsl_matrix *n_B_ssvs = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *n_A_ssvs = gsl_matrix_calloc(*p0_all, q_bin);
    /*
     gsl_matrix *n_A_hmc = gsl_matrix_calloc(*p0_all, q_bin);
     */
    int storeV, storeW;
    
    storeV = (int) store[0];
    storeW = (int) store[1];
    
    int VSc, VSb;
    
    VSc = (int) VS[0];
    VSb = (int) VS[1];
    
    gsl_matrix *B_mean = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *B_var = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *B_mean_temp = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *B_var_temp = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *s_B = gsl_matrix_calloc(*p1_all, *q);
    
    gsl_vector *beta0_mean = gsl_vector_calloc(*q);
    gsl_vector *beta0_var = gsl_vector_calloc(*q);
    
    gsl_matrix *V_mean = gsl_matrix_calloc(*n, *q);
    gsl_matrix *V_var = gsl_matrix_calloc(*n, *q);
    
    gsl_matrix *A_mean = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix *A_var = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix *A_mean_temp = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix *A_var_temp = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix *s_A = gsl_matrix_calloc(*p0_all, q_bin);
    
    gsl_vector *alpha0_mean = gsl_vector_calloc(q_bin);
    gsl_vector *alpha0_var = gsl_vector_calloc(q_bin);
    
    gsl_vector *m_mean = gsl_vector_calloc(q_bin);
    gsl_vector *m_var = gsl_vector_calloc(q_bin);
    
    gsl_matrix *SigmaV_mean = gsl_matrix_calloc(*q, *q);
    gsl_matrix *SigmaV_var = gsl_matrix_calloc(*q, *q);
    
    gsl_matrix *updateNonzB = gsl_matrix_calloc(*p1_all, *q);
    gsl_matrix *updateNonzA = gsl_matrix_calloc(*p0_all, q_bin);
    gsl_matrix_memcpy(updateNonzB, B);
    gsl_matrix_memcpy(updateNonzA, A);
    
    c_solve(SigmaV, invSigmaV);
    c_solve_corFA1(m_vec, invR);
    
    gsl_matrix_memcpy(cholSigmaV, SigmaV);
    gsl_linalg_cholesky_decomp(cholSigmaV);
    
    for(i = 0; i < *q; i ++)
    {
        for(j = 0; j < i; j ++)
        {
            gsl_matrix_set(cholSigmaV, i, j, 0);
        }
    }
    
    gsl_vector *sum_X0sq = gsl_vector_calloc(*p0_all);
    for(j = 0; j < *p0_all; j++)
    {
        for(i = 0; i < *n; i++)
        {
            gsl_vector_set(sum_X0sq, j, gsl_vector_get(sum_X0sq, j) + pow(gsl_matrix_get(X0, i, j), 2));
        }
    }
    
    double logPost_val;
    
    
    for(M = 0; M < *numReps; M++)
    {
        /* updating B, beta0, and V*/
        update_group_mmzip(p_adj, Y, X1, X0, sum_X0sq, xi, beta0, B, V, gamma_beta, alpha0, A, W, gamma_alpha, updateNonzB, sigSq_beta, v_beta, mu_beta0, sigSq_beta0, updateNonzA, sigSq_alpha, v_alpha, mu_alpha0, sigSq_alpha0, invSigmaV, cholSigmaV, invR, &accept_group, &accept_V, accept_group100, accept_V100, &eps_group, &eps_V, L_group, L_V, &n_group, numReps, burninPerc, M, M_B, M_beta0, M_V, M_A, M_alpha0, PtuneEps);
        
        /* updating B and gamma_beta -- SSVS*/
        if(VSc == 1)
        {
            updateRP_beta_mmzip_SSVS(p_adj, Y, X1, X0, xi, beta0, B, V, gamma_beta, alpha0, A, W, updateNonzB, sigSq_beta, v_beta, omega_beta, beta_prop_var, accept_B_ssvs, n_B_ssvs, pU_B_ssvs);
        }
        
        /* updating SigmaV*/
        update_SigmaV_mmzip(V, SigmaV, invSigmaV, cholSigmaV, Psi0, rho0);
        
        /* updating sigSq_beta0 */
        update_sigSq_beta0_mmzip(beta0, &sigSq_beta0, mu_beta0, a_beta0, b_beta0);
        
        /* updating sigSq_beta */
        update_sigSq_beta_mmzip(B, gamma_beta, sigSq_beta, v_beta, a_beta, b_beta);
        
        if(runif(0, 1) < 1.0)
        {
            /* updating W*/
            update_W_mmzip(Y, X1, X0, xi, beta0, B, V, alpha0, A, W, m_vec, invR, pU_W);
            
            
            /* updating m*/
            update_m_mmzip(X0, alpha0, A, W, m_vec, invR, mu_m, v_m, &accept_m, accept_m100, &eps_m, L_m, &n_m, numReps, burninPerc, M, M_m, PtuneEps);
            
            
            /* updating A and gamma_alpha -- SSVS*/
            if(VSb == 1)
            {
                updateRP_alpha_mmzip_SSVS(p_adj, Y, X1, X0, xi, beta0, B, V, alpha0, A, W, invR, gamma_alpha, updateNonzA, sigSq_alpha, v_alpha, omega_alpha, alpha_prop_var, accept_A_ssvs, n_A_ssvs, pU_A_ssvs);
            }
            
            /* updating sigSq_alpha0 */
            update_sigSq_alpha0_mmzip(alpha0, &sigSq_alpha0, mu_alpha0, a_alpha0, b_alpha0);
            
            /* updating sigSq_alpha */
            update_sigSq_alpha_mmzip(A, gamma_alpha, sigSq_alpha, v_alpha, a_alpha, b_alpha);
        }
        
        
        if(tuneM == 1)
        {
            if(M >= 1)
            {
                gsl_matrix_memcpy(B_var_temp, B_var);
                new_var_mat2(B_var, B_mean, s_B, B);
                for(j = 0; j < *q; j++)
                {
                    for(k = 0; k < *p1_all; k++)
                    {
                        if(gsl_matrix_get(gamma_beta, k, j) == 0)
                        {
                            gsl_matrix_set(B_var, k, j, gsl_matrix_get(B_var_temp, k, j));
                        }
                    }
                }
                new_var_vec(beta0_var, beta0_mean, M, beta0);
                new_var_mat(V_var, V_mean, M, V);
                
                gsl_matrix_memcpy(A_var_temp, A_var);
                new_var_mat2(A_var, A_mean, s_A, A);
                for(j = 0; j < q_bin; j++)
                {
                    for(k = 0; k < *p0_all; k++)
                    {
                        if(gsl_matrix_get(gamma_alpha, k, j) == 0)
                        {
                            gsl_matrix_set(A_var, k, j, gsl_matrix_get(A_var_temp, k, j));
                        }
                    }
                }
                new_var_vec(alpha0_var, alpha0_mean, M, alpha0);
                new_var_vec(m_var, m_mean, M, m_vec);
            }
            
            gsl_matrix_memcpy(B_mean_temp, B_mean);
            new_mean_mat2(B_mean, s_B, B);
            for(j = 0; j < *q; j++)
            {
                for(k = 0; k < *p1_all; k++)
                {
                    if(gsl_matrix_get(gamma_beta, k, j) == 0)
                    {
                        gsl_matrix_set(B_mean, k, j, gsl_matrix_get(B_mean_temp, k, j));
                    }
                }
            }
            new_mean_mat(V_mean, M, V);
            new_mean_vec(beta0_mean, M, beta0);
            gsl_matrix_memcpy(A_mean_temp, A_mean);
            new_mean_mat2(A_mean, s_A, A);
            for(j = 0; j < q_bin; j++)
            {
                for(k = 0; k < *p0_all; k++)
                {
                    if(gsl_matrix_get(gamma_alpha, k, j) == 0)
                    {
                        gsl_matrix_set(A_mean, k, j, gsl_matrix_get(A_mean_temp, k, j));
                    }
                }
            }
            new_mean_vec(alpha0_mean, M, alpha0);
            new_mean_vec(m_mean, M, m_vec);
            
            gsl_matrix_add(s_B, gamma_beta);
            gsl_matrix_add(s_A, gamma_alpha);
            
            
            if(M == (int) *numReps * 0.4 - 1)
            {
                /*
                 
                 Rprintf("acceptance rate for group = %0.5f\n", (double) accept_group/n_group);
                 Rprintf("M = %d, Mass matrices are tuned\n", M);
                 Rprintf("M = %d, Mass matrices are tuned\n", M);
                 */
                
                for(i = 0; i < *n; i++)
                {
                    for(j = 0; j < *q; j++)
                    {
                        if(gsl_matrix_get(V_var, i, j) > 0)
                        {
                            gsl_matrix_set(M_V, i, j, pow(gsl_matrix_get(V_var, i, j), -1));
                        }
                    }
                }
                
                for(i = 0; i < *p1_all; i++)
                {
                    for(j = 0; j < *q; j++)
                    {
                        if(gsl_matrix_get(s_B, i, j) > 10)
                        {
                            gsl_matrix_set(M_B, i, j, pow(gsl_matrix_get(B_var, i, j), -1));
                        }
                    }
                }
                
                for(j = 0; j < *q; j++)
                {
                    if(gsl_vector_get(beta0_var, j) > 0)
                    {
                        gsl_vector_set(M_beta0, j, pow(gsl_vector_get(beta0_var, j), -1));
                    }
                }
                
                for(i = 0; i < *p0_all; i++)
                {
                    for(j = 0; j < q_bin; j++)
                    {
                        if(gsl_matrix_get(s_A, i, j) > 10)
                        {
                            gsl_matrix_set(M_A, i, j, pow(gsl_matrix_get(A_var, i, j), -1));
                        }
                    }
                }
                
                for(j = 0; j < q_bin; j++)
                {
                    if(gsl_vector_get(alpha0_var, j) > 0)
                    {
                        gsl_vector_set(M_alpha0, j, pow(gsl_vector_get(alpha0_var, j), -1));
                    }
                }
                
                for(j = 0; j < q_bin; j++)
                {
                    if(gsl_vector_get(m_var, j) > 0)
                    {
                        gsl_vector_set(M_m, j, pow(gsl_vector_get(m_var, j), -1));
                    }
                }
                
            }
        }
        
        
        
        
        
        
        
        /* Storing posterior samples */
        
        if( ( (M+1) % *thin ) == 0 && (M+1) > (*numReps * *burninPerc))
        {
            StoreInx = (M+1)/(*thin)- (*numReps * *burninPerc)/(*thin);
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1_all; j++)
                {
                    samples_B[(StoreInx - 1) * (*q * *p1_all) + i * (*p1_all) + j] = gsl_matrix_get(B, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1_all; j++)
                {
                    samples_gamma_beta[(StoreInx - 1) * (*q * *p1_all) + i * (*p1_all) + j] = gsl_matrix_get(gamma_beta, j, i);
                }
            }
            
            
            for(i = 0; i < q_bin; i++)
            {
                for(j = 0; j < *p0_all; j++)
                {
                    samples_A[(StoreInx - 1) * (q_bin * *p0_all) + i * (*p0_all) + j] = gsl_matrix_get(A, j, i);
                }
            }
            
            for(i = 0; i < q_bin; i++)
            {
                for(j = 0; j < *p0_all; j++)
                {
                    samples_gamma_alpha[(StoreInx - 1) * (q_bin * *p0_all) + i * (*p0_all) + j] = gsl_matrix_get(gamma_alpha, j, i);
                }
            }
            
            if(storeV == 1 || *q < 10)
            {
                for(i = 0; i < *q; i++)
                {
                    for(j = 0; j < *n; j++)
                    {
                        samples_V[(StoreInx - 1) * (*q * *n) + i * (*n) + j] = gsl_matrix_get(V, j, i);
                    }
                }
            }else
            {
                for(i = 0; i < 10; i++)
                {
                    for(j = 0; j < 20; j++)
                    {
                        samples_V[(StoreInx - 1) * (10 * 20) + i * 20 + j] = gsl_matrix_get(V, j, i);
                    }
                }
            }
            
            if(storeW == 1 || q_bin < 10)
            {
                for(i = 0; i < q_bin; i++)
                {
                    for(j = 0; j < *n; j++)
                    {
                        samples_W[(StoreInx - 1) * (q_bin * *n) + i * (*n) + j] = gsl_matrix_get(W, j, i);
                    }
                }
            }else
            {
                for(i = 0; i < 10; i++)
                {
                    for(j = 0; j < 20; j++)
                    {
                        samples_W[(StoreInx - 1) * (10 * 20) + i * 20 + j] = gsl_matrix_get(W, j, i);
                    }
                }
            }
            
            for(j = 0; j < *q; j++)
            {
                samples_beta0[(StoreInx - 1) * (*q) + j] = gsl_vector_get(beta0, j);
            }
            
            for(j = 0; j < q_bin; j++)
            {
                samples_alpha0[(StoreInx - 1) * (q_bin) + j] = gsl_vector_get(alpha0, j);
            }
            
            for(j = 0; j < *q; j++)
            {
                samples_SigmaV_diag[(StoreInx - 1) * (*q) + j] = gsl_matrix_get(SigmaV, j, j);
            }
            
            for(j = 0; j < q_bin; j++)
            {
                samples_m[(StoreInx - 1) * (q_bin) + j] = gsl_vector_get(m_vec, j);
            }
            
            
            
            if(StoreInx-1 >= 1)
            {
                new_var_mat(SigmaV_var, SigmaV_mean, StoreInx-1, SigmaV);
            }
            
            new_mean_mat(SigmaV_mean, StoreInx, SigmaV);
            
            samples_sigSq_alpha0[StoreInx - 1] = sigSq_alpha0;
            samples_sigSq_beta0[StoreInx - 1] = sigSq_beta0;
            
            for(j = 0; j < *p0_all; j++)
            {
                samples_sigSq_alpha[(StoreInx - 1) * (*p0_all) + j] = gsl_vector_get(sigSq_alpha, j);
            }
            for(j = 0; j < *p1_all; j++)
            {
                samples_sigSq_beta[(StoreInx - 1) * (*p1_all) + j] = gsl_vector_get(sigSq_beta, j);
            }
            
            Cal_logPost_mmzip(p_adj, Y, X1, X0, xi, beta0, B, V, gamma_beta, alpha0, A, W, invSigmaV, invR, gamma_alpha, sigSq_beta, v_beta, omega_beta, sigSq_alpha, v_alpha, omega_alpha, a_beta, b_beta, a_alpha, b_alpha, mu_beta0, sigSq_beta0, a_beta0, b_beta0, mu_alpha0, sigSq_alpha0, a_alpha0, b_alpha0, m_vec, mu_m, v_m, &logPost_val);
            
            logpost[StoreInx - 1] = logPost_val;
        }
        
        if(M == (*numReps - 1))
        {
            samples_misc[0] = (double) accept_group/n_group;
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1_all; j++)
                {
                    samples_misc[1 + i * (*p1_all) + j] = (int) gsl_matrix_get(accept_B_ssvs, j, i) / gsl_matrix_get(n_B_ssvs, j, i);
                }
            }
            /*
             samples_misc[1 + *p1_all * *q] = (double) accept_alpha0/n_alpha0;
             */
            
            for(i = 0; i < q_bin; i++)
            {
                for(j = 0; j < *p0_all; j++)
                {
                    samples_misc[1 + *p1_all * *q + 1 + i * (*p0_all) + j] = (int) gsl_matrix_get(accept_A_ssvs, j, i) / gsl_matrix_get(n_A_ssvs, j, i);
                }
            }
            /*
             for(i = 0; i < q_bin; i++)
             {
             for(j = 0; j < *p0_all; j++)
             {
             samples_misc[1 + *p1_all * *q + 1 + *p0_all * q_bin + i * (*p0_all) + j] = (int) gsl_matrix_get(accept_A_hmc, j, i) / gsl_matrix_get(n_A_hmc, j, i);
             }
             }
             */
            samples_misc[1 + *p1_all * *q + 1 + *p0_all * q_bin + *p0_all * q_bin] = (double) accept_m/n_m;
            
            samples_misc[1 + *p1_all * *q + 1 + *p0_all * q_bin + *p0_all * q_bin + 1] = (double) accept_V/n_group;
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    SigV_mean[i * (*q) + j] = gsl_matrix_get(SigmaV_mean, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    SigV_var[i * (*q) + j] = gsl_matrix_get(SigmaV_var, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *p1_all; j++)
                {
                    M_B_ini[i * (*p1_all) + j] = gsl_matrix_get(M_B, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                M_beta0_ini[i] = gsl_vector_get(M_beta0, i);
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *n; j++)
                {
                    M_V_ini[i * (*n) + j] = gsl_matrix_get(M_V, j, i);
                }
            }
            
            for(i = 0; i < q_bin; i++)
            {
                for(j = 0; j < *p0_all; j++)
                {
                    M_A_ini[i * (*p0_all) + j] = gsl_matrix_get(M_A, j, i);
                }
            }
            
            for(i = 0; i < q_bin; i++)
            {
                M_alpha0_ini[i] = gsl_vector_get(M_alpha0, i);
            }
            
            for(i = 0; i < q_bin; i++)
            {
                M_m_ini[i] = gsl_vector_get(M_m, i);
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    final_SigmaV[i * (*q) + j] = gsl_matrix_get(SigmaV, j, i);
                }
            }
            
            for(i = 0; i < *q; i++)
            {
                for(j = 0; j < *n; j++)
                {
                    final_V[i * (*n) + j] = gsl_matrix_get(V, j, i);
                }
            }
            for(i = 0; i < q_bin; i++)
            {
                for(j = 0; j < *n; j++)
                {
                    final_W[i * (*n) + j] = gsl_matrix_get(W, j, i);
                }
            }
            
            final_eps[0] = eps_group;
            final_eps[1] = eps_A;
            final_eps[2] = eps_alpha0;
            final_eps[3] = eps_m;
            final_eps[4] = eps_V;
        }
        
        if( ( (M+1) % 2000 ) == 0)
        {
            time(&now);
            Rprintf("MCMC sampling: %d out of %d iterations: %s\n", M+1, *numReps, ctime(&now));
            R_FlushConsole();
            R_ProcessEvents();
        }
    }
    
    gsl_rng_free(rr);
    
    PutRNGstate();
    return;
}





















