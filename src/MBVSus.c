/*
 TO COMPILE USE THE CODE:

 R CMD SHLIB MBVS.c MBVS_Updates.c MBVS_Utilities.c -lgsl -lgslcblas
 
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_heapsort.h"

#include "R.h"
#include "Rmath.h"

#include "MBVSus.h"


/* */
void MBVSusmcmc(double Data[],
                    int *n,
                    int *p,
                    int *q,
                    int *q_adj,
                    double hyperParams[],
                    double startValues[],
                    double startB[],
                    double startGamma[],
                    double startSigma[],
                    double mcmcParams[],
                    double rwBetaVar[],
                    int *numReps,
                    int *thin,
                    double *burninPerc,      
                    double samples_beta0[],
                    double samples_B[],
                    double samples_gamma[],
                    double samples_Sigma[],
                    double samples_misc[])
{
    GetRNGstate();
    
    time_t now;    
    
    int i, j, M;
    
    /* Data */
    
    gsl_matrix *Y = gsl_matrix_calloc(*n, (*p));
    gsl_matrix *X = gsl_matrix_calloc(*n, (*q));
    for(i = 0; i < *n; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(Y, i, j, Data[(j* *n) + i]);
        }
        for(j = 0; j < *q; j++)
        {
            gsl_matrix_set(X, i, j, Data[((*p+j)* *n) + i]);
        }              
    }
    
    /* Hyperparameters */
    
    double eta      = hyperParams[0];
    double h0       = hyperParams[1];
    gsl_vector *mu0     = gsl_vector_calloc(*p);
    gsl_vector *v       = gsl_vector_calloc(*p);
    gsl_vector *omega   = gsl_vector_calloc(*q-*q_adj);
    gsl_matrix *Psi0    = gsl_matrix_calloc(*p, *p);
    
    for(j = 0; j < *p; j++)
    {
        gsl_vector_set(mu0, j, hyperParams[2+j]);
        gsl_vector_set(v, j, hyperParams[2+*p+j]);
    }
    
    for(j = 0; j < *q-*q_adj; j++)
    {
        gsl_vector_set(omega, j, hyperParams[2+*p+*p+j]);
    }
    
    double rho0 = hyperParams[2+*p+*p+*q-*q_adj];
    
    for(i = 0; i < *p; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(Psi0, j, i, hyperParams[2+*p+*p+*q-*q_adj+1+ i* *p + j]);
        }
    }
    
    /* varialbes for M-H algorithm */
    
    gsl_matrix *Psi_prop = gsl_matrix_calloc(*p, *p);
    
    
    for(i = 0; i < *p; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(Psi_prop, i, j, mcmcParams[(j* *p) + i]);
        }
    }
    
    double rho_prop = mcmcParams[*p * *p];
    
    gsl_matrix *rwbetaVar = gsl_matrix_calloc(*q, *p);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(rwbetaVar, i, j, rwBetaVar[(j* *q) + i]);
        }
    }
    
    /* Starting values */
        
    gsl_vector *beta0 = gsl_vector_calloc(*p);
    
    for(j = 0; j < *p; j++)
    {
        gsl_vector_set(beta0, j, startValues[j]);
    }
    
    gsl_matrix *B = gsl_matrix_calloc(*q, *p);
    gsl_matrix *gamma = gsl_matrix_calloc(*q, *p);
    gsl_matrix *Sigma = gsl_matrix_calloc(*p, *p);
    gsl_matrix *invSigma = gsl_matrix_calloc(*p, *p);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(B, i, j, startB[(j* *q) + i]);
            gsl_matrix_set(gamma, i, j, startGamma[(j* *q) + i]);
        }
    }
    
    for(i = 0; i < *p; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(Sigma, i, j, startSigma[(j* *p) + i]);
        }
    }
  
    /* Variables required for storage of samples */
    
    int StoreInx;
    
    gsl_matrix *accept_B = gsl_matrix_calloc(*q, *p);
    
    int accept_Sigma = 0;
    
    gsl_matrix *updateNonzB = gsl_matrix_calloc(*q, *p);
    gsl_matrix_memcpy(updateNonzB, B);
 
    c_solve(Sigma, invSigma);
    
    for(M = 0; M < *numReps; M++)
    {
        /* updating intercept: beta0 */

        updateIPus(Y, X, B, Sigma, invSigma, beta0, mu0, h0);

        /* updating regression parameter: B, gamma */

        updateRPus(q_adj, Y, X, B, gamma, Sigma, invSigma, updateNonzB, beta0, v, omega, eta, rwbetaVar, accept_B);

        /* updating variance-covariance matrix */
        
        updateCPus(q_adj, Y, X, B, gamma, Sigma, invSigma, beta0, omega, eta, Psi0, rho0, Psi_prop, rho_prop, &accept_Sigma);
    
        /* Storing posterior samples */
        
        if( ( (M+1) % *thin ) == 0 && (M+1) > (*numReps * *burninPerc))
        {
            StoreInx = (M+1)/(*thin)- (*numReps * *burninPerc)/(*thin);
            
            for(j = 0; j < *p; j++)
            {
                samples_beta0[(StoreInx - 1) * (*p) + j] = gsl_vector_get(beta0, j);
            }
            for(i = 0; i < *p; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    samples_B[(StoreInx - 1) * (*p * *q) + i * (*q) + j] = gsl_matrix_get(B, j, i);
                    samples_gamma[(StoreInx - 1) * (*p * *q) + i * (*q) + j]   = gsl_matrix_get(gamma, j, i);
                }
            }
            for(i = 0; i < *p; i++)
            {
                for(j = 0; j < *p; j++)
                {
                    samples_Sigma[(StoreInx - 1) * (*p * *p) + i * (*p) + j] = gsl_matrix_get(Sigma, j, i);
                }
            }
        }
        
        if(M == (*numReps - 1))
        {
            for(i = 0; i < *p; i++)
            {
                for(j = 0; j < *q; j++)
                {
                    samples_misc[i * (*q) + j] = (int) gsl_matrix_get(accept_B, j, i);
                }
            }
            samples_misc[*p * *q] = accept_Sigma;
        }
        
        if( ( (M+1) % 5000 ) == 0)
        {
            time(&now);
            
            Rprintf("iteration: %d: %s\n", M+1, ctime(&now));
            
            R_FlushConsole();
            R_ProcessEvents();
        }
    }
    
    PutRNGstate();
    return;
}






