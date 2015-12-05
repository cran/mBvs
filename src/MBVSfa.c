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

#include "MBVSfa.h"

/* */
void MBVSfamcmc(double Data[],
                    int *n,
                    int *p,
                    int *q,
                    int *q_adj,
                    double hyperParams[],
                    double startValues[],
                    double startB[],
                    double startGamma[],
                    double mcmcParams[],
                    double rwBetaVar[],
                    int *numReps,
                    int *thin,
                    double *burninPerc,      
                    double samples_beta0[],
                    double samples_lambda[],
                    double samples_sigSq[],
                    double samples_B[],
                    double samples_gamma[],
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
    double hLam     = hyperParams[2];
    double nu0      = hyperParams[3];
    double sigSq0   = hyperParams[4];

    gsl_vector *mu0     = gsl_vector_calloc(*p);
    gsl_vector *muLam   = gsl_vector_calloc(*p);
    gsl_vector *v       = gsl_vector_calloc(*p);
    gsl_vector *omega   = gsl_vector_calloc(*q-*q_adj);
    
    for(j = 0; j < *p; j++)
    {
        gsl_vector_set(mu0, j, hyperParams[5+j]);
        gsl_vector_set(muLam, j, hyperParams[5+*p+j]);
        gsl_vector_set(v, j, hyperParams[5+*p+*p+j]);
    }
    for(j = 0; j < *q-*q_adj; j++)
    {
        gsl_vector_set(omega, j, hyperParams[5+*p+*p+*p+j]);
    }
    
    /* varialbes for M-H algorithm */
    
    double rwlambdaVar  = mcmcParams[0];
    int numLamUpdate = mcmcParams[3];
    
    gsl_matrix *rwbetaVar = gsl_matrix_calloc(*q, *p);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(rwbetaVar, i, j, rwBetaVar[(j* *q) + i]);
        }
    }
    
    /* Starting values */
    
    double sigmaSq  = startValues[0];
    
    gsl_vector *beta0 = gsl_vector_calloc(*p);
    gsl_vector *lambda = gsl_vector_calloc(*p);
    
    for(j = 0; j < *p; j++)
    {
        gsl_vector_set(beta0, j, startValues[1+j]);
        gsl_vector_set(lambda, j, startValues[*p+1+j]);
    }
    
    gsl_matrix *B = gsl_matrix_calloc(*q, *p);
    gsl_matrix *gamma = gsl_matrix_calloc(*q, *p);
    
    for(i = 0; i < *q; i++)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_matrix_set(B, i, j, startB[(j* *q) + i]);
            gsl_matrix_set(gamma, i, j, startGamma[(j* *q) + i]);
        }
    }
    
    /* Variables required for storage of samples */
    
    int StoreInx;
    
    gsl_matrix *accept_B = gsl_matrix_calloc(*q, *p);
    gsl_vector *accept_lambda = gsl_vector_calloc(*p);
    
    gsl_matrix *updateNonzB = gsl_matrix_calloc(*q, *p);
    gsl_matrix_memcpy(updateNonzB, B);
    
    gsl_matrix *SigmaLam = gsl_matrix_calloc(*p, *p);
    gsl_matrix_set_identity(SigmaLam);
    gsl_blas_dger(1, lambda, lambda, SigmaLam);
    
    gsl_matrix *invSigmaLam = gsl_matrix_calloc(*p, *p);
    c_solve(SigmaLam, invSigmaLam);
    
    for(M = 0; M < *numReps; M++)
    {
        /* updating lambda              */
        
        for(i = 0; i < numLamUpdate; i++)
        {
            updateCPfa(q_adj, Y, X, B, gamma, beta0, lambda, sigmaSq, SigmaLam, invSigmaLam, hLam, eta, omega, muLam, rwlambdaVar, accept_lambda);
        }
        
        /* updating residual variance : sigmaSq */
        
        updateVPfa(Y, X, B, beta0, lambda, &sigmaSq, invSigmaLam, h0, hLam, nu0, sigSq0, v, mu0, muLam);
        
        /* updating intercept: beta0 */

        updateIPfa(Y, X, B, beta0, sigmaSq, invSigmaLam, mu0, h0);

        /* updating regression parameter: B, gamma */
        
        updateRPfa(q_adj, Y, X, B, gamma, updateNonzB, beta0, lambda, sigmaSq, invSigmaLam, mu0, v, omega, h0, eta, rwbetaVar, accept_B);
        
        
        /* Storing posterior samples */
        
        if( ( (M+1) % *thin ) == 0 && (M+1) > (*numReps * *burninPerc))
        {
            StoreInx = (M+1)/(*thin)- (*numReps * *burninPerc)/(*thin);

            samples_sigSq[StoreInx - 1] = sigmaSq;
            
            for(j = 0; j < *p; j++)
            {
                samples_lambda[(StoreInx - 1) * (*p) + j] = gsl_vector_get(lambda, j);
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
            for(j = 0; j < *p; j++)
            {
                samples_misc[*p * *q + j] = (int) gsl_vector_get(accept_lambda, j);
            }
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






