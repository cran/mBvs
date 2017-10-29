
extern void matrixInv(gsl_matrix *X, gsl_matrix *Xinv);

extern void c_colSums(gsl_matrix *X, gsl_vector *v);

extern void c_rowSums(gsl_matrix *X, gsl_vector *v);

extern void c_repVec_Rowmat(gsl_vector *v, gsl_matrix *X);

extern void c_repVec_Colmat(gsl_vector *v, gsl_matrix *X);

extern double c_min(double value1,
                    double value2);

extern double c_max(double value1,
                    double value2);

extern void c_solve(gsl_matrix *M,
                    gsl_matrix *Minv);

extern void c_quadform_vMv(gsl_vector *v,
                           gsl_matrix *Minv,
                           double     *value);


extern void c_quadform_vMu(gsl_vector *v,
                           gsl_matrix *Minv,
                           gsl_vector *u,
                           double     *value);

extern void updateRPfa(int *q_adj,
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
                     gsl_matrix *accept_B);

extern void updateIPfa(gsl_matrix *Y,
                     gsl_matrix *X,
                     gsl_matrix *B,
                     gsl_vector *beta0,
                     double sigmaSq,
                     gsl_matrix *invSigmaLam,
                     gsl_vector *mu0,                     
                     double h0);


extern void updateVPfa(gsl_matrix *Y,
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
                     gsl_vector *muLam);

extern void updateCPfa(int *q_adj,
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
                     gsl_vector *accept_lambda);

extern void c_rigamma(double *sample,
                      double alpha,
                      double beta);

extern void c_rmvnorm(gsl_matrix *sample,
                      gsl_vector *mean,
                      gsl_matrix *Var);

extern void c_dmvnorm(gsl_vector *x,
                      gsl_vector *mu,
                      double     sigma,
                      gsl_matrix *AInv,
                      double     *value);

extern double sumCorfa_j(gsl_vector *lambda,
                       gsl_matrix *gamma,
                       int j,
                       int k);

extern double logit(double val);

extern double invLogit(double val);

extern double one_invLogit(double val);


