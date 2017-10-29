
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

extern double c_det(gsl_matrix *A);

extern void updateRPus(int *q_adj,
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
                     gsl_matrix *accept_B);

extern void updateIPus(gsl_matrix *Y,
                     gsl_matrix *X,
                     gsl_matrix *B,
                     gsl_matrix *Sigma,
                     gsl_matrix *invSigma,
                     gsl_vector *beta0,
                     gsl_vector *mu0,
                     double h0);

extern void updateCPus(int *q_adj,
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
                     int *accept_Sigma);

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

extern double sumCorus_j(gsl_matrix *Sigma,
                       gsl_matrix *gamma,
                       int j,
                       int k);

extern double logit(double val);

extern double invLogit(double val);

extern double one_invLogit(double val);

extern void c_riwishart(double v,
                        gsl_matrix *X_ori,
                        gsl_matrix *sample);

