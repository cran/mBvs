
extern void removeRowColumn(gsl_matrix *R, int i, gsl_matrix *R_sub);

extern void Get_subColumnVector(gsl_matrix *R, int i, gsl_vector *R_subvec);

extern void c_rtnorm(double mean,
                     double sd,
                     double LL,
                     double UL,
                     int LL_neginf,
                     int UL_posinf,
                     double *value);

extern void c_rmvnorm(gsl_matrix *sample,
                      gsl_vector *mean,
                      gsl_matrix *Var);

extern double logistic(double x);

extern void c_rowSums(gsl_matrix *X, gsl_vector *v);

extern void c_solve(gsl_matrix *M,
                    gsl_matrix *Minv);

extern double c_det(gsl_matrix *A);

extern double c_ldet(gsl_matrix *A);

extern int c_sgn_det(gsl_matrix *A);

extern double get_det(gsl_matrix *A);

extern void psd_chk(gsl_matrix *A, gsl_vector *check);

extern void c_riwishart(double v,
                        gsl_matrix *X_ori,
                        gsl_matrix *sample,
                        gsl_matrix *sampleInv);

extern void c_riwishart2(double v,
                         gsl_matrix *X_ori,
                         gsl_matrix *sample,
                         gsl_matrix *sampleInv);

extern double c_min(double value1,
                    double value2);

extern double c_max(double value1,
                    double value2);

extern void c_colSums(gsl_matrix *X, gsl_vector *v);

extern void matrixInv(gsl_matrix *X, gsl_matrix *Xinv);

extern void c_quadform_vMv(gsl_vector *v,
                           gsl_matrix *Minv,
                           double     *value);

extern void c_dmvnorm2(gsl_vector *x,
                       gsl_vector *mu,
                       double     sigma,
                       gsl_matrix *AInv,
                       double     *value);

extern void c_dmvnorm3(gsl_vector *x,
                       gsl_vector *mu,
                       double     sigma,
                       gsl_matrix *AInv,
                       double     *value);

extern void c_rigamma(double *temp,
                      double alpha,
                      double beta);

extern int c_multinom_sample(gsl_rng *rr,
                             gsl_vector *prob,
                             int length_prob);

extern void mzipBVS_general_updateRP_beta(int *p_adj,
                                          gsl_matrix *Y,
                                          gsl_matrix *X1,
                                          gsl_vector *xi,
                                          gsl_matrix *W,
                                          gsl_vector *beta0,
                                          gsl_matrix *B,
                                          gsl_matrix *V,
                                          gsl_matrix *gamma_beta,
                                          gsl_matrix *updateNonzB,
                                          gsl_vector *sigSq_beta,
                                          gsl_vector *v_beta,
                                          gsl_vector *omega_beta,
                                          double beta_prop_var,
                                          gsl_matrix *accept_B);

extern void mzipBVS_general_updateRP_alpha(int *p_adj,
                                           gsl_matrix *Y,
                                           gsl_matrix *X0,
                                           gsl_vector *alpha0,
                                           gsl_matrix *A,
                                           gsl_matrix *W,
                                           gsl_matrix *gamma_alpha,
                                           gsl_matrix *updateNonzA,
                                           gsl_matrix *invR,
                                           gsl_vector *sigSq_alpha,
                                           gsl_vector *phi,
                                           double nu_t,
                                           double sigSq_t,
                                           gsl_vector *v_alpha,
                                           gsl_vector *omega_alpha,
                                           double alpha_prop_var,
                                           gsl_matrix *accept_A);


extern void mzipBVS_general_update_SigmaV_MH(gsl_matrix *V,
                                             gsl_matrix *SigmaV,
                                             gsl_matrix *invSigmaV,
                                             gsl_matrix *Psi0,
                                             double rho0,
                                             double rho_s,
                                             int *accept_SigmaV);

extern void mzipBVS_general_update_SigmaV(gsl_matrix *V,
                                          gsl_matrix *SigmaV,
                                          gsl_matrix *invSigmaV,
                                          gsl_matrix *Psi0,
                                          double rho0);

extern void mzipBVS_general_update_R_Gibbs(gsl_matrix *X0,
                                           gsl_vector *alpha0,
                                           gsl_matrix *A,
                                           gsl_matrix *W,
                                           gsl_matrix *R,
                                           gsl_matrix *invR,
                                           gsl_vector *S,
                                           gsl_vector *phi,
                                           double sigSq_alpha0,
                                           double sigSq_t,
                                           gsl_matrix *PsiR,
                                           double rhoR);

extern void mzipBVS_general_updateA(gsl_matrix *Y,
                                    gsl_matrix *X0,
                                    gsl_vector *alpha0,
                                    gsl_matrix *A,
                                    gsl_matrix *W,
                                    gsl_matrix *invR,
                                    gsl_vector *sigSq_alpha,
                                    gsl_vector *phi,
                                    double nu_t,
                                    double sigSq_t);

extern void mzipBVS_general_update_W(gsl_matrix *Y,
                                     gsl_matrix *X0,
                                     gsl_matrix *X1,
                                     gsl_vector *xi,
                                     gsl_vector *alpha0,
                                     gsl_matrix *A,
                                     gsl_matrix *W,
                                     gsl_vector *beta0,
                                     gsl_matrix *B,
                                     gsl_matrix *V,
                                     gsl_matrix *R,
                                     gsl_matrix *invR,
                                     gsl_vector *phi,
                                     double nu_t,
                                     double sigSq_t);


extern void mzipBVS_general_update_sigSq_beta0(gsl_vector *beta0,
                                               double *sigSq_beta0,
                                               double a_beta0,
                                               double b_beta0);

extern void mzipBVS_general_update_sigSq_alpha(gsl_matrix *A,
                                               gsl_matrix *gamma_alpha,
                                               gsl_vector *sigSq_alpha,
                                               gsl_vector *v_alpha,
                                               gsl_vector *a_alpha,
                                               gsl_vector *b_alpha);

extern void mzipBVS_general_update_sigSq_alpha0_new(gsl_vector *alpha0,
                                                    gsl_matrix *invR,
                                                    double *sigSq_alpha0,
                                                    double a_alpha0,
                                                    double b_alpha0);

extern void mzipBVS_general_update_sigSq_beta(gsl_matrix *B,
                                              gsl_matrix *gamma_beta,
                                              gsl_vector *sigSq_beta,
                                              gsl_vector *v_beta,
                                              gsl_vector *a_beta,
                                              gsl_vector *b_beta);

extern void mzipBVS_general_update_V(gsl_matrix *Y,
                                     gsl_matrix *X1,
                                     gsl_vector *xi,
                                     gsl_matrix *W,
                                     gsl_vector *beta0,
                                     gsl_matrix *B,
                                     gsl_matrix *V,
                                     gsl_matrix *invSigmaV,
                                     gsl_matrix *accept_V,
                                     double V_prop_var);

extern void mzipBVS_general_update_beta0(gsl_matrix *Y,
                                         gsl_matrix *X1,
                                         gsl_vector *xi,
                                         gsl_matrix *W,
                                         gsl_vector *beta0,
                                         gsl_matrix *B,
                                         gsl_matrix *V,
                                         double mu_beta0,
                                         double sigSq_beta0,
                                         double beta0_prop_var,
                                         gsl_vector *accept_beta0);

extern void mzipBVS_general_update_alpha0_new(gsl_matrix *X0,
                                              gsl_vector *alpha0,
                                              gsl_matrix *A,
                                              gsl_matrix *W,
                                              gsl_matrix *R,
                                              gsl_vector *phi,
                                              double nu_t,
                                              double sigSq_t,
                                              double mu_alpha0,
                                              double sigSq_alpha0);

extern void mzipBVS_general_updateB(gsl_matrix *Y,
                                    gsl_matrix *X1,
                                    gsl_matrix *W,
                                    gsl_vector *beta0,
                                    gsl_matrix *B,
                                    gsl_matrix *V,
                                    gsl_vector *mu_beta,
                                    gsl_vector *sigSq_beta,
                                    double beta_prop_var,
                                    gsl_matrix *accept_B);





