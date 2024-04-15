
extern void Hessian_A_alpha0_mmzip(gsl_matrix *XA,
                                   gsl_matrix *X0,
                                   gsl_vector *alpha0,
                                   gsl_vector *sum_X0sq,
                                   gsl_matrix *A,
                                   gsl_matrix *W,
                                   gsl_matrix *Y,
                                   gsl_matrix *LamSI,
                                   gsl_matrix *invR,
                                   gsl_matrix *APriorV,
                                   double sigSq_alpha0,
                                   gsl_vector *M_alpha0,
                                   gsl_matrix *M_A);

extern void Hessian_alpha0_mmzip2(gsl_matrix *LamSI,
                                 gsl_matrix *invR,
                                 double sigSq_alpha0,
                                 gsl_vector *M_alpha0);

extern void Hessian_B_mmzip(gsl_matrix *LamSI,
                            gsl_matrix *X1,
                            gsl_matrix *BPriorV,
                            gsl_matrix *M_B);

extern void Hessian_V_mmzip(gsl_matrix *LamSI,
                            gsl_matrix *invSigmaV,
                            gsl_matrix *M_V);

extern void Hessian_beta0_mmzip(gsl_matrix *LamSI,
                                double sigSq_beta0,
                                gsl_vector *M_beta0);

extern void new_var_mat2(gsl_matrix *var,
                         gsl_matrix *oldmean,
                         gsl_matrix *s_Param,
                         gsl_matrix *newObs);

extern void new_mean_mat2(gsl_matrix *mean,
                          gsl_matrix *s_Param,
                          gsl_matrix *newObs);


extern void new_mean_vec(gsl_vector *mean,
                         int n_old,
                         gsl_vector *newObs);

extern void new_var_vec(gsl_vector *var,
                        gsl_vector *oldmean,
                        int n_old,
                        gsl_vector *newObs);

extern void new_mean_mat(gsl_matrix *mean,
                         int n_old,
                         gsl_matrix *newObs);

extern void new_var_mat(gsl_matrix *var,
                        gsl_matrix *oldmean,
                        int n_old,
                        gsl_matrix *newObs);


extern void Grad_A_mmzip(gsl_matrix *YI_LamSI,
                         gsl_matrix *X0,
                         gsl_matrix *FI,
                         gsl_matrix *W_XAinvR,
                         gsl_vector *invRalpha0,
                         gsl_matrix *A,
                         gsl_matrix *gamma_alpha,
                         gsl_matrix *APriorV,
                         gsl_matrix *Delta);

extern void Grad_alpha0_mmzip(gsl_matrix *YI_LamSI,
                              gsl_matrix *FI,
                              gsl_vector *alpha0,
                              gsl_matrix *W_XAinvR,
                              gsl_vector *invRalpha0Trans,
                              gsl_vector *mu_alpha0,
                              double sigSq_alpha0,
                              gsl_vector *Delta);


extern void LH_all_mmzip(gsl_matrix *Y,
                         gsl_matrix *logLamSI,
                         gsl_matrix *LamSI,
                         gsl_matrix *XA,
                         gsl_vector *alpha0,
                         gsl_matrix *W,
                         gsl_matrix *invR,
                         double *logLH_val);

extern void Cal_FI_mmzip(gsl_matrix *XA,
                         gsl_vector *alpha0,
                         gsl_matrix *W,
                         gsl_matrix *FI);


extern void Cal_LamSI_mmzip(gsl_matrix *XB,
                            gsl_matrix *XA,
                            gsl_vector *xi,
                            gsl_vector *beta0,
                            gsl_matrix *V,
                            gsl_vector *alpha0,
                            gsl_matrix *W,
                            gsl_matrix *LamSI,
                            gsl_matrix *logLamSI);


extern void update_SigmaV_mmzip(gsl_matrix *V,
                                gsl_matrix *SigmaV,
                                gsl_matrix *invSigmaV,
                                gsl_matrix *cholSigmaV,
                                gsl_matrix *Psi0,
                                double rho0);

extern void Grad_B_mmzip(gsl_matrix *YI_LamSI,
                         gsl_matrix *X1,
                         gsl_matrix *B,
                         gsl_matrix *gamma_beta,
                         gsl_matrix *BPriorV,
                         gsl_matrix *Delta);


extern void update_group_mmzip(int *p_adj,
                               gsl_matrix *Y,
                               gsl_matrix *X1,
                               gsl_matrix *X0,
                               gsl_vector *sum_X0sq,
                               gsl_vector *xi,
                               gsl_vector *beta0,
                               gsl_matrix *B,
                               gsl_matrix *V,
                               gsl_matrix *gamma_beta,
                               gsl_vector *alpha0,
                               gsl_matrix *A,
                               gsl_matrix *W,
                               gsl_matrix *gamma_alpha,
                               gsl_matrix *updateNonzB,
                               gsl_vector *sigSq_beta,
                               gsl_vector *v_beta,
                               gsl_vector *mu_beta0,
                               double sigSq_beta0,
                               gsl_matrix *updateNonzA,
                               gsl_vector *sigSq_alpha,
                               gsl_vector *v_alpha,
                               gsl_vector *mu_alpha0,
                               double sigSq_alpha0,
                               gsl_matrix *invSigmaV,
                               gsl_matrix *cholSigmaV,
                               gsl_matrix *invR,
                               int *accept_group,
                               int *accept_V,
                               gsl_vector *accept_group100,
                               gsl_vector *accept_V100,
                               double *eps_group,
                               double *eps_V,
                               int L_group,
                               int L_V,
                               int *n_group,
                               int *numReps,
                               double *burninPerc,
                               int M,
                               gsl_matrix *M_B,
                               gsl_vector *M_beta0,
                               gsl_matrix *M_V,
                               gsl_matrix *M_A,
                               gsl_vector *M_alpha0,
                               double PtuneEps);

extern void Grad_V_mmzip(gsl_matrix *YI_LamSI,
                         gsl_matrix *V,
                         gsl_matrix *W,
                         gsl_matrix *invSigmaV,
                         gsl_matrix *Delta);


extern void Grad_beta0_mmzip(gsl_matrix *YI_LamSI,
                             gsl_vector *beta0,
                             gsl_vector *mu_beta0,
                             double sigSq_beta0,
                             gsl_vector *Delta);

extern void LH_count_mmzip(gsl_matrix *Y,
                           gsl_matrix *XB,
                           gsl_matrix *Fixed,
                           gsl_vector *beta0,
                           gsl_matrix *V,
                           gsl_matrix *W,
                           double *logLH_val);



extern void updateRP_beta_mmzip_SSVS(int *p_adj,
                                     gsl_matrix *Y,
                                     gsl_matrix *X1,
                                     gsl_matrix *X0,
                                     gsl_vector *xi,
                                     gsl_vector *beta0,
                                     gsl_matrix *B,
                                     gsl_matrix *V,
                                     gsl_matrix *gamma_beta,
                                     gsl_vector *alpha0,
                                     gsl_matrix *A,
                                     gsl_matrix *W,
                                     gsl_matrix *updateNonzB,
                                     gsl_vector *sigSq_beta,
                                     gsl_vector *v_beta,
                                     gsl_vector *omega_beta,
                                     gsl_matrix *beta_prop_var,
                                     gsl_matrix *accept_B,
                                     gsl_matrix *n_B,
                                     double pU_B_ssvs);


extern void Cal_logPost_mmzip(int *p_adj,
                              gsl_matrix *Y,
                              gsl_matrix *X1,
                              gsl_matrix *X0,
                              gsl_vector *xi,
                              gsl_vector *beta0,
                              gsl_matrix *B,
                              gsl_matrix *V,
                              gsl_matrix *gamma_beta,
                              gsl_vector *alpha0,
                              gsl_matrix *A,
                              gsl_matrix *W,
                              gsl_matrix *invSigmaV,
                              gsl_matrix *invR,
                              gsl_matrix *gamma_alpha,
                              gsl_vector *sigSq_beta,
                              gsl_vector *v_beta,
                              gsl_vector *omega_beta,
                              gsl_vector *sigSq_alpha,
                              gsl_vector *v_alpha,
                              gsl_vector *omega_alpha,
                              gsl_vector *a_beta,
                              gsl_vector *b_beta,
                              gsl_vector *a_alpha,
                              gsl_vector *b_alpha,
                              gsl_vector *mu_beta0,
                              double sigSq_beta0,
                              double a_beta0,
                              double b_beta0,
                              gsl_vector *mu_alpha0,
                              double sigSq_alpha0,
                              double a_alpha0,
                              double b_alpha0,
                              gsl_vector *m_vec,
                              gsl_vector *mu_m,
                              double v_m,
                              double *logPost_val);


extern void update_sigSq_beta_mmzip(gsl_matrix *B,
                                    gsl_matrix *gamma_beta,
                                    gsl_vector *sigSq_beta,
                                    gsl_vector *v_beta,
                                    gsl_vector *a_beta,
                                    gsl_vector *b_beta);

extern void update_sigSq_alpha_mmzip(gsl_matrix *A,
                                     gsl_matrix *gamma_alpha,
                                     gsl_vector *sigSq_alpha,
                                     gsl_vector *v_alpha,
                                     gsl_vector *a_alpha,
                                     gsl_vector *b_alpha);


extern void update_sigSq_alpha0_mmzip(gsl_vector *alpha0,
                                      double *sigSq_alpha0,
                                      gsl_vector *mu_alpha0,
                                      double a_alpha0,
                                      double b_alpha0);



extern void update_sigSq_beta0_mmzip(gsl_vector *beta0,
                                     double *sigSq_beta0,
                                     gsl_vector *mu_beta0,
                                     double a_beta0,
                                     double b_beta0);

extern void update_m_mmzip(gsl_matrix *X0,
                           gsl_vector *alpha0,
                           gsl_matrix *A,
                           gsl_matrix *W,
                           gsl_vector *m_vec,
                           gsl_matrix *invR,
                           gsl_vector *mu_m,
                           double v_m,
                           int *accept_m,
                           gsl_vector *accept_m100,
                           double *eps_m,
                           int L_m,
                           int *n_m,
                           int *numReps,
                           double *burninPerc,
                           int M,
                           gsl_vector *M_m,
                           double PtuneEps);


extern void update_W_mmzip(gsl_matrix *Y,
                           gsl_matrix *X1,
                           gsl_matrix *X0,
                           gsl_vector *xi,
                           gsl_vector *beta0,
                           gsl_matrix *B,
                           gsl_matrix *V,
                           gsl_vector *alpha0,
                           gsl_matrix *A,
                           gsl_matrix *W,
                           gsl_vector *m_vec,
                           gsl_matrix *invR,
                           double pU_W);

extern void updateRP_alpha_mmzip_SSVS(int *p_adj,
                                      gsl_matrix *Y,
                                      gsl_matrix *X1,
                                      gsl_matrix *X0,
                                      gsl_vector *xi,
                                      gsl_vector *beta0,
                                      gsl_matrix *B,
                                      gsl_matrix *V,
                                      gsl_vector *alpha0,
                                      gsl_matrix *A,
                                      gsl_matrix *W,
                                      gsl_matrix *invR,
                                      gsl_matrix *gamma_alpha,
                                      gsl_matrix *updateNonzA,
                                      gsl_vector *sigSq_alpha,
                                      gsl_vector *v_alpha,
                                      gsl_vector *omega_alpha,
                                      gsl_matrix *alpha_prop_var,
                                      gsl_matrix *accept_A,
                                      gsl_matrix *n_A,
                                      double pU_A_ssvs);





extern void c_solve_covFA1(gsl_vector *lambda,
                           gsl_matrix *Minv);

extern void c_solve_corFA1(gsl_vector *lambda,
                           gsl_matrix *Rinv);


extern void cov_FA1(gsl_vector *lambda,
                    gsl_matrix *M);

extern void c_cov2cor(gsl_matrix *Sigma,
                      gsl_matrix *R);


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

extern void c_rmvnorm2(gsl_matrix *sample,
                       gsl_vector *mean,
                       gsl_matrix *chol);

extern double logistic(double x);

extern void c_rowSums(gsl_matrix *X, gsl_vector *v);

extern void c_solve(gsl_matrix *M,
                    gsl_matrix *Minv);

extern double c_det(gsl_matrix *A);

extern double c_ldet(gsl_matrix *A);

extern int c_sgn_det(gsl_matrix *A);

extern double get_det(gsl_matrix * A);

extern void c_riwishart(double v,
                        gsl_matrix *X_ori,
                        gsl_matrix *sample,
                        gsl_matrix *sampleInv);

extern void c_riwishart3(double v,
                         gsl_matrix *X_ori,
                         gsl_matrix *sample,
                         gsl_matrix *sampleInv,
                         gsl_matrix *cholX);

extern double c_min(double value1,
                    double value2);

extern double c_max(double value1,
                    double value2);

extern void c_colSums(gsl_matrix *X, gsl_vector *v);


extern void c_quadform_vMu(gsl_vector *v,
                           gsl_matrix *Minv,
                           gsl_vector *u,
                           double *value);


extern void matrixInv(gsl_matrix *X, gsl_matrix *Xinv);

extern void c_quadform_vMv(gsl_vector *v,
                           gsl_matrix *Minv,
                           double     *value);


extern void c_dmvnorm2_FA(gsl_vector *x,
                          gsl_vector *mu,
                          double     sigma,
                          gsl_matrix *AInv,
                          gsl_vector *m,
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

extern void c_ldmvn_noDet(gsl_vector *x,
                          gsl_vector *mu,
                          double     sigma,
                          gsl_matrix *AInv,
                          double     *value);

extern void Get_subVector(gsl_vector *vec, int i, gsl_vector *subvec);




