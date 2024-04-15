\name{mmzipBvs}
\alias{mmzipBvs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The function to perform variable selection for marginalized multivariate zero-inflated Poisson models
}
\description{
The function can be used to perform variable selection for marginalized multivariate zero-inflated Poisson models.
}
\usage{
mmzipBvs(Y, lin.pred, data, offset = NULL, zero_cutoff = 0.05, hyperParams,
 startValues, mcmcParams)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
a data.frame containing \eqn{q} count outcomes from \code{n} subjects. It is of dimension \eqn{n\times q}.
}
  \item{lin.pred}{
a list containing three formula objects: the first formula specifies the \eqn{p_z} covariates for which variable selection is to be performed in the binary component of the model; the second formula specifies the \eqn{p_x} covariates for which variable selection is to be performed in the count part of the model; the third formula specifies the \eqn{p_0} confounders to be adjusted for (but on which variable selection is not to be performed) in the regression analysis.
}
  \item{data}{
a data.frame containing the variables named in the formulas in \code{lin.pred}.
}
  \item{offset}{
an optional numeric vector with an a priori known component to be included as the
linear predictor in the count part of model.
}
  \item{zero_cutoff}{
Response variable with proportions of zeros less than \code{zero.cutoff} will be removed from the binary model.
}
  \item{hyperParams}{
(update this)
	a list containing lists or vectors for hyperparameter values in hierarchical models. Components include,
    \code{rho0} (degrees of freedom for inverse-Wishart prior for \eqn{\Sigma_V}),
    \code{Psi0} (a scale matrix for inverse-Wishart prior for \eqn{\Sigma_V}),
    \code{mu_alpha0} (hyperparameter \eqn{\mu_{\alpha_0}} in the prior of \eqn{\alpha_0}),
    \code{mu_alpha} (a numeric vector of length \eqn{q} for hyperparameter \eqn{\mu_{\alpha}} in the prior of \eqn{\alpha}),
    \code{mu_beta0} (hyperparameter \eqn{\mu_{\beta_0}} in the prior of \eqn{\beta_0}),
    \code{mu_beta} (a numeric vector of length \eqn{q} for hyperparameter \eqn{\mu_{\beta}} in the prior of \eqn{\beta}),
    \code{a_alpha0} (hyperparameter \eqn{a_{\alpha_0}} in the prior of \eqn{\sigma^2_{\alpha_0}}),
    \code{b_alpha0} (hyperparameter \eqn{b_{\alpha_0}} in the prior of \eqn{\sigma^2_{\alpha_0}}),
    \code{a_alpha} (hyperparameter \eqn{a_{\alpha}} in the prior of \eqn{\sigma^2_{\alpha}}),
    \code{b_alpha} (hyperparameter \eqn{b_{\alpha}} in the prior of \eqn{\sigma^2_{\alpha}}),
    \code{a_beta0} (hyperparameter \eqn{a_{\beta_0}} in the prior of \eqn{\sigma^2_{\beta_0}}),
    \code{b_beta0} (hyperparameter \eqn{b_{\beta_0}} in the prior of \eqn{\sigma^2_{\beta_0}}),
    \code{a_beta} (hyperparameter \eqn{a_{\beta}} in the prior of \eqn{\sigma^2_{\beta}}),
    \code{b_beta} (hyperparameter \eqn{b_{\beta}} in the prior of \eqn{\sigma^2_{\beta}}),
    \code{v_beta} (a numeric vector of length \eqn{q} for the standard deviation hyperparameter \eqn{v_{\beta}} of the regression parameter \eqn{\beta} prior),
    \code{omega_beta} (a numeric vector of length \eqn{p_x-p_0} for the hyperparameter \eqn{\omega_{\beta}} in the prior of the variable selection indicator),
    \code{v_alpha} (a numeric vector of length \eqn{q} for the standard deviation hyperparameter \eqn{v_{\alpha}} of the regression parameter \eqn{\alpha} prior),
\code{omega_alpha} (a numeric vector of length \eqn{p_z-p_0} for the hyperparameter \eqn{\omega_{\alpha}} in the prior of the variable selection indicator),
	See Examples below.
}
  \item{startValues}{
  	a numeric vector containing starting values for model parameters. See Examples below.
}
  \item{mcmcParams}{
(update this)
	a list containing variables required for MCMC sampling. Components include,
	\code{run} (a list containing numeric values for setting the overall run: 
	\code{numReps}, total number of scans; \code{thin}, extent of thinning; \code{burninPerc}, the proportion of burn-in).
	\code{tuning} (a list containing numeric values relevant to tuning parameters for specific updates in Metropolis-Hastings algorithm: 
    \code{beta0.prop.var}, variance of the proposal density for \eqn{\beta_0};\code{beta.prop.var}, variance of the proposal density for \eqn{B};\code{alpha.prop.var}, variance of the proposal density for \eqn{A};\code{V.prop.var}, variance of the proposal density for \eqn{V}.)
	See Examples below.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
\code{mmzipBvs} returns an object of class \code{mmzipBvs}.
}
\references{
update this
}
\author{
Kyu Ha Lee, Brent A. Coull, Jacqueline R. Starr \cr
Maintainer: Kyu Ha Lee <klee15239@gmail.com>
}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
\dontrun{

# loading a data set
data(simData_mzip)
Y <- simData_mzip$Y
data <- simData_mzip$X

n = dim(Y)[1]
q = dim(Y)[2]

form.bin     <- as.formula(~cov.1)
form.count    <- as.formula(~cov.1)
form.adj    <- as.formula(~1)
form <- list(form.bin, form.count, form.adj)

p_adj = dim(model.frame(form[[3]], data=data))[2]
p0 <- dim(model.frame(form[[1]], data=data))[2] + p_adj
p1 <- dim(model.frame(form[[2]], data=data))[2] + p_adj

#####################
## Hyperparameters ##

Sigma_me <- 0.5
Sigma_var <- 1
rho0 <- 2*Sigma_me^2/Sigma_var+q+3
psi0 <- Sigma_me*(rho0-q-1)

hyperParams_mmzip <- list(v_beta=rep(3, q), omega_beta=rep(0.5, p1-p_adj), 
a_beta=rep(0.5, p1), b_beta=rep(0.5, p1), mu_beta0=rep(0, q), a_beta0=0.5, b_beta0=0.5, 
v_alpha=rep(3, q), omega_alpha=rep(0.5, p0-p_adj), 
a_alpha=rep(0.5, p0), b_alpha=rep(0.5, p0), mu_alpha0=rep(0, q), a_alpha0=0.5, b_alpha0=0.5, 
rho0=rho0, Psi0=diag(psi0, q), mu_m=rep(0, q), v_m=0.5)


###################
## MCMC SETTINGS ##

run <- list(numReps=100, thin=1, burninPerc=0.5)
storage <- list(storeV=FALSE, storeW=FALSE)
vs <- list(count=TRUE, binary=TRUE)
tuning <- list(L_group=100, L_m=20, eps_group=0.00001, eps_m=0.00001, 
Mvar_group=1, Mvar_m=1, beta_prop_var=0.0001, alpha_prop_var=0.0001)

mcmc_mmzip <- list(run=run, storage=storage, vs=vs, tuning=tuning)


#####################
## Starting Values

startValues_mmzip <- initiate_startValues(form, Y, data, "MMZIP")

#####################
## Other settings

offset <- data$total
zero_cutoff=0.05

#######################
## Fitting the MMZIP ##
#######################

fit.mmzip <- mmzipBvs(Y, form, data, offset, zero_cutoff, hyperParams_mmzip, 
startValues_mmzip, mcmc_mmzip)

print(fit.mmzip)
summ.fit.mmzip <- summary(fit.mmzip); names(fit.mmzip)
summ.fit.mmzip

}


}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{multivariate Bayesian variable selection}
\keyword{multivariate zero-inflated count data}

