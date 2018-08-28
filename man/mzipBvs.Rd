\name{mzipBvs}
\alias{mzipBvs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The function to perform variable selection for multivariate zero-inflated count responses
}
\description{
The function can be used to perform variable selection for multivariate zero-inflated count responses.
}
\usage{
mzipBvs(Y, lin.pred, data, model = "generalized", offset = NULL, hyperParams, startValues,
 mcmcParams)
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
  \item{model}{
a character that specifies the type of model: A generalized multivariate Bayesian variable selection method of Lee et al.(2017+) can be implemented by setting \code{model="generalized"}. A simpler model that assumes one common variable selection indicator (\eqn{\gamma_{j,k}=\delta_{j,k}}) and the same covariance pattern (\eqn{R=R_V}) for two model parts can be used by setting \code{model="restricted1"}. iii) Another simpler model that assumes the same covariance pattern (\eqn{R=R_V}) but separate variable selection indicators for the binary and count parts of the model can be implemented by setting \code{model="restricted2"}.
}
  \item{offset}{
an optional numeric vector with an a priori known component to be included as the
linear predictor in the count part of model.
}
  \item{hyperParams}{
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
\code{mzipBvs} returns an object of class \code{mzipBvs}. \cr}
\references{
Lee, K. H., Coull, B. A., Moscicki, A.-B., Paster, B. J., Starr, J. R. (2018+), 
Bayesian variable selection for multivariate zero-inflated models: application to microbiome count data, \emph{under review}. \cr
}
\author{
Kyu Ha Lee, Brent A. Coull, Jacqueline R. Starr \cr
Maintainer: Kyu Ha Lee <klee@hsph.harvard.edu>
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
lin.pred <- list(form.bin, form.count, form.adj)

Xmat0 <- model.frame(lin.pred[[1]], data=data)
Xmat1 <- model.frame(lin.pred[[2]], data=data)
Xmat_adj <- model.frame(lin.pred[[3]], data=data)

p_adj = ncol(Xmat_adj)
p0 <- ncol(Xmat0) + p_adj
p1 <- ncol(Xmat1) + p_adj

nonz <- rep(NA, q)
for(j in 1:q) nonz[j] <- sum(Y[,j] != 0)

#####################
## Hyperparameters ##

## Generalized model
##
rho0     <- q + 3 + 1
Psi0    <- diag(3, q)

mu_alpha0     <- 0
mu_alpha    <- rep(0, q)

mu_beta0    <- 0
mu_beta        <- rep(0, q)

a_alpha0    <- 0.7
b_alpha0     <- 0.7

a_alpha        <- rep(0.7, p0)
b_alpha     <- rep(0.7, p0)

a_beta0        <- 0.7
b_beta0     <- 0.7

a_beta        <- rep(0.7, p1)
b_beta         <- rep(0.7, p1)

v_beta = rep(1, q)
omega_beta = rep(0.1, p1-p_adj)
v_alpha = rep(1, q)
omega_alpha = rep(0.1, p0-p_adj)

##
hyperParams.gen <- list(rho0=rho0, Psi0=Psi0, mu_alpha0=mu_alpha0, mu_alpha=mu_alpha,
mu_beta0=mu_beta0, mu_beta=mu_beta, a_alpha0=a_alpha0, b_alpha0=b_alpha0,
a_alpha=a_alpha, b_alpha=b_alpha, a_beta0=a_beta0, b_beta0=b_beta0,
a_beta=a_beta, b_beta=b_beta, v_beta=v_beta, omega_beta=omega_beta,
v_alpha=v_alpha, omega_alpha=omega_alpha)

###################
## MCMC SETTINGS ##

## Setting for the overall run
##
numReps    <- 100
thin       <- 1
burninPerc <- 0.5

## Settings for storage
##
storeV      <-    TRUE
storeW      <-    TRUE

## Tuning parameters for specific updates
##
##  - Generalized model
beta0.prop.var    <- 0.5
alpha.prop.var    <- 0.5
beta.prop.var    <- 0.5
V.prop.var    <- 0.05

##
mcmc.gen <- list(run=list(numReps=numReps, thin=thin, burninPerc=burninPerc),
storage=list(storeV=storeV, storeW=storeW),
tuning=list(beta0.prop.var=beta0.prop.var, alpha.prop.var=alpha.prop.var,
beta.prop.var=beta.prop.var, V.prop.var=V.prop.var))

#####################
## Starting Values ##

## Generalized model
##
B <- matrix(0.1, p1, q, byrow = T)
A <- matrix(0.1, p0, q, byrow = T)

V <- matrix(rnorm(n*q, 0, 0.1), n, q)
W <- matrix(rnorm(n*q, 0, 0.1), n, q)

beta0 <- log(as.vector(apply(Y, 2, mean)))
alpha0 <- log(nonz/n / ((n-nonz)/n))

Sigma_V    <- matrix(0, q, q)
diag(Sigma_V) <- 1

R        <- matrix(0, q, q)
diag(R) <- 1

sigSq_alpha0 <- 1
sigSq_alpha <- rep(1, p0)
sigSq_beta0 <- 1
sigSq_beta <- rep(1, p1)

startValues.gen <- vector("list", 2)
startValues.gen[[1]] <- list(B=B, A=A, V=V, W=W, beta0=beta0, alpha0=alpha0, R=R,
sigSq_alpha0=sigSq_alpha0,
sigSq_alpha=sigSq_alpha, sigSq_beta0=sigSq_beta0, sigSq_beta=sigSq_beta, Sigma_V=Sigma_V)

B <- matrix(-0.1, p1, q, byrow = T)
A <- matrix(-0.1, p0, q, byrow = T)

V <- matrix(rnorm(n*q, 0, 0.1), n, q)
W <- matrix(rnorm(n*q, 0, 0.1), n, q)

Sigma_V    <- matrix(0.1, q, q)
diag(Sigma_V) <- 1.1

startValues.gen[[2]] <- list(B=B, A=A, V=V, W=W, beta0=beta0, alpha0=alpha0, R=R,
sigSq_alpha0=sigSq_alpha0,
sigSq_alpha=sigSq_alpha, sigSq_beta0=sigSq_beta0, sigSq_beta=sigSq_beta, Sigma_V=Sigma_V)

###################################
## Fitting the generalized model ##
###################################
fit.gen <- mzipBvs(Y, lin.pred, data, model="generalized", offset=NULL, hyperParams.gen,
startValues.gen, mcmc.gen)

print(fit.gen)
summ.fit.gen <- summary(fit.gen); names(summ.fit.gen)
summ.fit.gen

}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate Bayesian variable selection}
\keyword{multivariate zero-inflated count data}

