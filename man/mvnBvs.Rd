\name{mvnBvs}
\alias{mvnBvs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The function to perform variable selection for multivariate normal responses
}
\description{
The function can be used to perform variable selection for multivariate normal responses incorporating not only information on the mean model, but also information on the variance-covariance structure of the outcomes. A multivariate prior is specified on the latent binary selection indicators to incorporate the dependence between outcomes into the variable selection procedure.
}
\usage{
mvnBvs(Y, lin.pred, data, model = "unstructured", hyperParams, startValues, mcmcParams)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
	a data.frame containing \eqn{q} continuous multivariate outcomes from \code{n} subjects. It is of dimension \eqn{n\times q}.
}
  \item{lin.pred}{
	a list containing two formula objects: the first formula specifies the \eqn{p} covariates for which variable selection is to be performed; the second formula specifies the confounders to be adjusted for (but on which variable selection is not to be performed) in the regression analysis.
}
  \item{data}{
	a data.frame containing the variables named in the formulas in \code{lin.pred}.
}
  \item{model}{
	a character that specifies the covariance structure of the model: either "unstructured" or "factor-analytic". 
}
  \item{hyperParams}{
	a list containing lists or vectors for hyperparameter values in hierarchical models. Components include, 
	\code{eta} (a numeric value for the hyperparameter \eqn{\eta} that regulates the extent to which the correlation between response variables influences the prior of the variable selection indicator), 
	\code{v} (a numeric vector of length \eqn{q} for the standard deviation hyperparameter \eqn{v} of the regression parameter \eqn{\beta} prior), 
	\code{omega} (a numeric vector of length \eqn{p} for the hyperparameter \eqn{\omega} in the prior of the variable selection indicator), 		
	\code{beta0} (a numeric vector of length \eqn{q+1} for hyperparameter \eqn{\mu_0} and \eqn{h_0} in the prior of the intercept \eqn{\beta_0}), 
	\code{US} (a list containing numeric vectors for hyperparameters in the unstructured model: \code{US.Sigma}),
	\code{FA} (a list containing numeric vectors for hyperparameters in the factor-analytic model: \code{lambda} and \code{sigmaSq}).
	See Examples below.
}
  \item{startValues}{
  	a numeric vector containing starting values for model parameters: c(\code{beta0}, \code{B}, \code{gamma}, \code{Sigma}) for the unstructured model; c(\code{beta0}, \code{B}, \code{gamma}, \code{sigmaSq}, \code{lambda}) for the factor-analytic model. See Examples below.
}
  \item{mcmcParams}{
	a list containing variables required for MCMC sampling. Components include,
	\code{run} (a list containing numeric values for setting the overall run: 
	\code{numReps}, total number of scans; \code{thin}, extent of thinning; \code{burninPerc}, the proportion of burn-in).
	\code{tuning} (a list containing numeric values relevant to tuning parameters for specific updates in Metropolis-Hastings algorithm: 
	\code{mhProp_beta_var}, variance of the proposal density for \eqn{B}; \code{mhrho_prop}, degrees of freedom of the inverse-Wishart proposal density for \eqn{\Sigma} in the unstructured model; 
	\code{mhPsi_prop}, scale matrix of inverse-Wishart proposal density for \eqn{\Sigma} in the unstructured model; 
	\code{mhProp_lambda_var}, variance of the proposal density for \eqn{\lambda} in the factor-analytic model.)
	See Examples below.
}
}
%\details{
%
%}
\value{
\code{mvnBvs} returns an object of class \code{mvnBvs}. \cr
}
\references{
Lee, K. H., Tadesse, M. G., Baccarelli, A. A., Schwartz J., and Coull, B. A. (2016), 
Multivariate Bayesian variable selection exploiting dependence structure among outcomes: 
application to air pollution effects on DNA methylation, \emph{Biometrics}, Volume 73, Issue 1, pages 232-241. \cr
}
\author{
Kyu Ha Lee, Mahlet G. Tadesse, Brent A. Coull \cr
Maintainer: Kyu Ha Lee <klee@hsph.harvard.edu>

}
%\note{
%%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{

# loading a data set	
data(simData_cont)
Y <- simData_cont$Y
data <- simData_cont$X
form1 <- as.formula( ~ cov.1+cov.2)
form2 <- as.formula( ~ 1)
lin.pred <- list(form1, form2)

p 		<- dim(data)[2]
p_adj 	<- 0
q 		<- dim(Y)[2]

#####################
## Hyperparameters ##

## Common hyperparameters
##
eta = 0.1
v = rep(10, q)
omega = rep(log(0.5/(1-0.5)), p-p_adj)
common.beta0 <- c(rep(0, q), 10^6)

## Unstructured model
##
rho0 	<- q + 4
Psi0	<- diag(3, q)
US.Sigma <- c(rho0, Psi0)

## Factor-analytic model
##
FA.lam		<- c(rep(0, q), 10^6)
FA.sigSq	<- c(2, 1)

##
hyperParams <- list(eta=eta, v=v, omega=omega, beta0=common.beta0,
					US=list(US.Sigma=US.Sigma),
					FA=list(lambda=FA.lam, sigmaSq=FA.sigSq))
                    
###################
## MCMC SETTINGS ##

## Setting for the overall run
##
numReps    <- 50
thin       <- 1
burninPerc <- 0.5

## Tuning parameters for specific updates
##
##  - those common to all models
mhProp_beta_var  <- matrix(0.5, p+p_adj, q)
##
## - those specific to the unstructured model
mhrho_prop <- 1000
mhPsi_prop <- diag(1, q)
##
## - those specific to the factor-analytic model
mhProp_lambda_var  <- 0.5
      
##
mcmc.US <- list(run=list(numReps=numReps, thin=thin, burninPerc=burninPerc),
                tuning=list(mhProp_beta_var=mhProp_beta_var,
                				mhrho_prop=mhrho_prop, mhPsi_prop=mhPsi_prop))
                
##
mcmc.FA  <- list(run=list(numReps=numReps, thin=thin, burninPerc=burninPerc),
                tuning=list(mhProp_beta_var=mhProp_beta_var,
                			mhProp_lambda_var=mhProp_lambda_var))
                			
#####################
## Starting Values ##

##  - those common to all models
beta0		<- rep(0, q)
B			<- matrix(sample(x=c(0.3, 0), size=q, replace = TRUE), p+p_adj, q)
gamma		<- B
gamma[gamma != 0]	<- 1
##
## - those specific to the unstructured model
Sigma <- diag(1, q) %*% cor(Y) %*% diag(1, q)
##
## - those specific to the factor-analytic model
lambda		<- rep(0.5, q)
sigmaSq		<- 1
                
####################################
## Fitting the unstructured model ##
####################################

startValues	<- vector("list", 2)                
 
startValues[[1]] <- as.vector(c(beta0, B, gamma, Sigma))

beta0		<- rep(0.2, q)
Sigma <- diag(0.5, q) %*% cor(Y) %*% diag(0.5, q)

startValues[[2]] <- as.vector(c(beta0, B, gamma, Sigma))

fit.us <- mvnBvs(Y, lin.pred, data, model="unstructured", hyperParams, 
				startValues, mcmcParams=mcmc.US)
				
fit.us
summ.fit.us <- summary(fit.us); names(summ.fit.us)
summ.fit.us				

#######################################
## Fitting the factor-analytic model ##
#######################################

startValues	<- vector("list", 2)                
 
startValues[[1]] <- as.vector(c(beta0, B, gamma, sigmaSq, lambda))

beta0		<- rep(0.2, q)
sigmaSq		<- 0.5
startValues[[2]] <- as.vector(c(beta0, B, gamma, sigmaSq, lambda))

fit.fa <- mvnBvs(Y, lin.pred, data, model="factor-analytic", hyperParams, 
				startValues, mcmcParams=mcmc.FA)
				
fit.fa
summ.fit.fa <- summary(fit.fa); names(summ.fit.fa)
summ.fit.fa				


}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate Bayesian variable selection}
\keyword{multivariate continuous data}


