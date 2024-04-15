\name{initiate_startValues}
\alias{initiate_startValues}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The function that initiates starting values
}
\description{
The function initiates starting values. Users are allowed to set some non-null values to starting values for a set of parameters. The function will automatically generate starting values for any parameters whose values are not specified.
}
\usage{
initiate_startValues(Formula, Y, data, model = "MMZIP", B = NULL, beta0 = NULL, 
V = NULL, SigmaV = NULL, gamma_beta = NULL, A = NULL, alpha0 = NULL, W = NULL, 
m = NULL, gamma_alpha = NULL, sigSq_beta = NULL, sigSq_beta0 = NULL, 
sigSq_alpha = NULL, sigSq_alpha0 = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Formula}{
a list containing three formula objects: the first formula specifies the \eqn{p_z} covariates for which variable selection is to be performed in the binary component of the model; the second formula specifies the \eqn{p_x} covariates for which variable selection is to be performed in the count part of the model; the third formula specifies the \eqn{p_0} confounders to be adjusted for (but on which variable selection is not to be performed) in the regression analysis.
}
  \item{Y}{
a data.frame containing \eqn{q} count outcomes from \code{n} subjects. It is of dimension \eqn{n\times q}.
}
  \item{data}{
a data.frame containing the variables named in the formulas in \code{lin.pred}.
}
  \item{model}{
MMZIP
}
  \item{B}{
starting values of \eqn{B}
}
  \item{beta0}{
starting values of \eqn{\beta_0}
}
  \item{V}{
starting values of \eqn{B}
}
  \item{SigmaV}{
starting values of \eqn{\Sigma_V}
}
  \item{gamma_beta}{
starting values of \eqn{\gamma_{\beta}}
}
  \item{A}{
starting values of \eqn{A}
}
  \item{alpha0}{
starting values of \eqn{\alpha_0}
}
  \item{W}{
starting values of \eqn{W}
}
  \item{m}{
starting values of \eqn{m}
}
  \item{gamma_alpha}{
starting values of \eqn{\gamma_{\alpha}}
}
  \item{sigSq_beta}{
starting values of \eqn{\sigma_{\beta}^2}
}
  \item{sigSq_beta0}{
starting values of \eqn{\sigma_{\beta_0}^2}
}
  \item{sigSq_alpha}{
starting values of \eqn{\sigma_{\alpha}^2}
}
  \item{sigSq_alpha0}{
starting values of \eqn{\sigma_{\alpha_0}^2}
}
}
%\details{
%...
%}
\value{
\code{initiate_startValues} returns a list containing starting values that can be used for \code{mmzipBvs}.
}
\references{
update..
}
\author{
Maintainer: Kyu Ha Lee <klee@hsph.harvard.edu>
}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{mmzipBvs}}
}
\examples{
## See Examples in \code{\link{mmzipBvs}}.
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{multivariate Bayesian variable selection}
\keyword{multivariate zero-inflated count data}
