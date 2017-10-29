\name{simData_mzip}
\alias{simData_mzip}
\docType{data}
\title{
A simulated data set containing multivariate zero-inflated count responses and a continuous covariate
}
\description{
A simulated data set containing multivariate zero-inflated count responses and a continuous covariate
}
\usage{data("simData_mzip")}
\format{
a list of two data frame objects. Components include, 
  \describe{
    \item{\code{Y}}{a data frame for 10 multivariate count responses from 300 observations: \code{Y.1}-\code{Y.10}}
    \item{\code{X}}{a data frame for a single continuous covariate from 300 observations: \code{cov.1}}    
  }
}
%\details{
%%  ~~ If necessary, more details than the __description__ above ~~
%}
%\source{
%%  ~~ reference to a publication or URL from which the data were obtained ~~
%}
%\references{
%%  ~~ possibly secondary sources and usages ~~
%}
\examples{
data(simData_mzip)
}
\keyword{datasets}
\keyword{multivariate zero-inflated count data}
