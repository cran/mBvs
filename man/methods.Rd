\name{methods}
\alias{print.mvnBvs}
\alias{print.summ.mvnBvs}
\alias{summary.mvnBvs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Methods for objects of class, \code{mvnBvs}.
}
\description{
The \code{mvnBvs} class represents results from Bayesian variable selection using multivariate normal regression models.\cr
}
\usage{
\method{print}{mvnBvs}(x, digits=3, ...)
\method{print}{summ.mvnBvs}(x, digits=3, ...)
\method{summary}{mvnBvs}(object, digits=3, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    an object of class \code{mvnBvs} or \code{summ.mvnBvs}.
}
  \item{digits}{
    a numeric value indicating the number of digits to display.
}
  \item{object}{
    an object of class \code{mvnBvs}.
}
	\item{...}{
	additional arguments.
}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
	\code{\link{mvnBvs}}
}


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate Bayesian variable selection}
