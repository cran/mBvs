\name{methods}
\alias{print.mvnBvs}
\alias{print.mzipBvs}
\alias{print.mmzipBvs}
\alias{print.summ.mvnBvs}
\alias{print.summ.mzipBvs}
\alias{print.summ.mmzipBvs}
\alias{summary.mvnBvs}
\alias{summary.mzipBvs}
\alias{summary.mmzipBvs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Methods for objects of class, \code{mvnBvs}, \code{mzipBvs}, and \code{mmzipBvs}.
}
\description{
The \code{mvnBvs} class represents results from Bayesian variable selection using multivariate normal regression models. The \code{mzipBvs} and \code{mmzipBvs} classes represent results from conditional and marginalized multivariate zero-inflated regression models, respectively.\cr
}
\usage{
\method{print}{mvnBvs}(x, digits=3, ...)
\method{print}{mzipBvs}(x, digits=3, ...)
\method{print}{mmzipBvs}(x, digits=3, ...)
\method{print}{summ.mvnBvs}(x, digits=3, ...)
\method{print}{summ.mzipBvs}(x, digits=3, ...)
\method{print}{summ.mmzipBvs}(x, digits=3, ...)
\method{summary}{mvnBvs}(object, digits=3, ...)
\method{summary}{mzipBvs}(object, digits=3, ...)
\method{summary}{mmzipBvs}(object, digits=3, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    an object of class \code{mvnBvs}, \code{summ.mvnBvs}, \code{mzipBvs}, \code{summ.mzipBvs}, \code{mmzipBvs}, \code{summ.mmzipBvs}.
}
  \item{digits}{
    a numeric value indicating the number of digits to display.
}
  \item{object}{
    an object of class \code{mvnBvs}, \code{mzipBvs}, or \code{mmzipBvs}.
}
	\item{...}{
	additional arguments.
}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
	\code{\link{mvnBvs}}, \code{\link{mzipBvs}}, \code{\link{mmzipBvs}}
}


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate Bayesian variable selection}
