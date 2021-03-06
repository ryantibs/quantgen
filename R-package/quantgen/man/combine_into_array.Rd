% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/quantile_ensemble.R
\name{combine_into_array}
\alias{combine_into_array}
\title{Combine matrices into an array}
\usage{
combine_into_array(mat, ...)
}
\arguments{
\item{mat}{First matrix to combine into an array. Alternatively, a list of
matrices to combine into an array.}

\item{...}{Additional matrices to combine into an array. These additional
arguments will be ignored if \code{mat} is a list.}
}
\description{
Combine (say) p matrices, each of dimension n x r, into an n x p x r array.
}
