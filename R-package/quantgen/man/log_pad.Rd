% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{log_pad}
\alias{log_pad}
\alias{exp_pad}
\title{Convenience functions for log/exp mappings}
\usage{
log_pad(a = 1, b = 1)

exp_pad(a = 1, b = 1)
}
\description{
Returns functions that map \eqn{x \mapsto \log(ax+b)} and \eqn{x \mapsto 
(\exp(x)-b)/a}. (These are inverses.)
}
