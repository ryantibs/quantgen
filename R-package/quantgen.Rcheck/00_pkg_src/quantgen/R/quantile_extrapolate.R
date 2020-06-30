#' Quantile extrapolater
#'
#' Extrapolate a set of quantiles at new quantile levels: parametric in the
#' tails, nonparametric in the middle.
#'
#' @param tau Vector of quantile levels. Assumed to be distinct, and sorted in 
#'   increasing order.
#' @param qvals Vector or matrix quantiles; if a matrix, each row is a separate 
#'   set of quantiles, at the same (common) quantile levels, given by
#'   \code{tau}.
#' @param tau_out Vector of quantile levels at which to perform
#'   extrapolation. Default is a sequence of 23 quantile levels from 0.01 to
#'   0.99.
#' @param sort Should the returned quantile estimates be sorted? Default is
#'   TRUE. 
#' @param iso Should the returned quantile estimates be passed through isotonic
#'   regression? Default is FALSE; if TRUE, takes priority over \code{sort}.
#' @param nonneg Should the returned quantile estimates be truncated at 0?
#'   Natural for count data. Default is FALSE.
#' @param round Should the returned quantile estimates be rounded? Natural for
#'   count data. Default is FALSE.
#' @param qfun_left,qfun_right Quantile functions on which to base extrapolation
#'   in the left and right tails, respectively; each must be a function whose
#'   first two arguments are a quantile level and a distribution parameter (such
#'   as a mean parameter); these are assumed to be vectorized in the first
#'   argument when the second argument is fixed, and also vectorized in the
#'   second argument when the first argument is fixed. Default is
#'   \code{qnorm}. See details for further explanation.
#' @param n_tau_left,n_tau_right Integers between 1 and the length of
#'   \code{tau}, indicating how many elements quantile levels from the left and
#'   right ends, respectively, to use in defining the tails. For example, if
#'   \code{n_tau_left=1}, the default, then only the leftmost quantile is used
#'   for the left tail extrapolation; if \code{n_tau_left=2}, then the two
#'   leftmost quantiles are used, etc; and similarly for \code{n_tau_right}. See
#'   details for further explanation.
#' @param middle One of "cubic" or "linear", indicating the interpolation method
#'   to use in the middle (outside of the tails, as determined by
#'   \code{n_tau_left} and \code{n_tau_right}). If "cubic", the default, then a
#'   monotone cubic spline interpolant is fit to the given quantiles, and used
#'   to estimate quantiles in the middle. If "linear", then linear interpolation
#'   is used to estimate quantiles in the middle.
#' @param param0,param1 TODO
#' @param grid_size,tol,maxiter TODO
#'
#' @return A matrix of dimension (number of rows in \code{qvals}) x (length of 
#'   \code{tau_out}), where each row is the extrapolation of the set of
#'   quantiles in the corresponding row of \code{qvals}, at the quantile levels 
#'   specified in \code{tau_out}.
#'
#' @details TODO
#' @export

quantile_extrapolate = function(tau, qvals, tau_out=c(0.01, 0.025, seq(0.05,
                                0.95, by=0.05), 0.975, 0.99), sort=TRUE,
                                iso=FALSE, nonneg=FALSE, round=FALSE,
                                qfun_left=qnorm, qfun_right=qnorm,
                                n_tau_left=1, n_tau_right=1, 
                                middle=c("cubic", "linear"),
                                param0=NULL, param1=NULL, grid_size=1000, 
                                tol=0.01, max_iter=10) { 
  qvals = matrix(qvals, ncol=length(tau))
  return(t(apply(qvals, 1, function(v) {
    return(quantile_extrapolate_v(tau=tau, qvals=v, tau_out=tau_out, sort=sort,
                                  iso=iso, nonneg=nonneg, round=round, 
                                  qfun_left=qfun_left, qfun_right=qfun_right,
                                  n_tau_left=n_tau_left,
                                  n_tau_right=n_tau_right,  
                                  middle=middle, param0=param0, param1=param1,
                                  grid_size=grid_size, tol=tol))
  })))
}

# Extrapolate a vector of quantiles at new quantile levels.

quantile_extrapolate_v = function(tau, qvals, tau_out=c(0.01, 0.025, seq(0.05,
                                  0.95, by=0.05), 0.975, 0.99), sort=TRUE,
                                  iso=FALSE, nonneg=FALSE, round=FALSE,
                                  qfun_left=qnorm, qfun_right=qnorm,
                                  n_tau_left=1, n_tau_right=1, 
                                  middle=c("cubic", "linear"),
                                  param0=NULL, param1=NULL, grid_size=1000, 
                                  tol=0.01, max_iter=10) {
  # Set up some basics
  n = length(tau)
  i_left = n_tau_left; i_right = n - n_tau_right + 1
  inds_left = which(tau_out < tau[i_left])
  inds_mid = which(tau[i_left] <= tau_out & tau_out <= tau[i_right])
  inds_right = which(tau_out > tau[i_right])
  if (is.null(param0)) param0 = min(qvals)
  if (is.null(param1)) param1 = max(qvals) 
  if (param0 == param1) param1 = param0 + 1
  middle = match.arg(middle)
  qvals_out = rep(NA, length(tau_out))
  names(qvals_out) = tau_out

  # Left tail
  if (length(inds_left) != 0) {
    qvals_out_left = matrix(NA, length(inds_left), n_tau_left)
    for (i in 1:n_tau_left) {
      param = tryCatch({ mean(find_param(tau=tau[i], qval=qvals[i],
                                         qfun=qfun_left, param0=param0,
                                         param1=param1, grid_size=grid_size, 
                                         tol=tol, max_iter=max_iter)) }, 
                       error = function(e) return(NA))
      if (!is.na(param)) {
        qvals_out_left[,i] = qfun_left(tau_out[inds_left], param)
      }
    }
    qvals_out[inds_left] = rowMeans(qvals_out_left, na.rm=TRUE)
  }

  # Right tail 
  if (length(inds_right) != 0) {
    qvals_out_right = matrix(NA, length(inds_right), n_tau_right)
    for (i in 1:n_tau_right) {
      param = tryCatch({ mean(find_param(tau=tau[n-n_tau_right+i],
                                         qval=qvals[n-n_tau_right+i],
                                         qfun=qfun_right, param0=param0, 
                                         param1=param1, grid_size=grid_size, 
                                         tol=tol, max_iter=max_iter)) }, 
                       error = function(e) return(NA))
      if (!is.na(param)) {
        qvals_out_right[,i] = qfun_right(tau_out[inds_right], param)
      }
    }
    qvals_out[inds_right] = rowMeans(qvals_out_right, na.rm=TRUE)
  }

  # Middle
  if (length(inds_mid) != 0) {
    result = NULL # This will get set to 1 if cubic method fails
    if (middle == "cubic") { 
      result = tryCatch({
        # Fit quantile function via monotone cubic spline interpolation
        Q = splinefun(tau, qvals, method="hyman")
        qvals_out[inds_mid] = Q[tau_out[inds_mid]]
      }, error = function(e) { return(1) })
    }
    if (middle == "linear" || !is.null(result)) {  
      # Fit quantile function via linear interpolation
      qvals_out[inds_mid] = approx(tau, qvals, tau_out[inds_mid])$y
    }
  }

  # Run isotonic regression, sort, truncated, round, if we're asked to
  o = which(!is.na(qvals_out))
  if (sort && !iso) qvals_out[o] = sort(qvals_out[o])
  if (iso) qvals_out[o] = isoreg(qvals_out[o])$yf
  if (nonneg) qvals_out = pmax(qvals_out,0)
  if (round) qvals_out = round(qvals_out)
  return(qvals_out)
}

find_param = function(tau, qval, qfun, param0, param1, grid_size=1000, tol=0.01, 
                      max_iter=100) {   
  for (iter in 1:max_iter) {
    params = seq(param0, param1, length=grid_size)
    inc = (param1 - param0) / grid_size
    suppressWarnings({ qvals = qfun(tau, params) })
    
    # Check if we've exceeded the model parameter range (by accident), and if
    # so, restrict the range appropriately
    na_bools = is.na(qvals)
    if (sum(na_bools) > 0) {
      qvals = qvals[!na_bools]
      params = params[!na_bools]
      param0 = min(params)
      param1 = max(params)
    }
    
    bools_left = qvals <= qval
    bools_right = qvals >= qval
    delta = param1 - param0
    increasing = qvals[length(qvals)] >= qvals[1]
    
    # If no values of the parameter yield a quantile less than qval and the
    # quantile is increasing in the parameter, OR no values yield a quantile
    # greater than qval and the quantile is decreasing in the parameter, then 
    # shift left
    if ((sum(bools_left) == 0 && increasing) || 
        (sum(bools_right) == 0 && !increasing)) { 
      param1 = param0
      param0 = param0 - 2 * delta
    }
    # If no values of the parameter yield a quantile less than qval and the
    # quantile is decreasing in the parameter, OR no values yield a quantile
    # greater than qval and the quantile is increasing in the parameter, then  
    # shift right
    else if ((sum(bools_left) == 0 && !increasing) ||
             (sum(bools_right) == 0 && increasing)) { 
      param0 = param1
      param1 = param1 + 2 * delta
    } 
    # Otherwise we have properly straddled the quantile with the given values 
    else {
      i_left = max(which(bools_left)) 
      i_right = min(which(bools_right)) 
      param0 = params[i_left]
      param1 = params[i_right]
      # If they are out of order, then swap them (can happen, NOT a bug, when
      # dealing with discrete distributions like Poisson)
      if (param0 > param1) {
        val = param0
        param0 = param1
        param1 = val
      }
      # Iterate until our resolution is less than the specified tolerance
      if (inc < tol) return(c(param0, param1))
    }
  }
  warning(paste("Parameter not found after", max_iter,
                "iterations; try resetting param0 or param1, and run again.\n")) 
  return(NA)
}
