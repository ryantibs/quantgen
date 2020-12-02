# Seq function

Seq = function(a,b) {
  if (a<=b) return(a:b)
  else return(integer(0))
}

# max.row function

max.row = function(m, ties.method=c("random", "first", "last")) {
  return(max.col(t(m), ties.method))
}

# Quiet function

quiet = function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

# Enlist function

enlist = function(...) {
  result = list(...)
  if ((nargs() == 1) & is.character(n <- result[[1]])) {
    result = as.list(seq(n))
    names(result) = n
    for (i in n) result[[i]] = get(i)
  }
  else {
    n = sys.call()
    n = as.character(n)[-1]
    if (!is.null(n2 <- names(result))) {
      which = n2 != ""
      n[which] = n2[which]
    }
    names(result) = n
  }
  return(result)
}

# Setup function
setup_xyd = function(x, y, d, intercept=TRUE, standardize=TRUE, transform=NULL) {
  n = nrow(x)
  p = ncol(x)
  m = nrow(d)
  bx = NULL
  sx = NULL

  # Standardize the columns of x, if we're asked to
  if (standardize) {
    bx = apply(x,2,mean)
    sx = apply(x,2,sd)
    sx[sx < sqrt(.Machine$double.eps)] = 1 # Don't divide by zero!
    x = scale(x,bx,sx)
  }

  # Add all 1s column to x, and all 0s column to d, if we need to
  if (intercept) {
    x = cbind(rep(1,n), x)
    d = cbind(rep(0,m), d)
    p = p+1
  }

  # Transform y, if we're asked to
  if (!is.null(transform)) y = transform(y)

  return(enlist(x, y, d, bx, sx))
}

#' Quantile loss
#'
#' Compute the quantile (tilted absolute) loss for a single tau value.
#'
#' @export

quantile_loss = function(yhat, y, tau) {
  yhat = matrix(yhat, nrow=length(y))
  return(colSums(pmax(tau*(y-yhat), (tau-1)*(y-yhat)), na.rm=TRUE))
}

#' Convenience functions for log/exp mappings
#'
#' Returns functions that map \eqn{x \mapsto \log(ax+b)} and \eqn{x \mapsto 
#' (\exp(x)-b)/a}. (These are inverses.)
#'
#' @export

log_pad = function(a=1, b=1) return(function(x) log(a*x+b))

#' @rdname log_pad
#' @export
exp_pad = function(a=1, b=1) return(function(x) (exp(x)-b)/a)

#' Convenience functions for logit/sigmoid mappings
#'
#' Returns functions that map \eqn{x \mapsto \log(\frac{ax+b}{1-ax+b})} and
#' \eqn{x \mapsto \frac{\exp(x)(1+b)-b}{a(1+\exp(x))}}. (These are inverses.)

logit_pad = function(a=1, b=0.01) return(function(x) log((a*x+b) / (1-a*x+b))) 

#' @rdname logit_pad
#' @export

sigmd_pad = function(a=1, b=0.011) return(function(x) (exp(x) * (1+b)-b) /
                                                      ((1+exp(x)) * a))
#' Convenience function for uniform jitter
#'
#' Function to generate random draws from \eqn{\mathrm{Unif}[a,b]}.
#'
#' @export

unif_jitter = function(a=0, b=0.01) function(n) runif(n,a,b)

#' Difference matrix
#'
#' Construct a difference operator, of a given order, for use in trend filtering
#' penalties.
#'
#' @param p Dimension (number of columns) of the difference matrix.
#' @param k Order of the difference matrix.
#'
#' @return A sparse matrix of dimension (p - k) x p.
#' @importFrom Matrix bandSparse bdiag
#' @export

get_diff_mat = function(p, k) {
  I = Diagonal(p)
  D = bandSparse(p, k=c(-1,0), diagonals=list(rep(-1,p-1), rep(1,p)))
  B = I
  for (i in Seq(1,k)) {
    B = bdiag(I[Seq(1,i-1),Seq(1,i-1)], D[1:(p-i+1),1:(p-i+1)]) %*% B
  }
  return(B[-Seq(1,k),])
}
