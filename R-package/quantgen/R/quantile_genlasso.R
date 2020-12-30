#' Quantile generalized lasso
#'
#' Compute quantile generalized lasso solutions.
#'
#' @param x Matrix of predictors. If sparse, then passing it an appropriate
#'   sparse \code{Matrix} class can greatly help optimization.
#' @param y Vector of responses.
#' @param d Matrix defining the generalized lasso penalty; see details. If
#'   sparse, then passing it an appropriate sparse \code{Matrix} class can
#'   greatly help optimization. A convenience function \code{get_diff_mat} for
#'   constructing trend filtering penalties is provided.
#' @param tau,lambda Vectors of quantile levels and tuning parameter values. If
#'   these are not of the same length, the shorter of the two is recycled so
#'   that they become the same length. Then, for each \code{i}, we solve a
#'   separate quantile generalized lasso problem at quantile level \code{tau[i]}
#'   and tuning parameter value \code{lambda[i]}. The most common use cases are:
#'   specifying one tau value and a sequence of lambda values; or specifying a
#'   sequence of tau values and one lambda value.
#' @param weights Vector of observation weights (to be used in the loss
#'   function). Default is NULL, which is interpreted as a weight of 1 for each
#'   observation.
#' @param intercept Should an intercept be included in the regression model?
#'   Default is TRUE.
#' @param standardize Should the predictors be standardized (to have zero mean
#'   and unit variance) before fitting?  Default is TRUE.
#' @param noncross Should noncrossing constraints be applied? These force the
#'   estimated quantiles to be properly ordered across all quantile levels being
#'   considered. The default is FALSE. If TRUE, then noncrossing constraints are
#'   applied to the estimated quantiles at all points specified by the next
#'   argument \code{x0}. Note: this option only makes sense if the values in the
#'   \code{tau} vector are distinct, and sorted in increasing order.
#' @param x0 Matrix of points used to define the noncrossing
#'   constraints. Default is NULL, which means that we consider noncrossing
#'   constraints at the training points \code{x}.
#' @param lp_solver One of "glpk" or "gurobi", indicating which LP solver to
#'   use. If possible, "gurobi" should be used because it is much faster and
#'   more stable; default is "glpk"; however, because it is open-source.
#' @param time_limit This sets the maximum amount of time (in seconds) to allow
#'   Gurobi or GLPK to solve any single quantile generalized lasso problem (for
#'   a single tau and lambda value). Default is NULL, which means unlimited
#'   time.
#' @param warm_starts Should warm starts be used in the LP solver (from one LP
#'   solve to the next)? Only supported for Gurobi.
#' @param params List of control parameters to pass to Gurobi or GLPK. Default
#'   is \code{list()} which means no additional parameters are passed. For
#'   example: with Gurobi, we can use \code{list(Threads=4)} to specify that
#'   Gurobi should use 4 threads when available. (Note that if a time limit is
#'   specified through this \code{params} list, then its value will be overriden
#'   by the last argument \code{time_limit}, assuming the latter is not NULL.)
#' @param transform,inv_trans The first is a function to transform y before
#'   solving the quantile generalized lasso; the second is the corresponding
#'   inverse transform. For example: for count data, we might want to model
#'   log(1+y) (which would be the transform, and the inverse transform would be
#'   exp(x)-1). Both \code{transform} and \code{inv_trans} should be
#'   vectorized. Convenience functions \code{log_pad} and \code{exp_pad} are
#'   provided (these are inverses), as well as \code{logit_pad} and
#'   \code{sigmd_pad} (these are inverses).
#' @param jitter Function for applying random jitter to y, which might help
#'   optimization. For example: for count data, there can be lots of ties (with
#'   or without transformation of y), which can make optimization more
#'   difficult. The function \code{jitter} should take an integer n and return n
#'   random draws. A convenience function \code{unif_jitter} is provided.
#' @param verbose Should progress be printed out to the console? Default is
#'   FALSE.
#'
#' @return A list with the following components:
#'   \item{beta}{Matrix of generalized lasso coefficients, of dimension =
#'   (number of features + 1) x (number of quantile levels) assuming
#'   \code{intercept=TRUE}, else (number of features) x (number of quantile
#'   levels). Note}{these coefficients will always be on the appropriate scale;
#'   they are always on the scale of original features, even if
#'   \code{standardize=TRUE}}
#'   \item{status}{Vector of status flags returned by Gurobi's or GLPK's LP
#'   solver, of length = (number of quantile levels)}
#'   \item{tau,lambda}{Vectors of tau and lambda values used}
#'   \item{weights,intercept,...,jitter}{Values of these other arguments
#'   used in the function call}
#'
#' @details This function solves the quantile generalized lasso problem, for
#'   each pair of quantile level \eqn{\tau} and tuning parameter \eqn{\lambda}:
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_0,\beta} \;
#'   \sum_{i=1}^n w_i \psi_\tau(y_i-\beta_0-x_i^T\beta) + \lambda \|D\beta\|_1}
#'   for a response vector \eqn{y} with components \eqn{y_i}, predictor matrix
#'   \eqn{X} with rows \eqn{x_i}, and penalty matrix \eqn{D}. Here
#'   \eqn{\psi_\tau(v) = \max\{\tau v, (\tau-1) v\}} is the
#'   "pinball" or "tilted \eqn{\ell_1}" loss. When noncrossing constraints are
#'   applied, we instead solve one big joint optimization, over all quantile
#'   levels and tuning parameter values:
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_{0k}, \beta_k, k=1,\ldots,r} \;
#'   \sum_{k=1}^r \bigg(\sum_{i=1}^n w_i \psi_{\tau_k}(y_i-\beta_{0k}-
#'   x_i^T\beta_k) + \lambda_k \|D\beta_k\|_1\bigg)}
#'   \deqn{\mathrm{subject \; to} \;\; \beta_{0k}+x^T\beta_k \leq
#'   \beta_{0,k+1}+x^T\beta_{k+1} \;\; k=1,\ldots,r-1, \; x \in \mathcal{X}}
#'   where the quantile levels \eqn{\tau_k, k=1,\ldots,r} are assumed to be in
#'   increasing order, and \eqn{\mathcal{X}} is a collection of points over
#'   which to enforce the noncrossing constraints.
#'
#'   Either problem is readily converted into a linear program (LP), and solved
#'   using either Gurobi (which is free for academic use, and generally fast) or
#'   GLPK (which free for everyone, but slower).
#'
#' @importFrom graphics legend matplot
#' @importFrom stats approx coef isoreg median predict qnorm runif sd splinefun
#' @importFrom Rglpk Rglpk_solve_LP
#' @importFrom Matrix Matrix Diagonal
#' @author Ryan Tibshirani
#' @export

quantile_genlasso = function(x, y, d, tau, lambda, weights=NULL, intercept=TRUE,
                             standardize=TRUE, noncross=FALSE, x0=NULL,
                             lp_solver=c("glpk","gurobi"), time_limit=NULL,
                             warm_starts=TRUE, params=list(), transform=NULL,
                             inv_trans=NULL, jitter=NULL, verbose=FALSE) {
  # Check LP solver
  lp_solver = match.arg(lp_solver)
  
  # Set up x, y, d, weights
  a = setup_xyd(x, y, d, weights, intercept, standardize, transform)
  x = a$x
  y = a$y
  d = a$d
  sx = a$sx
  bx = a$bx
  weights = a$weights

  # Problem dimensions
  n = nrow(x)
  p = ncol(x)
  m = nrow(d)

  # Recycle tau or lambda so that they're the same length
  if (length(tau) != length(lambda)) {
    k = max(length(tau), length(lambda))
    tau = rep(tau, length=k)
    lambda = rep(lambda, length=k)
  }

  # Properly set up x0, if there's noncrossing constraints 
  if (noncross) {
    # If there's no x0 passed, then just use the training points
    if (is.null(x0)) x0 = x
    # If there's one passed, then check for standardization/intercept, and
    # adjust x0 if needed
    if (!is.null(x0)) {
      if (standardize) x0 = scale(x0,bx,sx)
      if (intercept) x0 = cbind(rep(1,nrow(x0)), x0)
    }
  }
  
  # Solve the quantile generalized lasso LPs
  obj = quantile_genlasso_lp(x=x, y=y, d=d, tau=tau, lambda=lambda,
                             weights=weights, noncross=noncross, x0=x0,
                             lp_solver=lp_solver, time_limit=time_limit,
                             warm_starts=warm_starts, params=params,
                             jitter=jitter, verbose=verbose)

  # Transform beta back to original scale, if we standardized
  if (standardize) {
    if (!intercept) obj$beta = rbind(rep(0,length(tau)), obj$beta)
    obj$beta[1,] = obj$beta[1,] - (bx/sx) %*% obj$beta[-1,]
    obj$beta[-1,] = Diagonal(x=1/sx) %*% obj$beta[-1,]
  }

  colnames(obj$beta) = sprintf("tau=%g, lam=%g", round(tau,2), round(lambda,2))
  obj = c(obj, enlist(tau, lambda, weights, intercept, standardize, lp_solver,
                      warm_starts, time_limit, params, transform, inv_trans,
                      jitter))
  class(obj) = "quantile_genlasso"
  return(obj)
}

# Solve quantile generalized lasso problems using an LP solver.

quantile_genlasso_lp = function(x, y, d, tau, lambda, weights, noncross=FALSE,
                                x0=NULL, lp_solver="gurobi", params=list(),
                                warm_starts=TRUE, time_limit=time_limit,
                                jitter=NULL, verbose=FALSE) {
  # Set up some basic objects that we will need
  n = nrow(x); p = ncol(x); m = nrow(d)
  Inn = Diagonal(n); Imm = Diagonal(m)
  Znm = Matrix(0,n,m,sparse=TRUE)
  Zmn = Matrix(0,m,n,sparse=TRUE)
  model = model_big = list()
  N = 2*n + 2*m; P = p + m + n
  r = length(tau); n0 = nrow(x0)
  model$sense = rep(">=", N)
  
  # Determine LP solver
  use_gurobi = FALSE
  if (lp_solver == "gurobi") {
    if (requireNamespace("gurobi", quietly=TRUE)) use_gurobi = TRUE
    else warning("gurobi R package not installed, using Rglpk instead.")
  }

  # Gurobi setup
  if (use_gurobi) {
    if (!is.null(time_limit)) params$TimeLimit = time_limit
    if (is.null(params$LogToConsole)) params$LogToConsole = 0
    # In verbose mode, if there's just one problem or one big noncrossing
    # problem, then display output differently
    if (verbose && (noncross || r == 1)) {
      verbose = FALSE
      params$LogToConsole = 1
    }
    model$lb = c(rep(-Inf,p), rep(0,m), rep(0,n))
  }

  # GLPK setup
  else {
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    # In verbose mode, if there's just one problem or one big noncrossing
    # problem, then display output differently
    if (verbose && (noncross || r == 1)) {
      verbose = FALSE
      params$verbose = TRUE
    }
    model$lb = list(lower=list(ind=1:p, val=rep(-Inf,p)))
  }

  # Noncrossing setup
  if (noncross) {
    model_big$obj = rep(0, P*r)
    model_big$A = Matrix(0, nrow=N*r, ncol=P*r, sparse=TRUE)
    model_big$sense = rep(model$sense, r)
  }
  
  # Loop over tau/lambda values
  beta = Matrix(0, nrow=p, ncol=r, sparse=TRUE)
  status = rep(NA, r)
  last_sol = NULL

  if (verbose) cat(sprintf("Problems solved (of %i): ", r))
  for (k in 1:r) {
    if (verbose && (r <= 10 || k %% 5 == 0)) cat(paste(k, "... "))

    # Apply random jitter, if we're asked to
    if (!is.null(jitter)) yy = y + jitter(n)
    else yy = y

    # Vector of objective coefficients
    model$obj = c(rep(0,p), rep(lambda[k],m), weights)

    # Matrix of constraint coefficients: depends only on tau, so we try to save
    # work if possible (check if we've already created this for last tau value)
    if (k == 1 || tau[k] != tau[k-1]) {
      model$A = rbind(
        cbind(tau[k]*x, Znm, Inn),
        cbind((tau[k]-1)*x, Znm, Inn),
        cbind(-d, Imm, Zmn),
        cbind(d, Imm, Zmn)
      )
    }

    # Right hand side of constraints
    model$rhs = c(tau[k]*y, (tau[k]-1)*y, rep(0,2*m))

    # For noncrossing constraints, save these for later
    if (noncross) {
      model_big$obj[(k-1)*P + 1:P] = model$obj
      model_big$A[(k-1)*N + 1:N, (k-1)*P + 1:P] = model$A
      model_big$rhs[(k-1)*N + 1:N] = model$rhs
    }

    # Otherwise, go ahead and solve the individual LP
    else {
      # Gurobi
      if (use_gurobi) {
        # Set a warm start, if we're asked to
        if (warm_starts && !is.null(last_sol)) {
          model$start = last_sol
        }

        # Call Gurobi's LP solver, store results
        a = gurobi::gurobi(model=model, params=params)
        beta[,k] = a$x[1:p]
        status[k] = a$status
        if (warm_starts) last_sol = a$x
      }

      # GLPK
      else {
        # Call GLPK's LP solver, store results
        a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                           rhs=model$rhs, bounds=model$lb, control=params)
        beta[,k] = a$solution[1:p]
        status[k] = a$status
      }
    }
  }; if (verbose) cat("\n")
  
  # Back to noncrossing constraints, we need to solve one big LP
  if (noncross) {
    # Add to the constraint matrix, and right hand side vector
    B = Matrix(0, nrow=n0*(r-1), ncol=P*r, sparse=TRUE)
    for (k in 1:(r-1)) {
      B[(k-1)*n0 + 1:n0, (k-1)*P + 1:p] = -x0
      B[(k-1)*n0 + 1:n0, k*P + 1:p] = x0
    }
    model_big$A = rbind(model_big$A, B)
    model_big$sense = c(model_big$sense, rep(">=", n0*(r-1)))
    model_big$rhs = c(model_big$rhs, rep(0, n0*(r-1)))

    # Gurobi
    if (use_gurobi) {
      # Extend lower bound parameter
      model_big$lb = rep(model$lb, r)

      # Call Gurobi's LP solver, store results
      a = gurobi::gurobi(model=model_big, params=params)
      beta = Matrix(a$x[rep(1:p, r) + rep(P*0:(r-1), each=p)], nrow=p, ncol=r, 
                    sparse=TRUE)  
      status = a$status
    }
    
    # GLPK
    else {
      # Extend lower bound parameter
      model_big$lb = list(lower=list(ind=rep(1:p, r) + rep(P*0:(r-1), each=p),
                                     val=rep(-Inf, p*r)))
      
      # Call GLPK's LP solver, store results
      a = Rglpk_solve_LP(obj=model_big$obj, mat=model_big$A,
                         dir=model_big$sense, rhs=model_big$rhs,
                         bounds=model_big$lb, control=params) 
      beta = Matrix(a$solution[rep(1:p, r) + rep(P*0:(r-1), each=p)], nrow=p,
                    ncol=r, sparse=TRUE) 
      status = a$status
    }
  }
  
  return(enlist(beta, status))
}

##############################

#' Coef function for quantile_genlasso object
#'
#' Retrieve generalized lasso coefficients for estimating the conditional
#' quantiles at specified tau or lambda values.
#'
#' @param object The \code{quantile_genlasso} object.
#' @param s Vector of integers specifying the tau and lambda values to consider
#'   for coefficients; for each \code{i} in this vector, coefficients are
#'   returned at quantile level \code{tau[i]} and tuning parameter value
#'   \code{lambda[i]}, according to the \code{tau} and \code{lambda} vectors
#'   stored in the given \code{quantile_genlasso} object \code{obj}. (Said
#'   differently, \code{s} specifies the columns of \code{obj$beta} to retrieve
#'   for the coefficients.)  Default is NULL, which means that all tau and
#'   lambda values will be considered.
#' @param ... Additional arguments (not used).
#' 
#' @method coef quantile_genlasso
#' @export

coef.quantile_genlasso = function(object, s=NULL, ...) {
  if (is.null(s)) s = 1:ncol(object$beta)
  return(object$beta[,s])
}

##############################

#' Predict function for quantile_genlasso object
#'
#' Predict the conditional quantiles at a new set of predictor variables, using
#' the generalized lasso coefficients at specified tau or lambda values.
#'
#' @param object The \code{quantile_genlasso} object.
#' @param newx Matrix of new predictor variables at which predictions should
#'   be made.
#' @param s Vector of integers specifying the tau and lambda values to consider
#'   for predictions; for each \code{i} in this vector, predictions are made at
#'   quantile level \code{tau[i]} and tuning parameter value \code{lambda[i]},
#'   according to the \code{tau} and \code{lambda} vectors stored in the given
#'   \code{quantile_genlasso} object \code{obj}. (Said differently, \code{s}
#'   specifies the columns of \code{object$beta} to use for the predictions.)
#'   Default is NULL, which means that all tau and lambda values will be
#'   considered.
#' @param sort Should the returned quantile estimates be sorted? Default is
#'   FALSE. Note: this option only makes sense if the values in the stored
#'   \code{tau} vector are distinct, and sorted in increasing order.
#' @param iso Should the returned quantile estimates be passed through isotonic
#'   regression? Default is FALSE; if TRUE, takes priority over \code{sort}.
#'   Note: this option only makes sense if the values in the stored \code{tau}
#'   vector are distinct, and sorted in increasing order.
#' @param nonneg Should the returned quantile estimates be truncated at 0?
#'   Natural for count data. Default is FALSE.
#' @param round Should the returned quantile estimates be rounded? Natural for
#'   count data. Default is FALSE.
#' @param ... Additional arguments (not used).
#' 
#' @method predict quantile_genlasso
#' @export

predict.quantile_genlasso = function(object, newx, s=NULL, sort=FALSE,
                                     iso=FALSE, nonneg=FALSE, round=FALSE,
                                     ...) {
  # Set up some basics
  if (!is.matrix(newx)) newx = matrix(newx, nrow=1)
  n0 = nrow(newx)
  if (object$intercept || object$standardize) newx = cbind(rep(1,n0), newx) 
  z = as.matrix(newx %*% coef(object,s))

  # Apply the inverse transform, if we're asked to
  if (!is.null(object$inv_trans)) {
    # Annoying, must handle carefully the case that z drops to a vector
    names = colnames(z)
    z = apply(z, 2, object$inv_trans)
    z = matrix(z, nrow=n0)
    colnames(z) = names
  }

  # Run isotonic regression, sort, truncated, round, if we're asked to
  for (i in 1:n0) {
    o = which(!is.na(z[i,]))
    if (sort && !iso) z[i,o] = sort(z[i,o])
    if (iso) z[i,o] = isoreg(z[i,o])$yf
  }
  if (nonneg) z = pmax(z,0)
  if (round) z = round(z)
  return(z)
}

##############################

#' Quantile generalized lasso on a tau by lambda grid
#'
#' Convenience function for computing quantile generalized lasso solutions on a
#' tau by lambda grid.
#'
#' @param nlambda Number of lambda values to consider, for each quantile
#'   level. Default is 30.
#' @param lambda_min_ratio Ratio of the minimum to maximum lambda value, for
#'   each quantile levels. Default is 1e-3.
#'
#' @details This function forms a \code{lambda} vector either determined by the
#'   \code{nlambda} and \code{lambda_min_ratio} arguments, or the \code{lambda}
#'   argument; if the latter is specified, then it takes priority. Then, for
#'   each \code{i} and \code{j}, we solve a separate quantile generalized lasso
#'   problem at quantile level \code{tau[i]} and tuning parameter value
#'   \code{lambda[j]}, using the \code{quantile_genlasso} function. All
#'   arguments (aside from \code{nlambda} and \code{lambda_min_ratio}) are as in
#'   the latter function; noncrossing constraints are disallowed.
#'
#' @export

quantile_genlasso_grid = function(x, y, d, tau, lambda=NULL, nlambda=30,
                                  lambda_min_ratio=1e-3, weights=NULL,
                                  intercept=TRUE, standardize=TRUE,
                                  lp_solver=c("glpk","gurobi"), time_limit=NULL,
                                  warm_starts=TRUE, params=list(),
                                  transform=NULL, inv_trans=NULL, jitter=NULL,
                                  verbose=FALSE) {
  # Check LP solver
  lp_solver = match.arg(lp_solver)

  # Set the lambda sequence, if we need to
  if (is.null(lambda)) {
    lambda = get_lambda_seq(x=x, y=y, d=d, nlambda=nlambda,
                            lambda_min_ratio=lambda_min_ratio, weights=weights,
                            intercept=intercept, standardize=standardize,
                            lp_solver=lp_solver, transform=transform)
  }

  # Create the grid: stack the problems so that tau is constant and lambda is
  # changing from one to the next, the way we've setup the LP solver, this will
  # be better for memory purposes (and also for warm starts?)
  tau = rep(tau, each=length(lambda))
  lambda = rep(lambda, length(unique(tau)))

  # Now just call quantile_genlasso
  obj = quantile_genlasso(x=x, y=y, d=d, tau=tau, lambda=lambda,
                          weights=weights, intercept=intercept,
                          standardize=standardize, noncross=FALSE, x0=NULL,
                          lp_solver=lp_solver, time_limit=time_limit,
                          warm_starts=warm_starts, params=params,
                          transform=transform, inv_trans=inv_trans,
                          jitter=jitter, verbose=verbose)
  class(obj) = c("quantile_genlasso_grid", class(obj))
  return(obj)
}

##############################

#' Lambda max for quantile generalized lasso
#'
#' Compute lambda max for a quantile generalized lasso problem.
#'
#' @details This is not exact, but should be close to the exact value of
#'   \eqn{\lambda} such that \eqn{D \hat\beta = 0} at the solution
#'   \eqn{\hat\beta} of the quantile generalized lasso problem. It is derived
#'   from the KKT conditions when \eqn{\tau = 1/2}.
#'
#' @export

get_lambda_max = function(x, y, d, weights=NULL, lp_solver=c("glpk","gurobi")) {
  # Set up some basic objects that we will need
  n = nrow(x); p = ncol(x); m = nrow(d)
  if (is.null(weights)) weights = rep(1,n)
  lp_solver = match.arg(lp_solver)
  Zmm = Matrix(0,m,m,sparse=TRUE); Imm = Diagonal(m)
  model = list()

  # First solve the constrained regression problem
  mat = rbind(cbind(t(x) %*% x, Matrix::t(d)), cbind(d, Zmm))
  b = solve(mat, c(t(x) %*% y, rep(0,m)))
  v = weights * sign(y - x %*% b[1:p]) / 2

  # Next remove zero columns from D (and drop from X)
  o = which(apply(d, 2, function(v) all(abs(v) <= sqrt(.Machine$double.eps))))
  if (length(o) > 0) {
    d = d[,-o]
    x = x[,-o]
    p = p - length(o)
  }

  # Finally solve the LP
  model$obj = c(rep(0,m), 1)
  model$A = rbind(cbind(Imm, rep(1,m)),
                  cbind(-Imm, rep(1,m)),
                  cbind(Matrix::t(d), rep(0,p)))
  model$rhs = c(rep(0, 2*m), t(x) %*% v)

  # Determine LP solver
  use_gurobi = FALSE
  if (lp_solver == "gurobi") {
    if (requireNamespace("gurobi", quietly=TRUE)) use_gurobi = TRUE
    else warning("gurobi R package not installed, using Rglpk instead.")
  }
  
  # Gurobi
  if (use_gurobi) {
    model$sense = c(rep(">=", 2*m), rep("=", p))
    model$lb = c(rep(-Inf,m), 0)
    a = gurobi::gurobi(model=model, params=list(LogToConsole=0)) 
    lambda_max = a$x[m+1]
  }

  # GLPK
  else {
    model$sense = c(rep(">=", 2*m), rep("==", p))
    model$lb = list(lower=list(ind=1:m, val=rep(-Inf,m)))
    a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                       rhs=model$rhs, bounds=model$lb)
    lambda_max = a$solution[m+1]
  }

  return(lambda_max)
}

#' Lambda sequence for quantile generalized lasso
#'
#' Compute a lambda sequence for a quantile generalized lasso problem.
#'
#' @details This function returns \code{nlambda} values log-spaced in between
#'   \code{lambda_max}, as computed by \code{get_lambda_max}, and
#'   \code{lamdba_max * lambda_min_ratio}. If \code{d} is not specified, we will
#'   set it equal to the identity (hence interpret the problem as a quantile
#'   lasso problem).
#'
#' @export

get_lambda_seq = function(x, y, d, nlambda, lambda_min_ratio, weights=NULL,
                          intercept=TRUE, standardize=TRUE,
                          lp_solver=c("glpk","gurobi"), transform=NULL) {
  # Check LP solver
  lp_solver = match.arg(lp_solver)

  # Set up x, y, d
  a = setup_xyd(x, y, d, weights, intercept, standardize, transform)
  x = a$x; y = a$y; d = a$d; weights = a$weights

  # Compute lambda max then form and return a lambda sequence
  lambda_max = get_lambda_max(x, y, d, weights, lp_solver)
  return(exp(seq(log(lambda_max), log(lambda_max * lambda_min_ratio),
                 length=nlambda)))
}

##############################

#' Predict function for quantile_genlasso_grid object
#'
#' Predict the conditional quantiles at a new set of predictor variables, using
#' the generalized lasso coefficients at given tau or lambda values.
#'
#' @details This function operates as in the \code{predict.quantile_genlasso}
#'   function for a \code{quantile_genlasso} object, but with a few key
#'   differences. First, the output is reformatted so that it is an array of
#'   dimension (number of prediction points) x (number of tuning parameter
#'   values) x (number of quantile levels). This output is generated from the
#'   full set of tau and lambda pairs stored in the given
#'   \code{quantile_genlasso_grid} object \code{obj} (selecting a subset is
#'   disallowed). Second, the arguments \code{sort} and \code{iso} operate on
#'   the appropriate slices of this array: for a fixed lambda value, we sort or
#'   run isotonic regression across all tau values.
#'
#' @method predict quantile_genlasso_grid
#' @export

predict.quantile_genlasso_grid = function(object, newx, sort=FALSE, iso=FALSE,  
                                          nonneg=FALSE, round=FALSE, ...) { 
  # Set up some basics
  if (!is.matrix(newx)) newx = matrix(newx, nrow=1)
  n0 = nrow(newx)
  if (object$intercept || object$standardize) newx = cbind(rep(1,n0), newx) 
  z = as.matrix(newx %*% coef(object))

  # Apply the inverse transform, if we're asked to
  if (!is.null(object$inv_trans)) {
    # Annoying, must handle carefully the case that z drops to a vector
    names = colnames(z)
    z = apply(z, 2, object$inv_trans)
    z = matrix(z, nrow=n0)
    colnames(z) = names
  }

  # Now format into an array
  z = array(z, dim=c(n0, length(unique(object$lambda)),
                     length(unique(object$tau)))) 
  dimnames(z)[[2]] = sprintf("lam=%g", round(unique(object$lambda),2))
  dimnames(z)[[3]] = sprintf("tau=%g", round(unique(object$tau),2))

  # Run isotonic regression, sort, truncated, round, if we're asked to
  for (i in 1:dim(z)[1]) {
    for (j in 1:dim(z)[2]) {
      o = which(!is.na(z[i,j,]))
      if (sort && !iso) z[i,j,o] = sort(z[i,j,o])
      if (iso) z[i,j,o] = isoreg(z[i,j,o])$yf
    }
  }
  if (nonneg) z = pmax(z,0)
  if (round) z = round(z)
  return(z)
}

##############################

#' Quantile generalized lasso objective
#'
#' Compute generalized lasso objective for a single tau and lambda value.
#'
#' @export

quantile_genlasso_objective = function(x, y, d, beta, tau, lambda) {
  loss = quantile_loss(x %*% beta, y, tau)
  pen = lambda * sum(abs(d %*% beta))
  return(loss + pen)
}
