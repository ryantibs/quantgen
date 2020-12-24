#' Cross-validation for quantile generalized lasso
#'
#' Run cross-validation for the quantile generalized lasso on a tau by lambda
#' grid. For each tau, the lambda value minimizing the cross-validation error is
#' reported.
#'
#' @param nfolds Number of cross-validation folds. Default is 5.
#' @param train_test_inds List of length two, with components named \code{train}
#'   and \code{test}. Each of \code{train} and \code{test} are themselves lists,
#'   of the same length; for each \code{i}, we will consider \code{train[[i]]}
#'   the indices (which index the rows of \code{x} and elements of \code{y}) to
#'   use for training, and \code{test[[i]]} as the indices to use for testing
#'   (validation). The validation error will then be summed up over all
#'   \code{i}. This allows for fine control of the "cross-validation" process
#'   (in quotes, because there need not be any crossing going on here). Default
#'   is NULL; if specified, takes priority over \code{nfolds}.
#'
#' @return A list with the following components:
#'   \item{qgl_obj}{A \code{quantile_genlasso} object obtained by fitting on the
#'   full training set, at all quantile levels and their corresponding optimal
#'   lambda values}
#'   \item{cv_mat}{Matrix of cross-validation errors (as measured by quantile
#'   loss), of dimension (number of tuning parameter values) x (number of
#'   quantile levels)}
#'   \item{lambda_min}{Vector of optimum lambda values, one per quantile level}
#'   \item{tau,lambda}{Vectors of tau and lambda values used}
#'
#' @details All arguments through \code{verbose} (except for \code{nfolds} and
#'   \code{train_test_inds}) are as in \code{quantile_genlasso_grid} and
#'   \code{quantile_genlasso}. Past \code{verbose}, the arguments are as in
#'   \code{predict.quantile_genlasso}, and control what happens with the
#'   predictions made on the validation sets.
#'
#' @importFrom graphics legend matplot
#' @importFrom  stats approx coef isoreg median predict qnorm runif sd splinefun
#' @export

cv_quantile_genlasso = function(x, y, d, tau, lambda=NULL, nlambda=30,
                                lambda_min_ratio=1e-3, weights=NULL, nfolds=5,
                                train_test_inds=NULL, intercept=TRUE,
                                standardize=TRUE, lp_solver=c("glpk","gurobi"),
                                time_limit=NULL, warm_starts=TRUE,
                                params=list(), transform=NULL, inv_trans=NULL,
                                jitter=NULL, verbose=FALSE, sort=FALSE,
                                iso=FALSE, nonneg=FALSE, round=FALSE) {
  # Set up some basics
  n = nrow(x); p = ncol(x); m = nrow(d)
  if (is.null(weights)) weights = rep(1,n)
  lp_solver = match.arg(lp_solver)

  # Set the lambda sequence, if we need to
  if (is.null(lambda)) {
    lambda = get_lambda_seq(x=x, y=y, d=d, nlambda=nlambda,
                            lambda_min_ratio=lambda_min_ratio, weights=weights,
                            standardize=standardize, intercept=intercept,
                            lp_solver=lp_solver, transform=transform)
  }

  # Grab the specified training and test inds, or else build them
  if (!is.null(train_test_inds)) {
    train = train_test_inds$train
    test = train_test_inds$test
    nfolds = length(train)
  }
  else {
    folds = rep(1:nfolds, n)[sample(n)]
    train = test = vector(mode="list", length=nfolds)
    for (k in 1:nfolds) {
      train[[k]] = which(folds != k)
      test[[k]] = which(folds == k)
    }
  }

  yhat = array(NA, dim=c(n, length(lambda), length(tau)))
  for (k in 1:nfolds) {
    if (verbose) cat(sprintf("CV fold %i ...\n", k))
    # Fit on training set
    obj = quantile_genlasso_grid(x=x[train[[k]],,drop=FALSE], y=y[train[[k]]],
                                 d=d, tau=tau, lambda=lambda, nlambda=NULL,
                                 lambda_min_ratio=NULL,
                                 weights=weights[train[[k]]],
                                 intercept=intercept, standardize=standardize,
                                 lp_solver=lp_solver, time_limit=time_limit,
                                 warm_starts=warm_starts, params=params,
                                 transform=transform, inv_trans=inv_trans,
                                 jitter=jitter, verbose=verbose)
    # Predict on test set
    yhat[test[[k]],,] = predict(obj, x[test[[k]],,drop=FALSE], sort=sort,
                                iso=iso, nonneg=nonneg, round=round)
  }

  # Record CV errors, according to appropriate quantile loss function
  if (verbose) cat("Computing CV errors and optimum lambdas ...\n")
  cv_mat = matrix(NA, length(lambda), length(tau))
  lambda_min = rep(NA, length(tau))
  for (j in 1:length(tau)) {
    num_test_j = colSums(!is.na(yhat[,,j]))
    cv_mat[,j] = quantile_loss(yhat[,,j], y, tau[j]) / num_test_j
    lambda_min[j] = lambda[which.min(cv_mat[,j])]
  }

  # Adjustment factor for optimum lambdas, accounting for training set sizes
  # within CV
  adj = n / sapply(train, length)
  
  # Fit quantile genlasso object on full training set, with optimum lambdas
  if (verbose) cat("Refitting on full training set with optimum lambdas ...\n")
  qgl_obj = quantile_genlasso(x=x, y=y, d=d, tau=tau, lambda=lambda*adj,
                              weights=weights, intercept=intercept,
                              standardize=standardize, noncross=FALSE, x0=NULL,
                              lp_solver=lp_solver, time_limit=time_limit,
                              warm_starts=warm_starts, params=params,
                              transform=transform, inv_trans=inv_trans,
                              jitter=jitter, verbose=verbose)
  obj = enlist(qgl_obj, cv_mat, lambda_min, tau, lambda, adj)
  class(obj) = "cv_quantile_genlasso"
  return(obj)
}

##############################

#' Plot function for quantile_genlasso object
#'
#' Plot the cross-validation error curves, for each quantile level, as functions
#' of the tuning parameter value.
#'
#' @param x The \code{cv_quantile_genlasso} object.
#' @param legend_pos Position for the legend; default is "topleft"; use NULL to
#'   suppress the legend.
#' @param ... Additional arguments (not used).
#' 
#' @method plot cv_quantile_genlasso
#' @export

plot.cv_quantile_genlasso = function(x, legend_pos="topleft", ...) {
  matplot(x$lambda, x$cv_mat, type="o", lty=1:5, col=1:6, pch=20, log="x",
          ylab="CV error", xlab="Lambda")
  if (!is.null(legend_pos)) legend(legend_pos, legend=paste("tau =", x$tau),
                                   lty=1:5, col=1:6)
}

##############################

#' Predict function for cv_quantile_genlasso object
#'
#' Predict the conditional quantiles at a new set of predictor variables, using
#' the generalized lasso coefficients tuned by cross-validation.
#'
#' @details This just calls the \code{predict} function on the
#'   \code{quantile_genlasso} that is stored within the given
#'   \code{cv_quantile_genlasso} object.
#'
#' @method predict cv_quantile_genlasso
#' @export

predict.cv_quantile_genlasso = function(object, newx, s=NULL, sort=FALSE,
                                        iso=FALSE, nonneg=FALSE, round=FALSE,
                                        ...) {
  return(predict(object$qgl_obj, newx=newx, s=s, sort=sort, iso=iso,
                 nonneg=nonneg, round=round))
}

##############################

#' Refit function for cv_quantile_genlasso object
#'
#' Refit generalized lasso solutions at a new set of quantile levels, given
#' an existing \code{cv_quantile_genlasso} object.
#'
#' @param obj The \code{cv_quantile_genlasso} object to start from.
#' @param x Matrix of predictors.
#' @param y Vector of responses.
#' @param d Matrix defining the generalized lasso penalty.
#' @param tau_new Vector of new quantile levels at which to fit new
#'   solutions. Default is a sequence of 23 quantile levels from 0.01 to 0.99.
#' @param noncross Should noncrossing constraints be applied? These force the
#'   estimated quantiles to be properly ordered across all quantile levels being
#'   considered. The default is FALSE. If TRUE, then noncrossing constraints are
#'   applied to the estimated quantiles at all points specified by the next
#'   argument \code{x0}.
#' @param x0 Matrix of points used to define the noncrossing
#'   constraints. Default is NULL, which means that we consider noncrossing
#'   constraints at the training points \code{x}.
#' @param verbose Should progress be printed out to the console? Default is
#'   FALSE.
#'
#' @return A \code{quantile_genlasso} object, with solutions at quantile levels
#'   \code{tau_new}.
#'
#' @details This function simply infers, for each quantile level in
#'   \code{tau_new}, a (very) roughly-CV-optimal tuning parameter value, then
#'   calls \code{quantile_genlasso} at the new quantile levels and corresponding
#'   tuning parameter values. If not specified, the arguments \code{weights},
#'   \code{intercept}, \code{standardize}, \code{lp_solver}, \code{time_limit},
#'   \code{warm_starts}, \code{params}, \code{transform}, \code{inv_transorm},
#'   \code{jitter} are all inherited from the given \code{cv_quantile_genlasso}
#'   object.
#'
#' @export

refit_quantile_genlasso = function(obj, x, y, d, tau_new=c(0.01, 0.025,
                                   seq(0.05, 0.95, by=0.05), 0.975, 0.99),
                                   weights=NULL, intercept=TRUE,
                                   standardize=TRUE, noncross=FALSE, x0=NULL,
                                   lp_solver=NULL, time_limit=NULL,
                                   warm_starts=NULL, params=NULL,
                                   transform=NULL, inv_trans=NULL, jitter=NULL,
                                   verbose=FALSE) {
  # For each new tau, find the nearest tau, and use its CV-optimal lambda
  tau = obj$tau
  lambda = obj$lambda_min * adj # Adjust for training set sizes within CV
  tau_mat = matrix(rep(tau, length(tau_new)), nrow=length(tau))
  tau_new_mat = matrix(rep(tau_new, each=length(tau)), nrow=length(tau))
  lambda_new = lambda[max.row(-abs(tau_mat - tau_new_mat))]

  # If not specified, inherit from the stored object
  if (is.null(weights)) weights = obj$qgl_obj$weights
  if (is.null(intercept)) intercept = obj$qgl_obj$intercept
  if (is.null(standardize)) standardize = obj$qgl_obj$standardize
  if (is.null(lp_solver)) lp_solver = obj$qgl_obj$lp_solver
  if (is.null(time_limit)) time_limit = obj$qgl_obj$time_limit
  if (is.null(warm_starts)) warm_starts = obj$qgl_obj$warm_starts
  if (is.null(params)) params = obj$qgl_obj$params
  if (is.null(transform)) transform = obj$qgl_obj$transform
  if (is.null(inv_trans)) inv_trans = obj$qgl_obj$inv_trans
  if (is.null(jitter)) jitter = obj$qgl_obj$jitter

  # Now just call quantile_genlasso
  return(quantile_genlasso(x=x, y=y, d=d, tau=tau_new, lambda=lambda_new,
                           weights=weights, intercept=intercept,
                           standardize=standardize, noncross=noncross, x0=x0,
                           lp_solver=lp_solver, time_limit=time_limit,
                           warm_starts=warm_starts, params=params,
                           transform=transform, inv_trans=inv_trans,
                           jitter=jitter, verbose=verbose))
}
