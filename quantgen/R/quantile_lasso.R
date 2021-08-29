#' Quantile lasso
#'
#' Compute quantile lasso solutions.
#'
#' @param x Matrix of predictors. If sparse, then passing it an appropriate 
#'   sparse \code{Matrix} class can greatly help optimization. 
#' @param y Vector of responses.
#' @param tau,lambda Vectors of quantile levels and tuning parameter values. If
#'   these are not of the same length, the shorter of the two is recycled so
#'   that they become the same length. Then, for each \code{i}, we solve a
#'   separate quantile lasso problem at quantile level \code{tau[i]} and tuning
#'   parameter value \code{lambda[i]}. The most common use cases are: specifying
#'   one tau value and a sequence of lambda values; or specifying a sequence of
#'   tau values and one lambda value.
#' @param weights Vector of observation weights (to be used in the loss
#'   function). Default is NULL, which is interpreted as a weight of 1 for each
#'   observation.  
#' @param no_pen_vars Indices of the variables that should be excluded from the
#'   lasso penalty. Default is \code{c()}, which means that no variables are to
#'   be excluded. 
#'
#' @return A list with the following components:
#'   \item{beta}{Matrix of lasso coefficients, of dimension = (number of
#'   features + 1) x (number of quantile levels) assuming \code{intercept=TRUE},
#'   else (number of features) x (number of quantile levels). Note: these
#'   coefficients will always be on the appropriate scale; they are always on
#'   the scale of original features, even if \code{standardize=TRUE}}
#'   \item{status}{Vector of status flags returned by Gurobi's or GLPK's LP
#'   solver, of length = (number of quantile levels)}
#'   \item{tau,lambda}{Vectors of tau and lambda values used}
#'   \item{weights,no_pen_vars,...,jitter}{Values of these other arguments  
#'   used in the function call}  
#'
#' @details This function solves the quantile lasso problem, for each pair of
#'   quantile level \eqn{\tau} and tuning parameter \eqn{\lambda}:  
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_0,\beta} \;
#'   \sum_{i=1}^n w_i \psi_\tau(y_i-\beta_0-x_i^T\beta) + \lambda \|\beta\|_1}    
#'   for a response vector \eqn{y} with components \eqn{y_i}, and predictor
#'   matrix \eqn{X} with rows \eqn{x_i}. Here \eqn{\psi_\tau(v) = \max\{\tau v, 
#'   (\tau-1) v\}} is the "pinball" or "tilted \eqn{\ell_1}" loss. When
#'   noncrossing constraints are applied, we instead solve one big joint
#'   optimization, over all quantile levels and tuning parameter values: 
#'   \deqn{\mathop{\mathrm{minimize}}_{\beta_{0k}, \beta_k, k=1,\ldots,r} \; 
#'   \sum_{k=1}^r \bigg(\sum_{i=1}^n w_i \psi_{\tau_k}(y_i-\beta_{0k}-
#'   x_i^T\beta_k) + \lambda_k \|\beta_k\|_1\bigg)} 
#'   \deqn{\mathrm{subject \; to} \;\; \beta_{0k}+x^T\beta_k \leq
#'   \beta_{0,k+1}+x^T\beta_{k+1} \;\; k=1,\ldots,r-1, \; x \in \mathcal{X}}
#'   where the quantile levels \eqn{\tau_j, j=1,\ldots,k} are assumed to be in
#'   increasing order, and \eqn{\mathcal{X}} is a collection of points over
#'   which to enforce the noncrossing constraints.
#'
#'   Either problem is readily converted into a linear program (LP), and solved
#'   using either Gurobi (which is free for academic use, and generally fast) or
#'   GLPK (which free for everyone, but slower).
#'
#'   All arguments not described above are as in the \code{quantile_genlasso}
#'   function. The associated \code{coef} and \code{predict} functions are just
#'   those for the \code{quantile_genlasso} class.  
#'
#' @author Ryan Tibshirani
#' @export

quantile_lasso = function(x, y, tau, lambda, weights=NULL, no_pen_vars=c(), 
                          intercept=TRUE, standardize=TRUE, lb=-Inf, ub=Inf,
                          noncross=FALSE, x0=NULL, lp_solver=c("glpk","gurobi"), 
                          time_limit=NULL, warm_starts=TRUE, params=list(),
                          transform=NULL, inv_trans=NULL, jitter=NULL,
                          verbose=FALSE) {
  # Define an identity penalty matrix 
  d = Diagonal(ncol(x))
  if (length(no_pen_vars) > 0) d = d[-no_pen_vars,]

  # Now just call quantile_genlasso
  obj = quantile_genlasso(x=x, y=y, d=d, tau=tau, lambda=lambda,
                          weights=weights, intercept=intercept,
                          standardize=standardize, lb=lb, ub=ub,
                          noncross=noncross, x0=x0, lp_solver=lp_solver,
                          time_limit=time_limit, warm_starts=warm_starts,
                          params=params, transform=transform,
                          inv_trans=inv_trans, jitter=jitter, verbose=verbose)
  class(obj) = c("quantile_lasso", class(obj))
  return(obj)
}

##############################

#' Quantile lasso on a tau by lambda grid
#'
#' Convenience function for computing quantile lasso solutions on a tau by
#' lambda grid.   
#'
#' @param nlambda Number of lambda values to consider, for each quantile
#'   level. Default is 30.  
#' @param lambda_min_ratio Ratio of the minimum to maximum lambda value, for
#'   each quantile levels. Default is 1e-3.
#'
#' @details This function forms a \code{lambda} vector either determined by the
#'   \code{nlambda} and \code{lambda_min_ratio} arguments, or the \code{lambda}
#'   argument; if the latter is specified, then it takes priority. Then, for
#'   each \code{i} and \code{j}, we solve a separate quantile lasso problem at
#'   quantile level \code{tau[i]} and tuning parameter value \code{lambda[j]},
#'   using the \code{quantile_lasso} function. All arguments (aside from
#'   \code{nlambda} and \code{lambda_min_ratio}) are as in the latter function;
#'   noncrossing constraints are disallowed. The associated \code{predict}
#'   function is just that for the \code{quantile_genlasso_grid} class.
#' 
#' @export

quantile_lasso_grid = function(x, y, tau, lambda=NULL, nlambda=30,
                               lambda_min_ratio=1e-3, weights=NULL,
                               no_pen_vars=c(), intercept=TRUE,
                               standardize=TRUE, lb=-Inf, ub=Inf,
                               lp_solver=c("glpk","gurobi"), time_limit=NULL,
                               warm_starts=TRUE, params=list(), transform=NULL,
                               inv_trans=NULL, jitter=NULL, verbose=FALSE) {
  # Define an identity penalty matrix 
  d = Diagonal(ncol(x))
  if (length(no_pen_vars) > 0) d = d[-no_pen_vars,]

  # Now just call quantile_genlasso_grid
  obj = quantile_genlasso_grid(x=x, y=y, d=d, tau=tau, lambda=lambda,
                               nlambda=nlambda,
                               lambda_min_ratio=lambda_min_ratio,
                               weights=weights, intercept=intercept,
                               standardize=standardize, lb=lb, ub=ub,
                               lp_solver=lp_solver, time_limit=time_limit,
                               warm_starts=warm_starts, params=params,
                               transform=transform, inv_trans=inv_trans,
                               jitter=jitter, verbose=verbose)
  class(obj) = c("quantile_lasso_grid", class(obj))
  return(obj)
}

##############################

#' Cross-validation for quantile lasso
#'
#' Run cross-validation for the quantile lasso on a tau by lambda grid. For each
#' tau, the lambda value minimizing the cross-validation error is reported.  
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
#'   \item{qgl_obj}{A \code{quantile_lasso} object obtained by fitting on the
#'   full training set, at all quantile levels and their corresponding optimal  
#'   lambda values}
#'   \item{cv_mat}{Matrix of cross-validation errors (as measured by quantile
#'   loss), of dimension (number of tuning parameter values) x (number of
#'   quantile levels)}
#'   \item{lambda_min}{Vector of optimum lambda values, one per quantile level}
#'
#' @details All arguments through \code{verbose} (except for \code{nfolds} and
#'   \code{train_test_inds}) are as in \code{quantile_lasso_grid} and
#'   \code{quantile_lasso}. Note that the \code{noncross} and \code{x0}
#'   arguments are not passed to \code{quantile_lasso_grid} for the calculation
#'   of cross-validation errors and optimal lambda values; they are only passed
#'   to \code{quantile_lasso} for the final object that is fit to the full
#'   training set. Past \code{verbose}, the arguments are as in
#'   \code{predict.quantile_lasso}, and control what happens with the
#'   predictions made on the validation sets. The associated \code{predict}
#'   function is just that for the \code{cv_quantile_genlasso} class.   
#' 
#' @export

cv_quantile_lasso = function(x, y, tau, lambda=NULL, nlambda=30,
                             lambda_min_ratio=1e-3, weights=NULL,
                             no_pen_vars=c(), nfolds=5, train_test_inds=NULL,
                             intercept=TRUE, standardize=TRUE, lb=-Inf, ub=Inf,
                             noncross=FALSE, x0=NULL,
                             lp_solver=c("glpk","gurobi"), time_limit=NULL,
                             warm_starts=TRUE, params=list(), transform=NULL,
                             inv_trans=NULL, jitter=NULL, verbose=FALSE,
                             sort=FALSE, iso=FALSE, nonneg=FALSE, round=FALSE) {
  # Define an identity penalty matrix 
  d = Diagonal(ncol(x))
  if (length(no_pen_vars) > 0) d = d[-no_pen_vars,]

  # Now just call cv_quantile_genlasso
  obj = cv_quantile_genlasso(x=x, y=y, d=d, tau=tau, lambda=lambda,
                             nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
                             weights=weights, nfolds=nfolds,
                             train_test_inds=train_test_inds,
                             intercept=intercept, standardize=standardize,
                             lb=lb, ub=ub, noncross=noncross, x0=x0,
                             lp_solver=lp_solver, time_limit=time_limit,
                             warm_starts=warm_starts, params=params,
                             transform=transform, inv_trans=inv_trans,
                             jitter=jitter, verbose=verbose, sort=sort, iso=iso,
                             nonneg=nonneg, round=round)
  class(obj) = c("cv_quantile_lasso", class(obj))
  return(obj)
}

##############################

#' Refit function for cv_quantile_lasso object
#'
#' Refit lasso solutions at a new set of quantile levels, given an existing
#' \code{cv_quantile_lasso} object. 
#'
#' @param obj The \code{cv_quantile_lasso} object to start from.
#' @param x Matrix of predictors.
#' @param y Vector of responses.
#' @param tau_new Vector of new quantile levels at which to fit new solutions.  
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
#' @return A \code{quantile_lasso} object, with solutions at quantile levels
#'   \code{tau_new}. 
#' 
#' @details This function simply infers, for each quantile level in
#'   \code{tau_new}, a (very) roughly-CV-optimal tuning parameter value, then   
#'   calls \code{quantile_lasso} at the new quantile levels and corresponding 
#'   tuning parameter values. If not specified, the arguments \code{weights},
#'   \code{no_pen_vars}, \code{intercept}, \code{standardize}, \code{lp_solver},
#'   \code{time_limit}, \code{warm_start}, \code{params}, \code{transform},
#'   \code{inv_transorm}, \code{jitter} are all inherited from the given
#'   \code{cv_quantile_lasso} object. 
#' 
#' @export

refit_quantile_lasso = function(obj, x, y, tau_new, weights=NULL,
                                no_pen_vars=NULL, intercept=NULL,
                                standardize=NULL, lb=NULL, ub=NULL,
                                noncross=FALSE, x0=NULL, lp_solver=NULL,
                                time_limit=NULL, warm_starts=NULL, params=NULL,
                                transform=NULL, inv_trans=NULL, jitter=NULL,
                                verbose=FALSE) {
  # Define an identity penalty matrix 
  d = Diagonal(ncol(x))
  if (length(no_pen_vars) > 0) d = d[-no_pen_vars,]

  # Now just call refit_quantile_genlasso
  ql_obj = refit_quantile_genlasso(obj=obj, x=x, y=y, d=d, tau_new=tau_new,
                                   weights=weights, intercept=intercept,
                                   standardize=standardize, lb=lb, ub=ub,
                                   noncross=noncross, x0=x0,
                                   lp_solver=lp_solver, time_limit=time_limit,
                                   warm_starts=warm_starts, params=params,
                                   transform=transform, inv_trans=inv_trans,
                                   jitter=jitter, verbose=verbose)
  class(ql_obj) = c("quantile_lasso", class(ql_obj))
  return(ql_obj)
}

##############################

#' Quantile lasso objective
#'
#' Compute lasso objective for a single tau and lambda value.    
#'
#' @export

quantile_lasso_objective = function(x, y, beta, tau, lambda) {  
  loss = quantile_loss(x %*% beta, y, tau)
  pen = lambda * sum(abs(beta))
  return(loss + pen)
}
