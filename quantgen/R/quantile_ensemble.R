#' Quantile ensemble
#'
#' Fit ensemble weights, given a set of quantile predictions.
#'
#' @param qarr Array of predicted quantiles, of dimension (number of prediction
#'   points) x (number or ensemble components) x (number of quantile levels).
#' @param y Vector of responses (whose quantiles are being predicted by
#'   \code{qarr}).
#' @param tau Vector of quantile levels at which predictions are made. Assumed
#'   to be distinct, and sorted in increasing order.
#' @param weights Vector of observation weights (to be used in the loss
#'   function). Default is NULL, which is interpreted as a weight of 1 for each
#'   observation.
#' @param tau_groups Vector of group labels, having the same length as
#'   \code{tau}. Common labels indicate that the ensemble weights for the
#'   corresponding quantile levels should be tied together. Default is
#'   \code{rep(1,length(tau))}, which means that a common set of ensemble
#'   weights should be used across all levels. See details.
#' @param intercept Should an intercept be included in the ensemble model?
#'   Default is FALSE.
#' @param nonneg Should the ensemble weights be constrained to be nonnegative?
#'   Default is TRUE.
#' @param unit_sum Should the ensemble weights be constrained to sum to 1?
#'   Default is TRUE.
#' @param noncross Should noncrossing constraints be enforced? Default is
#'   TRUE. Note: this option only matters when there is more than group of
#'   ensemble weights, as determined by \code{tau_groups}. See details.
#' @param q0 Array of points used to define the noncrossing
#'   constraints. Must have dimension (number of points) x (number of ensemble
#'   components) x (number of quantile levels). Default is NULL, which means
#'   that we consider noncrossing constraints at the training points
#'   \code{qarr}.
#' @param lp_solver One of "glpk" or "gurobi", indicating which LP solver to
#'   use. If possible, "gurobi" should be used because it is much faster and
#'   more stable; default is "glpk"; however, because it is open-source.
#' @param time_limit This sets the maximum amount of time (in seconds) to allow
#'   Gurobi or GLPK to solve any single quantile generalized lasso problem (for
#'   a single tau and lambda value). Default is NULL, which means unlimited
#'   time.
#' @param params List of control parameters to pass to Gurobi or GLPK. Default
#'   is \code{list()} which means no additional parameters are passed. For
#'   example: with Gurobi, we can use \code{list(Threads=4)} to specify that
#'   Gurobi should use 4 threads when available. (Note that if a time limit is
#'   specified through this \code{params} list, then its value will be overriden
#'   by the last argument \code{time_limit}, assuming the latter is not NULL.)
#' @param verbose Should progress be printed out to the console? Default is
#'   FALSE.
#'
#' @return A list with the following components:
#'   \item{alpha}{Vector or matrix of ensemble weights. If \code{tau_groups} has
#'   only one unique label, then this is a vector of length = (number of
#'   ensemble components); otherwise, it is a matrix, of dimension (number of 
#'   ensemble components) x (number of quantile levels)}
#'   \item{tau}{Vector of quantile levels used}
#'   \item{weights,tau_groups,...,params}{Values of these other arguments 
#'   used in the function call}
#'
#' @details This function solves the following quantile ensemble optimization
#'   problem, over quantile levels \eqn{\tau_k, k=1,\ldots,r}:
#'   \deqn{\mathop{\mathrm{minimize}}_{\alpha_j, j=1,\ldots,p} \; \sum_{k=1}^r
#'   \sum_{i=1}^n w_i \psi_{\tau_k} \bigg(y_i - \sum_{j=1}^p \alpha_j q_{ijk}
#'   \bigg)} \deqn{\mathrm{subject \; to} \;\; \sum_{j=1}^p \alpha_j = 1, \;
#'   \alpha_j \geq 0, \; j=1,\ldots,p}
#'   for a response vector \eqn{y} and quantile array \eqn{q}, where
#'   \eqn{q_{ijk}} is an estimate of the quantile of \eqn{y_i} at the level
#'   \eqn{\tau_k}, from ensemble component member \eqn{j}. Here
#'   \eqn{\psi_\tau(v) = \max\{\tau v, (\tau-1) v\}} is the "pinball" or "tilted
#'   \eqn{\ell_1}" loss. A more advanced version allows us to estimate a
#'   separate ensemble weight \eqn{\alpha_{jk}} per component method \eqn{j},
#'   per quantile level \eqn{k}:
#'   \deqn{\mathop{\mathrm{minimize}}{\alpha_{jk}, j=1,\ldots,p, k=1,\ldots,r}
#'   \; \sum_{k=1}^r \sum_{i=1}^n w_i \psi_{\tau_k} \bigg(y_i - \sum_{j=1}^p
#'   \alpha_{jk} q_{ijk} \bigg)} \deqn{\mathrm{subject \; to} \;\;
#'   \sum_{j=1}^p \alpha_{jk} = 1, \; k=1,\ldots,r, \;
#'   \alpha_{jk} \geq 0, \; j=1,\ldots,p, \; k=1,\ldots,r}
#'   As a form of regularization, we can additionally incorporate noncrossing
#'   constraints into the above optimization, which take the form:
#'   \deqn{\alpha_{\bullet,k}^T q \leq \alpha_{\bullet,k+1}^T q, \;
#'   k=1,\ldots,r-1, \; q \in \mathcal{Q}}
#'   where the quantile levels \eqn{\tau_k, k=1,\ldots,r} are assumed to be in
#'   increasing order, and \eqn{\mathcal{Q}} is a collection of points over
#'   which to enforce the noncrossing constraints. Finally, somewhere in between
#'   these two extremes is to allow one ensemble weight per component member
#'   \eqn{j}, per quantile group \eqn{g}. This can be interpreted as a set of
#'   further constraints which enforce equality between \eqn{\alpha_{jk}} and
#'   \eqn{\alpha_{j\ell}}, for all \eqn{k,\ell} that are in the same group
#'   \eqn{g}.
#'
#' @importFrom Rglpk Rglpk_solve_LP
#' @importFrom Matrix sparseMatrix
#' @export

quantile_ensemble = function(qarr, y, tau, weights=NULL,
                             tau_groups=rep(1,length(tau)), intercept=FALSE,
                             nonneg=TRUE, unit_sum=TRUE, noncross=TRUE, q0=NULL,
                             lp_solver=c("glpk","gurobi"), time_limit=NULL,
                             params=list(), verbose=FALSE) {
  # Set up some basics
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r = dim(qarr)[3]
  if (is.null(weights)) weights = rep(1,n)
  lp_solver = match.arg(lp_solver)
  
  # Add an all 1s matrix to qarr, if we need to
  if (intercept) {
    a = array(NA, dim=c(n,p+1,r))
    for (k in 1:r) a[,,k] = cbind(rep(1,n), qarr[,,k])
    qarr = a
    p = p+1
  }

  # Standard stacking
  if (length(unique(tau_groups)) == 1) {
    obj = quantile_ensemble_stand(qarr=qarr, y=y, tau=tau, weights=weights,
                                  intercept=intercept, nonneg=nonneg,
                                  unit_sum=unit_sum, lp_solver=lp_solver,
                                  time_limit=time_limit, params=params, verbose)
  }
  
  # Flexible stacking
  else {
    # First properly set up q0, if there's noncrossing constraints 
    if (noncross) {
      # If there's no q0 passed, then just use the training points
      if (is.null(q0)) q0 = qarr
      # If there's one passed, then account for intercept if needed
      else if (intercept) {
        n0 = dim(q0)[1]; a0 = array(NA, dim=c(n0,p,r))
        for (k in 1:r) a0[,,k] = cbind(rep(1,n0), q0[,,k])
        q0 = a0
      }
    }
    obj = quantile_ensemble_flex(qarr=qarr, y=y, tau=tau, weights=weights,
                                 tau_groups=tau_groups, intercept=intercept,
                                 nonneg=nonneg, unit_sum=unit_sum,
                                 noncross=noncross, q0=q0, lp_solver=lp_solver,
                                 time_limit=time_limit, params=params, verbose)
    colnames(obj$alpha) = tau
  }

  obj = c(obj, enlist(tau, weights, tau_groups, intercept, nonneg, unit_sum,
                      noncross, q0, lp_solver, time_limit, params))
  class(obj) = "quantile_ensemble"
  return(obj)
}

# Solve standard quantile stacking using an LP solver.

quantile_ensemble_stand = function(qarr, y, tau, weights, intercept=FALSE,
                                   nonneg=TRUE, unit_sum=TRUE,
                                   lp_solver=c("glpk","gurobi"), time_limit=NULL,
                                   params=list(), verbose=FALSE) {
  # Set up some basic objects that we will need
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r = dim(qarr)[3]; N = n*r
  INN = Diagonal(N)
  model = list()

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
    if (verbose) params$LogToConsole = 1
    equal_sign = "="
  }

  # GLPK setup
  else {
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    if (verbose) params$verbose = TRUE
    equal_sign = "=="
  }

  # Vector of objective coefficients
  model$obj = c(rep(0,p), rep(weights, each=r))

  # Matrix of constraint coefficients
  model$A = Matrix(0, nrow=2*N, ncol=p+N, sparse=TRUE)
  model$sense = model$rhs = rep(NA, 2*N)

  # First part of residual constraint
  for (k in 1:r) {
    model$A[(k-1)*n + 1:n, 1:p] = tau[k] * qarr[,,k]
    model$sense[(k-1)*n + 1:n] = ">="
    model$rhs[(k-1)*n + 1:n] = tau[k] * y
  }
  model$A[1:N, p + 1:N] = INN

  # Second part of residual constraint
  for (k in 1:r) {
    model$A[(r+k-1)*n + 1:n, 1:p] = (tau[k]-1) * qarr[,,k]
    model$sense[(r+k-1)*n + 1:n] = ">="
    model$rhs[(r+k-1)*n + 1:n] = (tau[k]-1) * y
  }
  model$A[N + 1:N, p + 1:N] = INN

  # Unit sum constraints on alpha, if we're asked to
  if (unit_sum) {
    vec = rep(1,p); if (intercept) vec[1] = 0
    model$A = rbind(model$A, c(vec, rep(0,N)))
    model$sense = c(model$sense, equal_sign)
    model$rhs = c(model$rhs, 1)
  }

  # Remove nonnegativity constraint on alpha, if we're asked to
  if (!nonneg) {
    if (use_gurobi) model$lb = c(rep(-Inf,p), rep(0,N))
    else model$lb = list(lower=list(ind=1:p, val=rep(-Inf,p)))
  }

  # Remove nonnegativity constraint on intercept, if needed
  if (intercept && nonneg) {
    if (use_gurobi) model$lb = c(-Inf, rep(0,p-1+N))
    else model$lb = list(lower=list(ind=1, val=-Inf))
  }

  # Call Gurobi's LP solver, store results
  if (use_gurobi) {
    a = gurobi::gurobi(model=model, params=params)
    alpha = a$x[1:p]
    status = a$status
  }

  # Call GLPK's LP solver, store results
  else {
    a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                       rhs=model$rhs, bounds=model$lb, control=params)
    alpha = a$solution[1:p]
    status = a$status
  }

  return(enlist(alpha, status))
}

# Solve flexible quantile stacking using an LP solver.

quantile_ensemble_flex = function(qarr, y, tau, weights, tau_groups,
                                  intercept=FALSE, nonneg=TRUE, unit_sum=TRUE,
                                  noncross=TRUE, q0=NULL,
                                  lp_solver=c("glpk","gurobi"), time_limit=NULL,
                                  params=list(), verbose=FALSE) {

  # Check inputs

  if (anyDuplicated(tau) != 0L || is.unsorted(tau) || any(tau < 0 | tau > 1)) {
    stop ("Entries of `tau` must be distinct, sorted in increasing order, >=0, and <=1")
  }

  if (intercept && noncross) {
    stop ("using multiple tau groups with intercept=TRUE and noncross=TRUE is currently unsupported")
    # (Matrix formation needs to be debugged or verified for this case.)
  }

  # Set up some basic objects that we will need
  n = dim(qarr)[1]
  p = dim(qarr)[2]
  r_using_taus = dim(qarr)[3]
  labels = unique(tau_groups); num_labels = length(labels)
  tau_group_runs = rle(tau_groups)
  if (verbose && noncross && anyDuplicated(tau_group_runs[["values"]]) != 0L) {
    warning('`noncross=TRUE` but `tau_groups` are interleaved; you may want to unify any interleaved tau groups to guarantee they will not cross for any test data (e.g., `tau_groups=c("a","b","a")` -> `tau_groups=c("ab","ab","ab")')
  }
  r_using_runs = length(tau_group_runs[["lengths"]])
  tau_group_run_inds = rep(seq_len(r_using_runs), tau_group_runs[["lengths"]])
  N_using_taus = n*r_using_taus # number of residual vars --- n per quantile level
  P_using_runs = p*r_using_runs # number of coefficient vars --- p per tau group run
  model = list()

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
    if (verbose) params$LogToConsole = 1
    equal_sign = "="
  }

  # GLPK setup
  else {
    if (!is.null(time_limit)) params$tm_limit = time_limit * 1000
    if (verbose) params$verbose = TRUE
    equal_sign = "=="
  }

  # Vector of objective coefficients
  model$obj = c(rep(0,P_using_runs), rep(weights, each=r_using_taus))

  # Matrix of constraint coefficients
  #
  # `A_nrow` will serve a dual purpose of (a) keeping track of the current number
  # of rows in A, updated after each row group is added, and (b) giving the
  # offset that needs to be added to row indices of another matrix being rbound
  # onto A as part of the rbinding process. Conceptually, A_nrow starts at 0L;
  # an equivalent approach below only defines it after the first row group.
  #
  # `A_ncol` remains unchanged throughout
  A_ncol = P_using_runs + N_using_taus
  # The A matrix will be constructed from a list of "parts", rather than row
  # groups. Most row groups will consist of one part each, but some (maybe just
  # the residual constraint row groups) will consist of multiple parts. Parts
  # must use the appropriate `i` values for the final A matrix (rather than
  # relative values within their row group).
  A_i_parts = list()
  A_j_parts = list()
  A_x_parts = list()
  A_part_ind = 0L
  model$sense = model$rhs = rep(NA, 2*N_using_taus)

  # First part of residual constraint
  for (k in 1:r_using_taus) {
    A_part_ind <- A_part_ind + 1L
    A_i_parts[[A_part_ind]] = rep((k-1)*n + 1:n, p)
    A_j_parts[[A_part_ind]] = rep((tau_group_run_inds[[k]]-1)*p + 1:p, each=n)
    A_x_parts[[A_part_ind]] = as.vector(tau[[k]] * qarr[,,k])
    model$sense[(k-1)*n + 1:n] = ">="
    model$rhs[(k-1)*n + 1:n] = tau[k] * y
  }
  A_part_ind <- A_part_ind + 1L
  A_i_parts[[A_part_ind]] = 1:N_using_taus
  A_j_parts[[A_part_ind]] = P_using_runs + 1:N_using_taus
  A_x_parts[[A_part_ind]] = rep(1, N_using_taus)

  # Second part of residual constraint
  for (k in 1:r_using_taus) {
    A_part_ind <- A_part_ind + 1L
    A_i_parts[[A_part_ind]] = rep((r_using_taus+k-1)*n + 1:n, p)
    A_j_parts[[A_part_ind]] = rep((tau_group_run_inds[[k]]-1)*p + 1:p, each=n)
    A_x_parts[[A_part_ind]] = as.vector((tau[[k]]-1) * qarr[,,k])
    model$sense[(r_using_taus+k-1)*n + 1:n] = ">="
    model$rhs[(r_using_taus+k-1)*n + 1:n] = (tau[k]-1) * y
  }
  A_part_ind <- A_part_ind + 1L
  A_i_parts[[A_part_ind]] = N_using_taus + 1:N_using_taus
  A_j_parts[[A_part_ind]] = P_using_runs + 1:N_using_taus
  A_x_parts[[A_part_ind]] = rep(1, N_using_taus)
  A_nrow = 2L*N_using_taus

  # Group equality constraints, if needed
  if (num_labels < r_using_taus) {
    for (label in labels) {
      absolute_run_inds_for_label = which(tau_group_runs[["values"]] == label)
      for (left_relative_run_ind in seq_len(length(absolute_run_inds_for_label)-1)) {
        right_relative_run_ind = left_relative_run_ind + 1L
        left_absolute_run_ind = absolute_run_inds_for_label[[left_relative_run_ind]]
        right_absolute_run_ind = absolute_run_inds_for_label[[right_relative_run_ind]]
        A_part_ind <- A_part_ind + 1L
        A_i_parts[[A_part_ind]] = A_nrow + c(1:p, 1:p)
        A_j_parts[[A_part_ind]] = c((left_absolute_run_ind-1)*p + 1:p, (right_absolute_run_ind-1)*p + 1:p)
        A_x_parts[[A_part_ind]] = c(rep(-1,p), rep(1,p))
        A_nrow <- A_nrow + p
      }
    }
    model$sense = c(model$sense, rep(equal_sign, p*(r_using_runs-num_labels)))
    model$rhs = c(model$rhs, rep(0, p*(r_using_runs-num_labels)))
  }

  # Unit sum constraints on alpha, if we're asked to
  if (unit_sum) {
    A_part_ind <- A_part_ind + 1L
    if (intercept) {
      A_i_parts[[A_part_ind]] = A_nrow + rep(seq_len(r_using_runs), each=p-1L)
      A_j_parts[[A_part_ind]] = rep(seq.int(0L, P_using_runs-p, p), each=p-1L) + seq.int(2L,p) # (recycling)
      A_x_parts[[A_part_ind]] = rep(1, (p-1L)*r_using_runs)
    } else {
      A_i_parts[[A_part_ind]] = A_nrow + rep(seq_len(r_using_runs), each=p)
      A_j_parts[[A_part_ind]] = seq_len(P_using_runs)
      A_x_parts[[A_part_ind]] = rep(1, P_using_runs)
    }
    A_nrow <- A_nrow + r_using_runs
    model$sense = c(model$sense, rep(equal_sign, r_using_runs))
    model$rhs = c(model$rhs, rep(1, r_using_runs))
  }

  # Remove nonnegativity constraint on alpha, if we're asked to
  if (!nonneg) {
    if (use_gurobi) model$lb = c(rep(-Inf,P_using_runs), rep(0,N_using_taus))
    else model$lb = list(lower=list(ind=1:P_using_runs, val=rep(-Inf,P_using_runs)))
  }

  # Remove nonnegativity constraint on intercepts, if needed
  if (intercept && nonneg) {
    if (use_gurobi) model$lb = c(rep(c(-Inf, rep(0,p-1)), r_using_runs), rep(0,N_using_taus))
    else model$lb = list(lower=list(ind=(0:(r_using_runs-1))*p + 1, val=rep(-Inf,r_using_runs)))
  }

  # Noncrossing constraints, if we're asked to
  if (noncross) {
    n0 = dim(q0)[1]
    if (nonneg && !any(apply(q0, 1:2, is.unsorted))) {
      # In this case, we can form noncrossing constraints by run rather than by
      # tau. For two taus within the same run or group, we already have that
      # group_coefs dot q0[i,,k1] <= group_coefs dot q0[i,,k2] for k1 <= k2
      # using nonnegativity and sortedness.
      ks_for_run_ends = cumsum(tau_group_runs[["lengths"]])
      for (left_absolute_run_ind in 1:(r_using_runs-1)) {
        right_absolute_run_ind = left_absolute_run_ind + 1L
        A_part_ind <- A_part_ind + 1L
        A_i_parts[[A_part_ind]] = A_nrow + c(rep(seq_len(n0), p),
                                             rep(seq_len(n0), p))
        A_j_parts[[A_part_ind]] = c(rep(( left_absolute_run_ind-1L)*p + 1:p, each=n0),
                                    rep((right_absolute_run_ind-1L)*p + 1:p, each=n0))
        A_x_parts[[A_part_ind]] = c(-q0[,,ks_for_run_ends[[left_absolute_run_ind]]   ],
                                     q0[,,ks_for_run_ends[[left_absolute_run_ind]]+1L])
        A_nrow <- A_nrow + n0
      }
      model$sense = c(model$sense, rep(">=", n0*(r_using_runs-1)))
      model$rhs = c(model$rhs, rep(0, n0*(r_using_runs-1)))
    } else {
      for (k in 1:(r_using_taus-1)) {
        A_part_ind <- A_part_ind + 1L
        A_i_parts[[A_part_ind]] = A_nrow + c(rep(seq_len(n0), p),
                                             rep(seq_len(n0), p))
        A_j_parts[[A_part_ind]] = c(rep((tau_group_run_inds[[k   ]]-1L)*p + 1:p, each=n0),
                                    rep((tau_group_run_inds[[k+1L]]-1L)*p + 1:p, each=n0))
        A_x_parts[[A_part_ind]] = c(-q0[,,k   ],
                                     q0[,,k+1L])
        A_nrow <- A_nrow + n0
      }
      model$sense = c(model$sense, rep(">=", n0*(r_using_taus-1)))
      model$rhs = c(model$rhs, rep(0, n0*(r_using_taus-1)))
    }
  }

  # Build model$A from parts:
  model$A = new("dgTMatrix", # `d`ouble-type entries, `g`eneral structure, `T`sparse (ijx format)
    i = as.integer(do.call(c, A_i_parts[seq_len(A_part_ind)])) - 1L,
    j = as.integer(do.call(c, A_j_parts[seq_len(A_part_ind)])) - 1L,
    x = as.numeric(do.call(c, A_x_parts[seq_len(A_part_ind)])),
    Dim = as.integer(c(A_nrow, A_ncol))
  )

  # Call Gurobi's LP solver, store results
  if (use_gurobi) {
    a = gurobi::gurobi(model=model, params=params)
    alpha_for_runs = matrix(a$x[1:P_using_runs],p,r_using_runs)
    status = a$status
  }

  # Call GLPK's LP solver, store results
  else {
    a = Rglpk_solve_LP(obj=model$obj, mat=model$A, dir=model$sense,
                       rhs=model$rhs, bounds=model$lb, control=params)
    alpha_for_runs = matrix(a$solution[1:P_using_runs],p,r_using_runs)
    status = a$status
  }

  # alpha_for_runs is p x r_using_runs, while the output alpha is expected to be
  # p x r_using_taus; duplicate appropriately:
  alpha = alpha_for_runs[, rep.int(seq_len(r_using_runs), tau_group_runs[["lengths"]]), drop=FALSE]

  return(enlist(alpha, status))
}


##############################

#' Coef function for quantile_ensemble object
#'
#' Retrieve ensemble coefficients for estimating the conditional quantiles at
#' given tau values.
#'
#' @param object The \code{quantile_ensemble} object.
#' @param ... Additional arguments (not used).
#
#' @method coef quantile_ensemble
#' @export

coef.quantile_ensemble = function(object, ...) {
  return(object$alpha)
}

##############################

#' Predict function for quantile_ensemble object
#'
#' Predict the conditional quantiles at a new set of ensemble realizations,
#' using
#' the ensemble coefficients at given tau values.
#'
#' @param object The \code{quantile_ensemble} object.
#' @param newq Array of new predicted quantiles, of dimension (number of new
#'   prediction points) x (number or ensemble components) x (number of quantile
#'   levels).
#' @param sort Should the returned quantile estimates be sorted? Default is
#'   TRUE.
#' @param iso Should the returned quantile estimates be passed through isotonic
#'   regression? Default is FALSE; if TRUE, takes priority over \code{sort}.
#' @param nonneg Should the returned quantile estimates be truncated at 0?
#'   Natural for count data. Default is FALSE.
#' @param round Should the returned quantile estimates be rounded? Natural for
#'   count data. Default is FALSE.
#' @param ... Additional arguments (not used).
#' 
#' @method predict quantile_ensemble
#' @export

predict.quantile_ensemble = function(object, newq, s=NULL, sort=TRUE, iso=FALSE,  
                                     nonneg=FALSE, round=FALSE, ...) {
  # Set up some basics
  n0 = dim(newq)[1]
  p = dim(newq)[2]
  r = dim(newq)[3]

  # Add an all 1s matrix to newq, if we need to
  if (object$intercept) {
    a = array(NA, dim=c(n0,p+1,r))
    for (k in 1:r) a[,,k] = cbind(rep(1,n0), newq[,,k])
    newq = a
    p = p+1
  }

  # Make predictions
  z = matrix(NA, nrow=n0, ncol=r)
  alpha = matrix(object$alpha, nrow=p, ncol=r)
  for (i in 1:n0) {
    qmat = matrix(newq[i,,], nrow=dim(newq)[2], ncol=dim(newq)[3])
    pmat = t(qmat) %*% alpha
    if (r == 1) z[i,] = pmat
    else z[i,] = diag(pmat)
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

#' Combine matrices into an array
#'
#' Combine (say) p matrices, each of dimension n x r, into an n x p x r array.
#'
#' @param mat First matrix to combine into an array. Alternatively, a list of
#'   matrices to combine into an array.
#' @param ... Additional matrices to combine into an array. These additional
#'   arguments will be ignored if \code{mat} is a list.
#'
#' @export

combine_into_array = function(mat, ...) {
  if (is.list(mat)) mat_list = mat
  else mat_list = c(list(mat), list(...))

  n = nrow(mat_list[[1]])
  r = ncol(mat_list[[1]])
  p = length(mat_list)
  a = array(NA, dim=c(n,p,r))
  for (j in 1:p) a[,j,] = mat_list[[j]]
  return(a)
}
