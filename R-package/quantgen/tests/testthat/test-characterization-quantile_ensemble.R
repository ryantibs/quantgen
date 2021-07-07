
## This file contains characterization / Golden Master tests for quantile
## ensemble methods to detect behavior changes. (Try to use snapshot/golden
## tests when possible, but potential differences in Rglpk and/or Gurobi
## behavior due to system differences or nondeterminism may cause issues with
## this approach.)

testthat::test_that("random number generation is reproducible using withr functions", {
  testthat::expect_snapshot(withr::with_rng_version("3.6.0", withr::with_seed(8899587L, rnorm(5L))))
  testthat::expect_snapshot(withr::with_rng_version("3.6.0", withr::with_seed(8899587L, rexp(5L))))
})

#' Helper function for expect_identical_custom_rds_snapshot below.  Don't call directly.
expect_save_custom_rds_snapshot = function(object, file, nonce.to.overwrite.custom.snapshot.with.new.value) {
    if (file.exists(file)) {
        old.object.and.nonce = readRDS(file)
        if (identical(old.object.and.nonce[["nonce"]], nonce.to.overwrite.custom.snapshot.with.new.value)) {
            testthat::fail('nonce.to.overwrite.custom.snapshot.with.new.value matched the saved old nonce value.  This may indicate that someone overwrote the custom snapshot already, but forgot to remove the nonce.to.overwrite.custom.snapshot.with.new.value argument to expect_identical_custom_rds_snapshot.  (With a TRUE/FALSE switch instead of a nonce, this would effectively disable this test, always overwriting the rds with a new value.)  The snapshot has not been updated.  TO FIX: (a) if you do not want to overwrite the snapshot, do not pass a nonce.to.overwrite.custom.snapshot.with.new.value argument to expect_identical_custom_rds_snapshot.  (b) If you want to overwrite the custom snapshot, change the nonce argument to a different integer (do not use RNG calls within your test code to come up with the value).')
            return()
        }
    }
    dirpath = dirname(file)
    if (!dir.exists(dirpath)) {
        dir.create(dirpath, recursive = TRUE)
    }
    object.and.nonce = list(object = object, nonce = nonce.to.overwrite.custom.snapshot.with.new.value)
    saveRDS(object.and.nonce, file)
    testthat::succeed()
}

#' A custom version of \code{\link[testthat:expect_snapshot]{testthat::expect_snapshot}} that tests that an object is identical to a object in an RDS file in the \code{tests/testthat/_custom_snaps} directory
#'
#' This seems like it may function roughly the same as \code{\link[testthat:expect_known_value]{testthat::expect_known_value}},
#' the precursor to \code{\link[testthat:expect_snapshot]{testthat::expect_snapshot}}.
#'
#' To overwrite the existing snapshot value, instead of using snapshot_review, either: (a) remove the snapshot file (but not within the testing code) and run the test, or (b) provide a new value to nonce.to.overwrite.custom.snapshot.with.new.value (but not by using RNG within the testing code), run the test, then remove this nonce argument from the testing code.
#'
#' @examples
#' \dontrun{
#'   expect_identical_custom_rds_snapshot("val A", "object1") # warns, passes, creates initial snapshot
#'   expect_identical_custom_rds_snapshot("val A", "object1") # succeeds
#'   expect_identical_custom_rds_snapshot("val B", "object1") # fails
#'   expect_identical_custom_rds_snapshot("val B", "object1", 1524L) # passes, overwrites snapshot
#'   expect_identical_custom_rds_snapshot("val C", "object1", 1524L) # fails
#'   expect_identical_custom_rds_snapshot("val C", "object1", 65274L) # overwrites snapshot, succeeds
#' }
expect_identical_custom_rds_snapshot = function(object, base.name.sans.ext, nonce.to.overwrite.custom.snapshot.with.new.value=NA_integer_) {
  file = file.path("_custom_snaps", paste0(base.name.sans.ext,".RDS"))
  if (!is.na(nonce.to.overwrite.custom.snapshot.with.new.value)) {
    expect_save_custom_rds_snapshot(object, file, nonce.to.overwrite.custom.snapshot.with.new.value)
  } else if (!file.exists(file)) {
    warning('Custom snapshot rds not found; initializing custom snapshot rds using object that was to be tested.')
    expect_save_custom_rds_snapshot(object, file, nonce.to.overwrite.custom.snapshot.with.new.value)
  } else {
    testthat::expect_identical(object, readRDS(!!file)[["object"]])
  }
}

#' A custom version of \code{\link[testthat:expect_snapshot]{testthat::expect_snapshot}} that tests that an object is "equal" to a object in an RDS file in the \code{tests/testthat/_custom_snaps} directory
#'
#' Like \code{\link{expect_identical_custom_rds_snapshot}}, but using "equal" rather than "identical".
#'
#' To overwrite the existing snapshot value, instead of using snapshot_review, either: (a) remove the snapshot file (but not within the testing code) and run the test, or (b) provide a new value to nonce.to.overwrite.custom.snapshot.with.new.value (but not by using RNG within the testing code), run the test, then remove this nonce argument from the testing code.
expect_equal_custom_rds_snapshot = function(object, base.name.sans.ext, nonce.to.overwrite.custom.snapshot.with.new.value=NA_integer_) {
  file = file.path("_custom_snaps", paste0(base.name.sans.ext,".RDS"))
  if (!is.na(nonce.to.overwrite.custom.snapshot.with.new.value)) {
    expect_save_custom_rds_snapshot(object, file, nonce.to.overwrite.custom.snapshot.with.new.value)
  } else if (!file.exists(file)) {
    warning('Custom snapshot rds not found; initializing custom snapshot rds using object that was to be tested.')
    expect_save_custom_rds_snapshot(object, file, nonce.to.overwrite.custom.snapshot.with.new.value)
  } else {
    testthat::expect_equal(object, readRDS(!!file)[["object"]])
  }
}

test_that("quantile_ensemble's A matrix (for Rgplk) encodes identical entries to previous version and that the output is equal", {
  withr::with_rng_version("3.6.0", withr::with_seed(222017L, {
    ## Set up fake problem:
    n = 4L # number of training instances
    p = 2L # number of ensemble components
    tau = c(0.01, 0.1, 0.5, 0.9, 0.99)
    tau_groups_choices = list(
      c(1L, 2L, 2L, 2L, 3L),
      rep(1L, 5L),
      c("a", "b", "c","b","a"),
      1:5
    )
    r = length(tau) # number of quantile levels
    qarr = aperm(apply(array(rexp(n * p * r), c(n, p, r)), 1:2, sort), c(2:3, 1L))
    y = rowSums(rnorm(p * r, , 5) * qarr + rnorm(n * p * r) * qarr) + rnorm(n)
    weights = rexp(n)
    q0a = NULL
    n0b = 2L
    q0b = aperm(apply(array(rexp(n0b * p * r), c(n, p, r)), 1:2, sort), c(2:3, 1L))
    q0c = q0a[1, , , drop = FALSE]
    q0d = q0a[, , c(2:1,3:5)] # out of order! might want to forbid, but for now, need to test special case handling
    ## Set up test double and inject into tested code:
    Rglpk_solve_LP_arg_recorder <- mockery::mock()
    Rglpk_solve_LP_test_double <- function(obj, mat, ...) {
      ## Don't test too much about arg ordering and naming. Ensure `mat` is named
      ## in order to manipulate below.
      Rglpk_solve_LP_arg_recorder(obj, mat = mat, ...)
      list(solution = rep(42, length(obj)))
    }
    mockery::stub(quantile_ensemble, "Rglpk_solve_LP", Rglpk_solve_LP_test_double, depth = 2L)
    ## Make calls with various configurations:
    q0.choices = list("a"=q0a, "b"=q0b, "c"=q0c, "d"=q0d)
    partial.configs = list(
      list(intercept=FALSE, nonneg=FALSE, unit_sum=FALSE, noncross=FALSE, which.q0="a"),
      list(intercept= TRUE, nonneg=FALSE, unit_sum=FALSE, noncross=FALSE, which.q0="a"),
      list(intercept= TRUE, nonneg= TRUE, unit_sum=FALSE, noncross=FALSE, which.q0="a"),
      list(intercept= TRUE, nonneg= TRUE, unit_sum= TRUE, noncross=FALSE, which.q0="a"),
      list(intercept= TRUE, nonneg= TRUE, unit_sum= TRUE, noncross= TRUE, which.q0="a"),
      ##
      list(intercept=FALSE, nonneg=FALSE, unit_sum=FALSE, noncross= TRUE, which.q0="a"),
      list(intercept= TRUE, nonneg=FALSE, unit_sum=FALSE, noncross= TRUE, which.q0="a"),
      list(intercept=FALSE, nonneg= TRUE, unit_sum=FALSE, noncross= TRUE, which.q0="a"),
      list(intercept=FALSE, nonneg=FALSE, unit_sum= TRUE, noncross= TRUE, which.q0="a"),
      ##
      list(intercept=FALSE, nonneg= TRUE, unit_sum= TRUE, noncross= TRUE, which.q0="b"),
      list(intercept=FALSE, nonneg= TRUE, unit_sum= TRUE, noncross= TRUE, which.q0="c"),
      list(intercept=FALSE, nonneg= TRUE, unit_sum= TRUE, noncross= TRUE, which.q0="d")
    )
    ## Remove some variables above with convenient names that might hide issues
    ## with renaming variables inside the quantile ensemble functions:
    rm(n,p,r)
    ## Label and perform the test calls:
    trial.config.names = character(0L)
    trial.config.outputs = list()
    trial.config.i = 0L
    for (tau_groups_i in seq_along(tau_groups_choices)) {
      tau_groups = tau_groups_choices[[tau_groups_i]]
      for (partial.config in partial.configs) {
        ## XXX currently, flex+intercept+noncross is broken; enable these tests when fixed
        if (!(length(unique(tau_groups)) != 1L && partial.config[["intercept"]] && partial.config[["noncross"]])) {
          trial.config.i <- trial.config.i + 1L
          trial.config.names[[trial.config.i]] = do.call(sprintf, c(list("trial_config_%d%d%d%d%d%s", tau_groups_i), partial.config))
          trial.config.outputs[[trial.config.i]] = quantile_ensemble(
            qarr, y, tau, weights, tau_groups,
            intercept = partial.config[["intercept"]], nonneg = partial.config[["nonneg"]],
            unit_sum = partial.config[["unit_sum"]], noncross = partial.config[["noncross"]],
            q0 = q0.choices[[partial.config[["which.q0"]]]],
            lp_solver = "glpk"
          )
        }
      }
    }
    ## Check the args, allowing different matrix formats for A matrix:
    ##   Convert A matrices to a canonical form for comparison:
    reformatted.arg.lists = lapply(mockery::mock_args(Rglpk_solve_LP_arg_recorder), function(arg.list) {
      ## Convert mat to Rsparse to get a canonical form:
      arg.list[["mat"]] <- as(arg.list[["mat"]], "RsparseMatrix")
      arg.list
    })
    ##   Make sure that the trial.config names don't contain duplicates (else testing bug):
    stopifnot(anyDuplicated(trial.config.names) == 0L)
    ##   Test that we have the same number of recorded arg lists and trial.config names:
    testthat::expect_identical(length(reformatted.arg.lists), length(trial.config.names))
    ##   Test inputs and outputs for changes:
    for (i in seq_along(reformatted.arg.lists)) {
      ## Check that inputs are identical. Provides more specificity about
      ## failures but also triggers on changes that might not impact the final
      ## outputs. May need to be commented out when making changes intended to
      ## change inputs but preserve outputs.
      ##
      ## `expect_snapshot_value` has issues with readable `style`s and an RDS
      ## seems preferable to the "serialize" style.
      expect_identical_custom_rds_snapshot(reformatted.arg.lists[[i]], paste0(trial.config.names[[i]],"_glpk_inputs"))
    }
    for (i in seq_along(reformatted.arg.lists)) {
      ## Check that outputs are "equal". Less specificity about failures but
      ## helps as backup check when making nonbreaking changes to input arg
      ## lists.
      expect_equal_custom_rds_snapshot(trial.config.outputs[[i]], paste0(trial.config.names[[i]],"_quantile_ensemble_output"))
    }
  }))
})
