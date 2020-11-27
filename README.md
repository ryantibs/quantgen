# quantgen

This package provides tools for generalized quantile modeling: regularized 
quantile regression (with generalized lasso penalties and noncrossing 
constraints), cross-validation, quantile extrapolation, and quantile ensembles.  

Its original was to support the development of 
[Delphi's](https://delphi.cmu.edu) COVID forecasts, and the development an
ensemble forecaster out of individual component models submitted to the
[COVID Forecast Hub](https://github.com/reichlab/covid19-forecast-hub/). The
latter is a collaborative repo organized by the Reich lab, containing COVID 
forecasts from many groups (visualized 
[here](https://viz.covid19forecasthub.org)), 
and serves as the official data source behind the 
[CDC's reports on COVID forecasting](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html).  

### Summary of package tools 

The R package `quantgen` provides:

- Tools for quantile regression not found in existing R packages (to our 
  knowledge), allowing for generalized lasso penalties, and noncrossing 
  constraints.
  
- Tools for building quantile model ensembles via linear stacking, with weights
  chosen to minimize the weighted interval score, in a variety of setups (one
  weight per model, one weight per model per quantile, and everything in
  between).

- Tools for extrapolating a smaller set of quantiles into richer set of
  quantiles, in such a way that is nonparametric "in the middle" of the
  distribution (monotone cubic spline interpolation), and parametric "in the
  tails" (with a tail behavior of the user's choosing).

### Linear program solvers

All quantile regression and stacking problems are reformulated as linear
programs (LPs), and solved using one of two LP solvers:

1. GLPK, the default, which is open-source, and available thorugh the `Rglpk`
   package.

2. Gurobi, which is not open-source but free for academic use, and available
   through the `gurobi` package (see below).

If at all possible, Gurobi should be used because it is much faster and more
stable.  The mathematical details for how these LPs are formed are given in the
notebooks linked below. 

### Documentation and examples

For examples in the form of R notebooks, see:

- [simple_test.html](https://ryantibs.github.io/quantgen/simple_test.html):
  simple tests comparing the outputs and speeds of the two LP solvers to `rqPen`
  (which is based on the `quantreg` package).

- [cv_example.html](https://ryantibs.github.io/quantgen/cv_example.html):
  examples of how to use cross-validation to select the tuning parameters in
  penalized quantile regression, how to extrapolate a smaller set of quantiles
  into richer set of quantiles at prediction time, and how to use noncrossing
  constraints.

- [stacking_example.html](https://ryantibs.github.io/quantgen/stacking_example.html):
  examples of how to use linear stacking to build quantile model ensembles.

### Install the `quantgen` R package

To install the `quantgen` R package directly from GitHub, run the following in 
R:

```{r}
devtools::install_github(repo="ryantibs/quantgen", subdir="R-package/quantgen")
```

### Install the `gurobi` R package

- First install the latest version of Gurobi optimizer
  [here](https://www.gurobi.com/products/gurobi-optimizer/).
  
 - For academics, you can obtain a free license
  [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

- Then follow
  [these instructions](https://www.gurobi.com/documentation/9.0/refman/ins_the_r_package.html)
  to install the `gurobi` R package.
