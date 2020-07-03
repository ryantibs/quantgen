# quantgen

Tools for generalized quantile modeling: penalized quantile regression,
penalties, noncrossing constraints, cross-validation, and ensembles.

This main purpose of this repo is to support the development of
[Delphi's](https://delphi.cmu.edu) COVID forecasts, and the development an
ensemble out of the models submitted to the
[COVID Forecast Hub](https://github.com/reichlab/covid19-forecast-hub/blob/master/README.md#the-covid-forecast-hub-team),
which is a collaborative repo with COVID forecasts from many groups different
(visualized [here](https://viz.covid19forecasthub.org)), and the data source
behind the
[CDC COVID-19 Forecasting page](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html).  

### Summary of package tools 

Briefly, the R package `quantgen` provides:

- More general tools for quantile regression compared to existing R packages 
  (to our knowledge), allowing for generalized lasso penalties, and noncrossing
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

### Example notebooks

For examples in the form of R notebooks, see:

- [simple_test.html](https://ryantibs.github.io/quantgen/simple_test.html):
  simple tests comparing the outputs and speeds of the two LP solvers to `rqPen`
  (which is based on the `quantreg` package).

- [cv_example.html](https://ryantibs.github.io/quantgen/cv_example.html):
  examples of how to use cross-validation to select the tuning parameters in
  penalized quantile regression, how to extrapolate a smaller set of quantiles
  into richer set of quantiles at prediction time, and how to use noncrossing
  constraints.

-
  [stacking_example.html](https://ryantibs.github.io/quantgen/stacking_example.html):
  examples of how to use linear stacking to build quantile model ensembles.

### Install the `quantgen` R package

To install the `quantgen` R package directly from GitHub, run the following in R:

```{r}
library(devtools)
install_github(repo="ryantibs/quantgen", subdir="R-package/quantgen")
```

### Install the `gurobi` R package

- First install the latest version of Gurobi optimizer
  [here](https://www.gurobi.com/products/gurobi-optimizer/); for academics, you
  can obtain a free license
  [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

- For R <= 3.6.1, you can then follow
  [these instructions](https://www.gurobi.com/documentation/9.0/refman/ins_the_r_package.html)
  to install the Gurobi R package.

- For R > 3.6.1, you will have to download the Gurobi R package directly from
  [this link](https://upload.gurobi.com/gurobiR/gurobi9.0.2_R.tar.gz), and build
  it yourself.  (It is apparently not yet fully released/supported by Gurobi.)
