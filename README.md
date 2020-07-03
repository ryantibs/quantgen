# quantgen

Tools for generalized quantile modeling: penalized quantile regression,
penalties, noncrossing constraints, cross-validation, and ensembles. 

### Install the R package

To install the quantgen R package directly from github, run the following in R:

```{r}
library(devtools)
install_github(repo="ryantibs/quantgen", subdir="R-package/quantgen")
```

### Install Gurobi for R

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
