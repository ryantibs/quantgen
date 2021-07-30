# random number generation is reproducible using withr functions

    Code
      withr::with_rng_version("3.6.0", withr::with_seed(8899587L, rnorm(5L)))
    Output
      [1] -0.5220935  0.1550776  0.7199733  0.3463515 -1.5900384

---

    Code
      withr::with_rng_version("3.6.0", withr::with_seed(8899587L, rexp(5L)))
    Output
      [1] 0.8963576 1.1547206 0.1232399 1.2111650 0.5284585

