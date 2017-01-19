## What is this?

R package which implements [Follow-the-Regularized-Leader](http://www.jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf) algorithm. It allows to solve very large problems with stochastic gradient descend online learning.

## Features

- Online learning - can easily learn model in online fashion. (like vowpal wabbit)
- Very fast - written in `Rcpp`
- Parallel, asyncronous. Benefit from multicore systems (if your compiler supports openmp) - asynchronous [Hogwild!](https://arxiv.org/abs/1106.5730) style updates under the hood

## Notes

- Only logistic regerssion implemented at the moment
- Core input format for matrix is CSR - `Matrix::RsparseMatrix`. Hoewer common R `Matrix::CpasrseMatrix` ( aka `dgCMatrix`) will be converted autamatocally

## Todo list

- gaussian family
- vignette
- test coverage (but package battle tested on [kaggle outbrain competition](https://www.kaggle.com/c/outbrain-click-prediction) and contribute to our 13 place)

## Quick reference

```r
library(Matrix)
library(FTRL)
i = sample(1000, 1000 * 100, TRUE)
j = sample(1000, 1000 * 100, TRUE)
y = sample(c(0, 1), 1000, TRUE)
x = sample(c(-1, 1), 1000 * 100, TRUE)
odd = seq(1, 99, 2)
x[i %in% which(y == 1) & j %in% odd] = 1
m = sparseMatrix(i = i, j = j, x = x, dims = c(1000, 1000), giveCsparse = FALSE)
X = as(m, "RsparseMatrix")
ftrl = FTRL$new(alpha = 0.01, beta = 0.1, lambda = 10, l1_ratio = 1, dropout = 0, n_features = ncol(m))
ftrl$partial_fit(X, y, nthread = 8)
w = ftrl$coef()
head(w)
sum(w != 0)
p = ftrl$predict(m)
```
