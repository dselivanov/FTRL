library(Matrix)
library(FTRL)
nr = 1000
nc = 500
n = 10000
n_odd = 199

i = sample(nr, n, T)
j = sample(nc, n, T)

y = sample(c(0, 1), nr, T)
x = sample(c(-1, 1), n, T)
useful_odd_features = seq(1, n_odd, 2)
x[i %in% which(y == 1) & j %in% useful_odd_features] = 1

m = sparseMatrix(i = i, j = j, x = x, dims = c(1000, 1000), giveCsparse = F)
m = as(m, "RsparseMatrix")

glmnet_classifier = cv.glmnet(x = m, y = y,
                              family = 'binomial',
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)
plot(glmnet_classifier)
