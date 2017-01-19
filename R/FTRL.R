#' @import methods
#' @import Matrix
#' @import Rcpp
#' @importFrom R6 R6Class
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @useDynLib FTRL

init_ftrl_param = function(x, n_features) {
  init = numeric(n_features)
  if(!is.null(x)) {
    stopifnot(length(x) != n_features || is.numeric(x))
    init = x
  }
  init
}

#' @export
FTRL = R6::R6Class(
  classname = "estimator",
  public = list(
    #-----------------------------------------------------------------
    initialize = function(alpha, beta,
                          lambda1, lambda2,
                          n_features,
                          z = NULL, n = NULL, dropout = 0) {
      private$z = init_ftrl_param(z, n_features)
      private$n = init_ftrl_param(n, n_features)

      private$alpha = alpha
      private$beta = beta

      private$lambda1 = lambda1
      private$lambda2 = lambda2

      private$n_features = n_features

      private$model_ptr = create_ftrl_model(private$z,
                                            private$n,
                                            alpha = alpha,
                                            beta = beta,
                                            lambda1 = lambda1,
                                            lambda2 = lambda2,
                                            n_features = n_features,
                                            dropout = dropout)
    },
    #-----------------------------------------------------------------
    partial_fit = function(X, y, ...) {
      stopifnot(inherits(X, "sparseMatrix"))
      if(!inherits(class(X), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix to ", private$internal_matrix_format)
        X = as(X, private$internal_matrix_format)
      }
      stopifnot(ncol(X) == private$n_features)
      stopifnot(nrow(X) == length(y))
      if(any(is.na(X)))
        stop("NA's in input matrix are not allowed")
      # stopifnot(is.na(X)))
      p = ftrl_partial_fit(m = X, y = y, ptr = private$model_ptr, do_update = TRUE)
      private$n = p$n
      private$z = p$z
      invisible(p$pred)
    },
    #-----------------------------------------------------------------
    predict = function(X, ...) {
      stopifnot(inherits(X, "sparseMatrix"))
      if(!inherits(class(X), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix to ", private$internal_matrix_format)
        X = as(X, private$internal_matrix_format)
      }
      stopifnot(ncol(X) == private$n_features)
      if(any(is.na(X)))
        stop("NA's in input matrix are not allowed")
      p = ftrl_partial_fit(m = X, y = numeric(0), ptr = private$model_ptr, do_update = FALSE)
      return(p$pred);
    },
    #-----------------------------------------------------------------
    coef = function() {
      get_ftrl_weights(private$model_ptr)
    },
    #-----------------------------------------------------------------
    dump = function() {
      model_dump = list(alpha = private$alpha, beta = private$beta,
                        lambda1 = private$lambda1, lambda2 = private$lambda2,
                        z = private$z, n = private$n,
                        n_features = private$n_features)
      class(model_dump) = "ftrl_model_dump"
      model_dump
    },
    #-----------------------------------------------------------------
    load = function(x) {
      if(class(x) != "ftrl_model_dump")
        stop("input should be class of 'ftrl_model_dump' -  list of model parameters")
      self$initialize(alpha = x$alpha, beta = x$beta,
                      lambda1 = x$lambda1, lambda2 = x$lambda2,
                      n_features = x$n_features,
                      z = x$z, n = x$n)
    }
    #-----------------------------------------------------------------
  ),
  private = list(
    internal_matrix_format = "RsparseMatrix",
    z = NULL,
    n = NULL,
    alpha = NULL,
    beta = NULL,
    lambda1 = NULL,
    lambda2 = NULL,
    n_features = NULL,
    model_ptr = NULL
  )
)


# Reference R implementation
# quite optimized, only 7-15x slower
FTRL_R = function(alpha, beta, lambda1, lambda2, nfeature, z = NULL, n = NULL) {
  z = init_ftrl_param(z, n_features)
  n = init_ftrl_param(n, n_features)
  alpha = alpha
  beta = beta
  lambda1 = lambda1
  lambda2 = lambda2
  ##########################################################################################
  sigmoid = function(x) {
    1 / (1 + exp(-x))
  }
  ##########################################################################################
  w_ftprl = function(i) {
    retval = numeric(length(i))
    # index = which(abs(z[i]) > lambda1)
    index = abs(z[i]) > lambda1
    j = i[index]
    z_j = z[j]
    n_j = n[j]
    retval[index] = - (z_j - sign(z_j) * lambda1) / (lambda2 + (beta + sqrt(n_j)) / alpha)
    retval
  }
  ##########################################################################################
  predict_internal = function(j, x) {
    w = w_ftprl(j)
    # print(w)
    sigmoid(crossprod(x, w)[[1L]])
  }
  ##########################################################################################

  partial_fit = function(x, y, with_pb = interactive()) {#x_cv = NULL, y_cv = NULL, check_each_n = 1e5, j = 1:1e5 ) {

    # cv_train_n = length(j)
    # x_cv_train = x[, j]
    p = numeric(ncol(x))
    if (with_pb)
      pb = txtProgressBar(max = ncol(x), style = 3)

    for(col in seq_len(ncol(x))) {
      # if(col %% 1e4 == 0) {
      # message(paste(Sys.time(), "sample", col))
      # }
      index =
        if (x@p[[col]] == x@p[[col + 1L]]) integer(0)
      else seq.int(x@p[[col]], x@p[[col + 1L]] - 1L, by = 1L)

      i = x@i[index + 1L] + 1L
      xx = x@x[index + 1L]

      p[[col]] = predict_internal(i, xx)
      # if(col %% check_each_n == 0)
      # message(p[[col]])
      n_i = n[i]
      z_i = z[i]

      g = (p[[col]] - y[[col]]) * xx
      n_i_g2 = n_i + g * g
      s = (sqrt(n_i_g2) - sqrt(n_i)) / alpha

      z[i] <<- z_i + g - s * w_ftprl(i)
      n[i] <<- n_i_g2
      # print(z)
      # if(col %% check_each_n == 0 && !is.null(x_cv)) {
      #   if(!is.null(x_cv) && !is.null(y_cv)) {
      #     cv_score = round(glmnet::auc(y = y_cv, prob = predict(x_cv, FALSE)), 4)
      #     train_score = round(glmnet::auc(y = y[j], prob = predict(x_cv_train, FALSE)), 4)
      #     message(Sys.time(), " ", col, " - ", "cv = ", cv_score, " train = ", train_score)
      #   }
      # }

      if (with_pb)
        setTxtProgressBar(pb, col)
    }
    if (with_pb)
      close(pb)
    list(p = p, z = z, n = n)
  }
  predict = function(x, with_pb = interactive(), check = 10) {
    p = numeric(ncol(x))

    if (with_pb)
      pb = txtProgressBar(max = ncol(x), style = 3)

    for(col in seq_len(ncol(x))) {
      index =
        if (x@p[[col]] == x@p[[col + 1L]])
          integer(0)
      else
        seq.int(x@p[[col]], x@p[[col + 1L]] - 1L, by = 1L)

      i = x@i[index + 1L] + 1L
      xx = x@x[index + 1L]
      p[[col]] = predict_internal(i, xx)
      # if(col %% check == 0)
      #   message(p[[col]])
      if (with_pb)
        setTxtProgressBar(pb, col)
    }
    if (with_pb)
      close(pb)
    p
  }
  list(predict = predict, partial_fit = partial_fit)
}
