#include <Rcpp.h>
#include <cmath>
#include <stdexcept>

using namespace Rcpp;

#ifdef _OPENMP
#include <omp.h>
#endif

// returns number of available threads
// omp_get_num_threads() for some reason doesn't work on all systems
// on my mac it always returns 1!!!
// check following link
// http://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
// [[Rcpp::export]]
int omp_thread_count() {
  int n = 0;
  #ifdef _OPENMP
  #pragma omp parallel reduction(+:n)
  #endif
  n += 1;
  return n;
}

inline double sign(double x) {
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

class FTRLModel {
public:
  FTRLModel(NumericVector z_inp, NumericVector n_inp, double alpha, double beta,
             double lambda, double l1_ratio, int n_features,
             double dropout, int family, double clip_grad = 1000):
  alpha(alpha), beta(beta),
  lambda1(lambda * l1_ratio), lambda2(lambda * (1.0 - l1_ratio)),
  n_features(n_features), dropout(dropout), family(family), clip_grad(clip_grad) {
    z = z_inp.begin();
    n = n_inp.begin();
  }
  double *z;
  double *n;
  double alpha;
  double beta;
  double lambda1;
  double lambda2;
  int n_features;
  double dropout;
  int family;
  double clip_grad;
  double link_function(double x) const {
    // binomial
    if(this->family == 1)
      return( 1 / (1 + exp(-x)) );
    // gaussian
    if(this->family == 2)
      return( x );
    // poisson
    if(this->family == 3)
      return( exp(x) );

    throw std::invalid_argument( "this should now happen - wrong 'family' encoding "  );
    return(-1);
  };
};

//calculates regression weights for whole model
// [[Rcpp::export]]
NumericVector get_ftrl_weights(const List &R_model) {
  FTRLModel model(R_model["z"] , R_model["n"] , R_model["alpha"], R_model["beta"],
                  R_model["lambda"], R_model["l1_ratio"], R_model["n_features"],
                  R_model["dropout"], R_model["family_code"]);
  NumericVector res(model.n_features);
  for (int j = 0; j < model.n_features; j++) {
    double z_j = model.z[j];
    if(std::abs(z_j) > model.lambda1) {
      double n_j = model.n[j];
      res[j] = (-1 / ((model.beta + sqrt(n_j)) / model.alpha  + model.lambda2)) *  (z_j - sign(z_j) * model.lambda1);
    }
  }
  return (res);
}

//calculates regression weights for inference for single sample
std::vector<double> w_ftprl(const std::vector<int> &nnz_index, const FTRLModel &model) {
  std::vector<double> retval(nnz_index.size());
  int k = 0;
  for (auto j:nnz_index) {
    double z_j = model.z[j];
    if(std::abs(z_j) > model.lambda1) {
      double n_j = model.n[j];
      retval[k] = (-1 / ((model.beta + sqrt(n_j)) / model.alpha  + model.lambda2)) *  (z_j - sign(z_j) * model.lambda1);
    }
    k++;
  }
  return(retval);
};

double predict_one(const std::vector<int> &index, const std::vector<double> &x, const FTRLModel &model) {
  std::vector<double> weights = w_ftprl(index, model);

  double prod = 0;
  for(int i = 0; i < index.size(); i++)
    prod += weights[i] * x[i];
  double res = model.link_function(prod);
  return(res);
}


// [[Rcpp::export]]
NumericVector ftrl_partial_fit(const S4 &m, const NumericVector &y, const List &R_model,
                               int do_update = 1, int nthread = 0) {

  FTRLModel model(R_model["z"] , R_model["n"] , R_model["alpha"], R_model["beta"],
                   R_model["lambda"], R_model["l1_ratio"], R_model["n_features"],
                   R_model["dropout"], R_model["family_code"]);

  // set number of threads to all available
  int nth = omp_thread_count();

  // override if user manually specified number of threads
  if(nthread > 0)
    nth = nthread;

  IntegerVector dims = m.slot("Dim");
  int N = dims[0];
  // allocate space for result
  NumericVector y_hat_R(N);
  // get pointers to not touch R API
  double *y_hat = y_hat_R.begin();

  // just to extract vectors from S4
  IntegerVector PP = m.slot("p");
  IntegerVector JJ = m.slot("j");
  NumericVector XX = m.slot("x");

  // get pointers to data - we can't touch R API in threads, so will use raw pointers
  int *P = PP.begin();
  int *J = JJ.begin();
  double *X = XX.begin();

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(nth)
  #endif
  for(int i = 0; i < N; i++) {
    size_t p1 = P[i];
    size_t p2 = P[i + 1];
    int len = p2 - p1;
    std::vector<int> example_index;
    example_index.reserve(len);
    std::vector<double> example_value;
    example_value.reserve(len);
    // int k = 0;
    for(int pp = p1; pp < p2; pp++) {
      if(do_update) {
        if(((double) rand() / (RAND_MAX)) > model.dropout) {
          example_index.push_back(J[pp]);
          example_value.push_back(X[pp] / (1.0 - model.dropout));
        }
      } else {
        example_index.push_back(J[pp]);
        example_value.push_back(X[pp]);
      }
    }
    y_hat[i] = predict_one(example_index, example_value, model);

    if(do_update) {
      double d = y_hat[i] - y[i];
      double grad;
      double n_i_g2;
      double sigma;
      std::vector<double> ww = w_ftprl(example_index, model);

      int k = 0;
      for(auto ii:example_index) {
        grad = d * example_value[k];

        if(grad > model.clip_grad)
          grad = model.clip_grad;
        if(grad < - model.clip_grad)
          grad = - model.clip_grad;

        n_i_g2 = model.n[ii] + grad * grad;
        sigma = (sqrt(n_i_g2) - sqrt(model.n[ii])) / model.alpha;
        model.z[ii] += grad - sigma * ww[k];
        model.n[ii] = n_i_g2;
        k++;
      }
    }
  }
  return y_hat_R;
}
