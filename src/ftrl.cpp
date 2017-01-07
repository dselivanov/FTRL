#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;
using namespace std;
#define N_CHECK_USER_INTERRUPT 100000

inline double sigmoid(double x) {
  return(1 / (1 + exp(-x)));
}

inline float sign(double x) {
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

class ftrl_model {
public:
  ftrl_model(NumericVector z_inp, NumericVector n_inp, double alpha, double beta, double lambda1, double lambda2, int n_features):
  alpha(alpha), beta(beta), lambda1(lambda1), lambda2(lambda2), n_features(n_features) {
    z = as< vector<double> > (z_inp);
    n = as< vector<double> > (n_inp);
  }
  vector<double> z;
  vector<double> n;
  double alpha;
  double beta;
  double lambda1;
  double lambda2;
  int n_features;
};

// [[Rcpp::export]]
SEXP create_ftrl_model(NumericVector z_inp, NumericVector n_inp, double alpha, double beta, double lambda1, double lambda2, int n_features) {
  ftrl_model* model = new ftrl_model( z_inp,  n_inp,  alpha,  beta,  lambda1,  lambda2,  n_features);
  // Rprintf("alpha = %f, beta = %f, lambda1=%f, lambda2=%f, n_features=%d\n",
  //         model->alpha, model->beta, model->lambda1, model->lambda2,  model->n_features);
  XPtr< ftrl_model> ptr(model, true);
  return ptr;
}

//calculates regression weights for inference
vector<double> w_ftprl(const vector<int> &nnz_index, const Rcpp::XPtr<ftrl_model> &model) {
  vector<double> retval(nnz_index.size());
  int k = 0;
  for (auto j:nnz_index) {
    double z_j = model->z[j];
    if(abs(z_j) > model->lambda1) {
      double n_j = model->n[j];
      retval[k] = (-1 / ((model->beta + sqrt(n_j)) / model->alpha  + model->lambda2)) *  (z_j - sign(z_j) * model->lambda1);
    }
    k++;
  }
  return(retval);
};

double predict_one(const vector<int> &index, const vector<double> &x, const Rcpp::XPtr<ftrl_model> &model) {
  // Rprintf("alpha = %f, beta = %f, lambda1=%f, lambda2=%f, n_features=%d\n",
  //         model->alpha, model->beta, model->lambda1, model->lambda2,  model->n_features);
  vector<double> weights = w_ftprl(index, model);

  double prod = 0;
  for(int i = 0; i < index.size(); i++)
    prod += weights[i] * x[i];
  double res = sigmoid(prod);
  return(res);
}


// [[Rcpp::export]]
NumericVector get_ftrl_weights(SEXP ptr) {
  Rcpp::XPtr<ftrl_model> model(ptr);
  NumericVector res(model->n_features);
  for (int j = 0; j < model->n_features; j++) {
    double z_j = model->z[j];
    if(abs(z_j) > model->lambda1) {
      double n_j = model->n[j];
      res[j] = (-1 / ((model->beta + sqrt(n_j)) / model->alpha  + model->lambda2)) *  (z_j - sign(z_j) * model->lambda1);
    }
  }
  return (res);
}

// [[Rcpp::export]]
List ftrl_partial_fit(S4 m, NumericVector y, SEXP ptr, int do_update = 1) {
  Rcpp::XPtr<ftrl_model> model(ptr);
  // Rprintf("alpha = %f, beta = %f, lambda1=%f, lambda2=%f, n_features=%d\n",
  //         model->alpha, model->beta, model->lambda1, model->lambda2,  model->n_features);
  IntegerVector P = m.slot("p");
  IntegerVector J = m.slot("j");
  NumericVector X = m.slot("x");
  IntegerVector dims = m.slot("Dim");
  int N = dims[0];
  NumericVector y_hat(N);

  for(int i = 0; i < N; i++) {

    if( i % N_CHECK_USER_INTERRUPT == 0)
      checkUserInterrupt();

    size_t p1 = P[i];
    size_t p2 = P[i + 1];
    int len = p2 - p1;
    vector<int> example_index(len);
    vector<double> example_value(len);
    int k = 0;
    for(int pp = p1; pp < p2; pp++) {
      example_index[k] = J[pp];
      example_value[k] = X[pp];
      k++;
    }
    y_hat[i] = predict_one(example_index, example_value, model);

    if(do_update) {
      double d = y_hat[i] - y[i];
      double grad;
      double n_i_g2;
      double sigma;
      vector<double> ww = w_ftprl(example_index, model);

      int k = 0;
      for(auto ii:example_index) {
        grad = d * example_value[k];
        n_i_g2 = model->n[ii] + grad * grad;
        sigma = (sqrt(n_i_g2) - sqrt(model->n[ii])) / model->alpha;
        model->z[ii] += grad - sigma * ww[k];
        model->n[ii] = n_i_g2;
        k++;
      }
    }
  }
  return List::create(_["pred"]  = y_hat,
                      _["z"] = wrap(model->z),
                      _["n"] = wrap(model->n)) ;
}
