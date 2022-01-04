#define ARMA_NO_DEBUG
#include <RcppDist.h>
#include <RcppThread.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppThread)]]

using namespace arma;

// forward filter portion of FFBS algorithm
// this function computes the Gaussian likelihood for each regime by treating the
// Markovian dynamical system as a latent (or augmented) state
void forward(mat& filter, mat& predict, mat& likelihood, vec& marginal,
             const mat& y, const mat& mu, const cube& Sigma, const mat& pr, const double init0)
{
  int T = y.n_rows;
  double p00 = pr(0,0);
  double p11 = pr(1,1);
  double init1 = 1 - init0;
  double p01 = 1 - p00; 
  double p10 = 1 - p11;
  
  for (int t=0; t<T; t++){
    if(t==0){
      likelihood(t,0) = dmvnorm(y.row(t), mu.col(0), Sigma.slice(0))(0);
      likelihood(t,1) = dmvnorm(y.row(t), mu.col(1), Sigma.slice(1))(0);
      
      marginal(t) = p00 * likelihood(t,0) * init0 +
        p10*likelihood(t,0) * init1 +
        p01*likelihood(t,1) * init0 +
        p11*likelihood(t,1) * init1;
      
      predict(t,0) = p00 * init0 + p10 * init1;
      predict(t,1) = p01 * init0 + p11 * init1;
      
      filter(t,0) = likelihood(t,0) * predict(t,0) / marginal(t);
      filter(t,1) = likelihood(t,1) * predict(t,1) / marginal(t);
    }else{
      likelihood(t,0) = dmvnorm(y.row(t), mu.col(0), Sigma.slice(0), 0)(0);
      likelihood(t,1) = dmvnorm(y.row(t), mu.col(1), Sigma.slice(1), 0)(0);
      
      marginal(t) = p00 * likelihood(t,0) * filter(t-1,0) +
        p10*likelihood(t,0) * filter(t-1,1) +
        p01*likelihood(t,1) * filter(t-1,0) +
        p11*likelihood(t,1) * filter(t-1,1);
      
      predict(t,0) = p00 * filter(t-1,0) + p10 * filter(t-1,1);
      predict(t,1) = p01 * filter(t-1,0) + p11 * filter(t-1,1);
      
      filter(t,0) = likelihood(t,0) * predict(t,0) / marginal(t);
      filter(t,1) = likelihood(t,1) * predict(t,1) / marginal(t);
    }
  }
}

// backward sample portion of FFBS
// a stochastic version of backward smoothing in time
void backward(const mat& filter, const mat& pr, vec& s, vec& n)
{
  int T = filter.n_rows; double pr0;
  n.zeros(4);
  pr0 = filter(T-1, 0);
  s(T-1) = (R::runif(0, 1) < pr0) ? 0 : 1;;
  (s(T-1)==0) ? n(0)++: n(3)++;
  for(int t = (T-2); t >= 0; t--){
    pr0 = pr(0,s(t+1))*filter(t, 0) / dot(pr.col(s(t+1)), filter.row(t));
    s(t) = (R::runif(0,1)<pr0) ? 0 : 1;
    if(s(t+1)==0 && s(t)==0){
      n(0)++;
    }else if(s(t+1)==1 && s(t)==0){
      n(1)++;
    }else if(s(t+1)==0 && s(t)==1){
      n(2)++;
    }else if(s(t+1)==1 && s(t)==1){
      n(3)++;
    }
  }
}

// compute the transition probabilities, conditional on a realized backwards sample
void transition(const int& n00, 
                const int& n10, const int& n01, const int& n11, mat& pr)
{
  int u00 = 5, u01 = 1, u10 = 1, u11 = 5;
  double p = R::rbeta(u11 + n11, u10 + n10);
  double q = R::rbeta(u00 + n00, u01 + n01);
  pr(0,0) = q;
  pr(0,1) = 1-q;
  pr(1,0) = 1-p;
  pr(1,1) = p;
}

// helper function for sampling regime mean estimates
mat colMeans(mat y) 
{
  mat x(y.n_cols, 1, fill::zeros);
  for(uword i=0; i<y.n_cols; i++){
    x.row(i) = mean(y.col(i));
  }
  return(x);
}

// helper function for sampling regime mean estimates
mat colDiff(mat y, vec mu) 
{
  mat x(y.n_rows, y.n_cols, fill::zeros);
  for(uword i=0; i<y.n_cols; i++){
    x.col(i) = y.col(i)-mu(i);
  }
  return(x);
}

// draw a sample for location parameter
void location(const mat& y, const vec& s, const mat& mu0, mat&mu_draw,
              const cube& Sigma0, const cube& Sigma1)
{
  
  uword p = y.n_cols;
  vec nn(2, fill::zeros);
  mat mu1(p,2, fill::zeros);
  cube Lambda(p, p, 2, fill::zeros);
  
  nn(0) =  uvec(find(s==0)).n_elem;
  Lambda.slice(0)=inv(inv(Sigma0.slice(0))+nn(0)*inv(Sigma1.slice(0)));
  mu1.col(0) = Lambda.slice(0)*(inv(Sigma0.slice(0))*mu0.col(0) +
    nn(0)*inv(Sigma1.slice(0))*colMeans(y.rows(find(s==0))));
  
  nn(1) =  uvec(find(s==1)).n_elem;
  Lambda.slice(1)=inv(inv(Sigma0.slice(1))+nn(1)*inv(Sigma1.slice(1)));
  mu1.col(1) = Lambda.slice(1)*(inv(Sigma0.slice(1))*mu0.col(1) +
    nn(1)*inv(Sigma1.slice(1))*colMeans(y.rows(find(s==1))));
  
  mu_draw.col(0) = rmvnorm(1, mu1.col(0), Lambda.slice(0)).t();
  mu_draw.col(1) = rmvnorm(1, mu1.col(1), Lambda.slice(1)).t();
}

// sample covariance parameters
void scale(const mat& y, const vec& s, const vec& v0, const mat& mu, const cube& S0, cube& S)
{
  uword p = y.n_cols;
  mat S1_0(p,p,fill::zeros);
  uvec s0 = find(s==0);
  S1_0 = S0.slice(0) + colDiff(y.rows(s0), mu.col(0)).t()*colDiff(y.rows(s0), mu.col(0));
  
  mat S1_1(p,p,fill::zeros);
  uvec s1 = find(s==1);
  S1_1 = S0.slice(1) + colDiff(y.rows(s1), mu.col(1)).t()*colDiff(y.rows(s1), mu.col(1));
  
  mat W0 = inv(rwish(v0(0)+s0.n_elem, inv(S1_0)));
  mat W1 = inv(rwish(v0(1)+s1.n_elem, inv(S1_1)));
  
  S = join_slices(W0,W1);
}

// compute posterior predictive distribution forecast
cube posterior_predictive_draws(const int niter, const int burnin, const int h, 
                               const cube& pr_save, const mat& S_save,
                               const mat& mu0_save, const mat& mu1_save, 
                               const cube& Sigma0_save, const cube& Sigma1_save){
  int sample = 0, p = mu0_save.n_rows, T = S_save.n_rows;
  vec s_(h, fill::zeros);
  cube y_pred(h, p, niter, fill::zeros);
  mat w(p, niter, fill::zeros);
  for(int i=0; i<niter; i++){
    sample = (int) Rcpp::runif(1, 0, niter-burnin)(0);
    s_(0) = (S_save.col(sample)(T-1) == 0) ?
    R::rbinom(1, pr_save.slice(sample)(0,0)): R::rbinom(1, pr_save.slice(sample)(1,1));
    if(s_(0) == 0){
      y_pred.slice(i).row(0) = rmvnorm(1, mu0_save.col(sample), Sigma0_save.slice(sample));
      w.col(i) = inv(Sigma0_save.slice(sample)) * mu0_save.col(sample) / sum(inv(Sigma0_save.slice(sample)) * mu0_save.col(sample));
    }else{
      y_pred.slice(i).row(0) = rmvnorm(1, mu1_save.col(sample), Sigma1_save.slice(sample));
      w.col(i) = inv(Sigma1_save.slice(sample)) * mu1_save.col(sample) / sum(inv(Sigma1_save.slice(sample)) * mu1_save.col(sample));
    }
    for(int j=1; j<h; j++){
      s_(j) = (s_(j-1) == 0) ?
      R::rbinom(1, pr_save.slice(sample)(0,0)): R::rbinom(1, pr_save.slice(sample)(1,1));
      if(s_(j)==0){
        y_pred.slice(i).row(j) = rmvnorm(1, mu0_save.col(sample), Sigma0_save.slice(sample));
      }else{
        y_pred.slice(i).row(j) = rmvnorm(1, mu1_save.col(sample), Sigma1_save.slice(sample));
      }
    }
  }
  return y_pred;
}

// feed in a vector of m which contains the returns to construct the stochastic frontier
// use expressions on PG 384 of 'Bayesian Inference of State Space Models' book
// to get portfolio weights for each resampled frontier
mat porftfolio_weights(const int niter, const int burnin, const int h, 
                                const cube& pr_save, const mat& S_save,
                                const mat& mu0_save, const mat& mu1_save, 
                                const cube& Sigma0_save, const cube& Sigma1_save, 
                                const double nugget, const double m){
  int sample = 0, p = mu0_save.n_rows, T = S_save.n_rows;
  vec s_(h, fill::zeros);
  mat w(p, niter, fill::zeros), Sig_inv(p, p, fill::ones);
  vec one(p, fill::ones), mu(p, fill::zeros);
  for(int i=0; i<niter; i++){
    sample = (int) Rcpp::runif(1, 0, niter-burnin)(0);
    s_(0) = (S_save.col(sample)(T-1) == 0) ?
    R::rbinom(1, pr_save.slice(sample)(0,0)): R::rbinom(1, pr_save.slice(sample)(1,1));
    if(s_(0) == 0){
      Sig_inv = inv(Sigma0_save.slice(sample)+nugget*eye(p,p));
      mu = mu0_save.col(sample);
      // w.col(i) = inv(Sigma0_save.slice(sample)+nugget*eye(p,p)) * mu0_save.col(sample) / sum(inv(Sigma0_save.slice(sample)+nugget*eye(p,p)) * mu0_save.col(sample));
      w.col(i) = Sig_inv * (m*as_scalar(one.t()*Sig_inv*one)*mu
                              - as_scalar(mu.t()*Sig_inv*one)*mu
                              + as_scalar(mu.t()*Sig_inv*mu)*one
                              - m*as_scalar(mu.t()*Sig_inv*one)*one)
                              / (dot(mu,Sig_inv*mu)*dot(one, Sig_inv*one) - pow(dot(mu, Sig_inv*one),2));
        
    }else{
      // w.col(i) = inv(Sigma1_save.slice(sample)+nugget*eye(p,p)) * mu1_save.col(sample) / sum(inv(Sigma1_save.slice(sample)+nugget*eye(p,p)) * mu1_save.col(sample));
      Sig_inv = inv(Sigma1_save.slice(sample)+nugget*eye(p,p));
      mu = mu1_save.col(sample);
      w.col(i) = Sig_inv * (m*as_scalar(one.t()*Sig_inv*one)*mu
                              -as_scalar(mu.t()*Sig_inv*one)*mu
                              + as_scalar(mu.t()*Sig_inv*mu)*one
                              - m*as_scalar(mu.t()*Sig_inv*one)*one)
                              / (dot(mu,Sig_inv*mu)*dot(one, Sig_inv*one) - pow(dot(mu, Sig_inv*one),2));
    }
  }
  return w;
}

vec frontier(const int niter, const int burnin, const int h, 
                       const cube& pr_save, const mat& S_save,
                       const mat& mu0_save, const mat& mu1_save, 
                       const cube& Sigma0_save, const cube& Sigma1_save, 
                       const double nugget, const double m){
  int sample = 0, p = mu0_save.n_rows, T = S_save.n_rows;
  vec s_(h, fill::zeros);
  mat w(p, niter, fill::zeros), Sig_inv(p, p, fill::ones);
  vec one(p, fill::ones), mu(p, fill::zeros), risk(niter, fill::zeros);
  for(int i=0; i<niter; i++){
    sample = (int) Rcpp::runif(1, 0, niter-burnin)(0);
    s_(0) = (S_save.col(sample)(T-1) == 0) ?
    R::rbinom(1, pr_save.slice(sample)(0,0)): R::rbinom(1, pr_save.slice(sample)(1,1));
    if(s_(0) == 0){
      Sig_inv = inv(Sigma0_save.slice(sample)+nugget*eye(p,p));
      mu = mu0_save.col(sample);
      // w.col(i) = inv(Sigma0_save.slice(sample)+nugget*eye(p,p)) * mu0_save.col(sample) / sum(inv(Sigma0_save.slice(sample)+nugget*eye(p,p)) * mu0_save.col(sample));
      w.col(i) = Sig_inv * (m*as_scalar(one.t()*Sig_inv*one)*mu
                              - as_scalar(mu.t()*Sig_inv*one)*mu
                              + as_scalar(mu.t()*Sig_inv*mu)*one
                              - m*as_scalar(mu.t()*Sig_inv*one)*one)
                              / (dot(mu,Sig_inv*mu)*dot(one, Sig_inv*one) - pow(dot(mu, Sig_inv*one),2));
      risk(i) = sqrt(dot(w.col(i), Sigma0_save.slice(sample)*w.col(i)));
    }else{
      // w.col(i) = inv(Sigma1_save.slice(sample)+nugget*eye(p,p)) * mu1_save.col(sample) / sum(inv(Sigma1_save.slice(sample)+nugget*eye(p,p)) * mu1_save.col(sample));
      Sig_inv = inv(Sigma1_save.slice(sample)+nugget*eye(p,p));
      mu = mu1_save.col(sample);
      w.col(i) = Sig_inv * (m*as_scalar(one.t()*Sig_inv*one)*mu
                              -as_scalar(mu.t()*Sig_inv*one)*mu
                              + as_scalar(mu.t()*Sig_inv*mu)*one
                              - m*as_scalar(mu.t()*Sig_inv*one)*one)
                              / (dot(mu,Sig_inv*mu)*dot(one, Sig_inv*one) - pow(dot(mu, Sig_inv*one),2));
      risk(i) = sqrt(dot(w.col(i), Sigma1_save.slice(sample)*w.col(i)));
    }
  }
  return risk;
}

// [[Rcpp::export]]
Rcpp::List gibbs(const uword& niter, const uword& burnin, const mat& y, 
                 const cube& Sigma0, const vec& v0,
                 const mat& mu0, const cube& S0, const int& h, const double nugget, const vec m)
{
  uword T = y.n_rows;
  int p = y.n_cols;
  double init0 = R::runif(0, 1);
  
  mat filter(T, 2, fill::zeros), predict(T, 2, fill::zeros), likelihood(T, 2, fill::zeros), mu(p,2,fill::zeros);
  vec marginal(T, fill::zeros), s(T, fill::zeros), n(4, fill::zeros);
  
  mat mu0_save(p, niter-burnin, fill::zeros), 
  mu1_save(p, niter-burnin, fill::zeros), 
  pr(2, 2, fill::zeros),
  S_save(T, niter-burnin);
  pr(0, 0) = R::rbeta(0, 1);
  pr(0, 1) = 1 - pr(0, 0);
  pr(1, 1) = R::rbeta(0, 1);
  pr(1, 0) = 1 - pr(1, 1);

  cube Sigma0_save(p, p, niter-burnin, fill::zeros), Sigma1_save(p, p, niter-burnin, fill::zeros), 
  filter_save(T, 2, niter-burnin, fill::zeros),
  pr_save(2, 2, niter-burnin, fill::zeros), S(p, p,2,fill::zeros);
  S.slice(0) = S.slice(1) = eye(p, p);
  
  cube y_pred(h, p, niter, fill::zeros);
  
  cube w_save(p, niter, m.n_elem, fill::zeros);
  
  for(uword i=0; i < niter; i++){
    // // step one: forward filter
    forward(filter, predict, likelihood, marginal, y, mu, S, pr, init0);
    // // step two: backward sample
    backward(filter, pr, s, n);
    // // step three: sample transition parameters
    transition(n(0), n(1), n(2), n(3), pr);
    // // step four: sample mean parameters
    location(y, s, mu0, mu, Sigma0, S);
    // // step five: sample scale parameters
    scale(y, s, v0, mu, S0, S);
    // // save MCMC draws after burnin period
    if(i >= burnin){
      filter_save.slice(i-burnin) = filter;
      S_save.col(i-burnin) = s;
      pr_save.slice(i-burnin) = pr;
      mu0_save.col(i-burnin) = mu.col(0);
      mu1_save.col(i-burnin) = mu.col(1);
      Sigma0_save.slice(i-burnin) = S.slice(0);
      Sigma1_save.slice(i-burnin) = S.slice(1);
    }
  }
  // draw from posterior predictive distribution
  y_pred = posterior_predictive_draws(niter, burnin, h,
                                     pr_save, S_save, mu0_save, mu1_save,
                                     Sigma0_save, Sigma1_save);
  mat risk(niter, m.n_elem, fill::zeros);
  for(int i=0; i<m.n_elem; i++){
    w_save.slice(i) = porftfolio_weights(niter, burnin, h, pr_save, S_save, mu0_save, mu1_save, Sigma0_save, Sigma1_save, nugget, m(i));
    risk.col(i) = frontier(niter, burnin, h, pr_save, S_save, mu0_save, mu1_save, Sigma0_save, Sigma1_save, nugget, m(i));
  }
  
  Rcpp::List out(8);
  out["filter"] = filter_save;
  out["mu0_save"] = mu0_save;
  out["mu1_save"] = mu1_save;
  out["Sigma0_save"] = Sigma0_save;
  out["Sigma1_save"] = Sigma1_save;
  out["pr_save"] = pr_save;
  out["S_save"] = S_save;
  out["y_pred"] = y_pred;
  out["pf_weights"] = w_save;
  out["risk"] = risk;
  return(out);
}
