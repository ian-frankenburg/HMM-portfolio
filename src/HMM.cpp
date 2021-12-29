#include <RcppDist.h>
#include <RcppThread.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppThread)]]

using namespace arma;

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

mat colMeans(mat y) 
{
  mat x(y.n_cols, 1, fill::zeros);
  for(uword i=0; i<y.n_cols; i++){
    x.row(i) = mean(y.col(i));
  }
  return(x);
}

mat colDiff(mat y, vec mu) 
{
  mat x(y.n_rows, y.n_cols, fill::zeros);
  for(uword i=0; i<y.n_cols; i++){
    x.col(i) = y.col(i)-mu(i);
  }
  return(x);
}

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

// [[Rcpp::export]]
Rcpp::List gibbs(const uword& niter, const uword& burnin, const mat& y, 
                 const cube& Sigma0, const vec& v0,
                 const mat& mu0, const cube& S0)
{
  uword T = y.n_rows;
  int p = y.n_cols;
  double init0 = R::runif(0, 1);
  
  mat filter(T, 2, fill::zeros), predict(T, 2, fill::zeros), likelihood(T, 2, fill::zeros), mu(p,2,fill::zeros);
  vec marginal(T, fill::zeros), s(T, fill::zeros), n(4, fill::zeros);
  
  mat mu0_save(p, niter-burnin, fill::zeros), mu1_save(p, niter-burnin, fill::zeros), pr(2, 2, fill::zeros);
  pr(0, 0) = R::rbeta(0, 1);
  pr(0, 1) = 1 - pr(0, 0);
  pr(1, 1) = R::rbeta(0, 1);
  pr(1, 0) = 1 - pr(1, 1);

  cube Sigma0_save(p, p, niter-burnin, fill::zeros), Sigma1_save(p, p, niter-burnin, fill::zeros), 
  filter_save(T, 2, niter-burnin, fill::zeros),
  pr_save(2, 2, niter-burnin, fill::zeros), S(p, p,2,fill::zeros);
  S.slice(0) = S.slice(1) = eye(p, p);
  
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
      pr_save.slice(i-burnin) = pr;
      mu0_save.col(i-burnin) = mu.col(0);
      mu1_save.col(i-burnin) = mu.col(1);
      Sigma0_save.slice(i-burnin) = S.slice(0);
      Sigma1_save.slice(i-burnin) = S.slice(1);
    }
  }
  Rcpp::List out(6);
  out["filter"] = filter_save;
  out["mu0_save"] = mu0_save;
  out["mu1_save"] = mu1_save;
  out["Sigma0_save"] = Sigma0_save;
  out["Sigma1_save"] = Sigma1_save;
  out["pr_save"] = pr_save;
  return(out);
}