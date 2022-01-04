#define ARMA_NO_DEBUG
#include <RcppDist.h>
#include <RcppThread.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppThread)]]

using namespace arma;

mat forward(const mat& y, const mat& mu, const cube& Sigma, const mat& pr, const double init0)
{
  int T = y.n_rows;
  mat filter(T, 2, fill::zeros), predict(T, 2, fill::zeros), likelihood(T, 2, fill::zeros);
  vec marginal(T, fill::zeros);
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
  return(filter);
}

vec backward1(const mat& filter, const mat& pr)
{
  int T = filter.n_rows;
  double pr0;
  vec s(T, fill::zeros);
  pr0 = filter(T-1, 0);
  s(T-1) = (R::runif(0, 1) < pr0) ? 0 : 1;
  for(int t = (T-2); t >= 0; t--){
    pr0 = pr(0,s(t+1))*filter(t, 0) / dot(pr.col(s(t+1)), filter.row(t));
    s(t) = (R::runif(0,1)<pr0) ? 0 : 1;
  }
  return(s);
}

vec backward2(const vec& s)
{
  int T = s.n_elem;
  vec n(4, fill::zeros);
  for(int t = (T-2); t >= 0; t--){
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
  return(n);
}

mat transition(const double& n00, const double& n10, const double& n01, const double& n11)
{
  mat pr(2, 2, fill::eye);
  int u00 = 5, u01 = 1, u10 = 1, u11 = 5;
  double p = R::rbeta(u11 + n11, u10 + n10);
  double q = R::rbeta(u00 + n00, u01 + n01);
  pr(0,0) = q;
  pr(0,1) = 1 - q;
  pr(1,0) = 1 - p;
  pr(1,1) = p;
  return(pr);
}

mat colMeans(const mat& y) 
{
  mat x(y.n_cols, 1, fill::zeros);
  for(uword i=0; i<y.n_cols; i++){
    x.row(i) = mean(y.col(i));
  }
  return(x);
}

mat colDiff(const mat& y, const vec& mu) 
{
  mat x(y.n_rows, y.n_cols, fill::zeros);
  for(uword i=0; i < y.n_cols; i++){
    x.col(i) = y.col(i) - mu(i);
  }
  return(x);
}

mat location(const mat& y, const vec& s, const mat& mu0,
              const cube& Sigma0, const cube& Sigma1)
{
  uword p = y.n_cols;
  vec nn(2, fill::zeros);
  mat mu1(p,2, fill::zeros), mu_draw(p, 2, fill::zeros);
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
  
  return(mu_draw);
}

cube scale(const mat& y, const vec& s, const vec& v0, const mat& mu, const cube& S0)
{
  uword p = y.n_cols;
  cube S(p,p,2,fill::zeros);
  mat S1_0(p,p,fill::zeros);
  uvec s0 = find(s==0);
  S1_0 = S0.slice(0) + colDiff(y.rows(s0), mu.col(0)).t()*colDiff(y.rows(s0), mu.col(0));
  
  mat S1_1(p,p,fill::zeros);
  uvec s1 = find(s==1);
  S1_1 = S0.slice(1) + colDiff(y.rows(s1), mu.col(1)).t()*colDiff(y.rows(s1), mu.col(1));
  
  mat W0 = inv(rwish(v0(0)+s0.n_elem, inv(S1_0)));
  mat W1 = inv(rwish(v0(1)+s1.n_elem, inv(S1_1)));
  
  S = join_slices(W0,W1);
  return(S);
}

// [[Rcpp::export]]
Rcpp::List gibbs_parallel(const uword& niter, const uword& burnin, const uword& num_chains,
                 const mat& y, const cube& Sigma0, const vec& v0,
                 const mat& mu0, const cube& S0)
{
  uword T = y.n_rows;
  int p = y.n_cols;
  vec init0 = .5*ones(num_chains); //Rcpp::runif(num_chains, .3, .7);
  cube pr_st(2, 2, num_chains, fill::zeros);
  cube pr(2, 2, num_chains, fill::zeros);
  
  for(int c=0; c<num_chains; c++){
    pr.slice(c)(0,0) = R::rbeta(10,1);
    pr.slice(c)(0,1) = 1-pr.slice(c)(0,0);
    pr.slice(c)(1,1) = R::rbeta(10,1);
    pr.slice(c)(1,0) = 1-pr.slice(c)(1,1);
    
    // pr_st.slice(c)(0,0) = .9;
    // pr_st.slice(c)(0,1) = 1-pr_st.slice(c)(0,0);
    // pr_st.slice(c)(1,1) = .9;
    // pr_st.slice(c)(1,0) = 1-pr_st.slice(c)(1,1);
  }
  
  mat n(4, num_chains, fill::zeros), s(T, num_chains, fill::zeros);
  vec marginal(T, fill::zeros);
  
  cube mu0_save(p, niter-burnin, num_chains, fill::zeros), mu1_save(p, niter-burnin, num_chains, fill::zeros);

  cube filter(T, 2, num_chains, fill::zeros), mu(p, 2, num_chains, fill::zeros);
  
  field<cube> Sigma1(num_chains), filter_save(num_chains), 
  Sigma0_save(num_chains), Sigma1_save(num_chains), pr_save(num_chains);
  
  cube blank(p, p, 2, fill::zeros), blank2(p, p, niter-burnin, fill::zeros),
  blank3(T, 2, niter-burnin, fill::zeros), blank4(2, 2, niter-burnin, fill::zeros);
  
  for(int i=0; i<2; i++){
    blank.slice(i) = eye(p, p);
  }
  Sigma1.fill(blank);
  filter_save.fill(blank3);
  Sigma0_save.fill(blank2);
  Sigma1_save.fill(blank2);
  pr_save.fill(blank4);
  
  RcppThread::parallelFor(0, num_chains, [&] (int chain){
    for(uword i=0; i < niter; i++){
      // // step one: forward (discrete state Kalman) filter
      filter.slice(chain) = forward(y, mu.slice(chain), Sigma1(chain), pr.slice(chain), init0(chain));
      // // step two: backward sample
      s.col(chain) = backward1(filter.slice(chain), pr.slice(chain));
      n.col(chain) = backward2(s.col(chain));
      // // step three: sample transition parameters
      pr.slice(chain) = transition(n(0, chain), n(1, chain), n(2, chain), n(3, chain));
      // // step four: sample mean and (co)variance parameters
      mu.slice(chain) = location(y, s.col(chain), mu0, Sigma0, Sigma1(chain));
      // // step five: sample scale parameters
      Sigma1(chain) = scale(y, s.col(chain), v0, mu.slice(chain), S0);
      // save MCMC iterations for each chain after burnin
      if(i >= burnin){
        filter_save(chain).slice(i-burnin) = filter.slice(chain);
        pr_save(chain).slice(i-burnin) = pr.slice(chain);
        mu0_save.slice(chain).col(i-burnin) = mu.slice(chain).col(0);
        mu1_save.slice(chain).col(i-burnin) = mu.slice(chain).col(1);
        Sigma0_save(chain).slice(i-burnin) = Sigma1(chain).slice(0);
        Sigma1_save(chain).slice(i-burnin) = Sigma1(chain).slice(1);
      }
    }
  });
  Rcpp::List out(6);
  out["filter"] = filter_save;
  out["mu0_save"] = mu0_save;
  out["mu1_save"] = mu1_save;
  out["Sigma0_save"] = Sigma0_save;
  out["Sigma1_save"] = Sigma1_save;
  out["pr_save"] = pr_save;
  return(out);
}