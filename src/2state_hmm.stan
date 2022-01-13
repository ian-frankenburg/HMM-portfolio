functions{
}
data {
  int<lower=0> T;
  vector[T] y;
  vector[2] kmeans;
}
parameters {
  vector[2] mu; // mean vector of series one, including regime change
  vector<lower=0>[2] sigma; // variance of series one
  real<lower=0,upper=1> p11;
  real<lower=0,upper=1> p22;
  real<lower=0,upper=1> init_pred1;
}
transformed parameters{
  vector[2] likelihood[T];
  vector[T] marginal;
  vector[2] predict[T];
  vector[2] filter[T];
  real<lower=0,upper=1> init_pred2 = 1-init_pred1;
  real<lower=0,upper=1> p12 = 1-p11;
  real<lower=0,upper=1> p21 = 1-p22;
  for(t in 1:T){
    if(t==1){
      likelihood[t,1] = exp(normal_lpdf(y[1]| mu[1], sigma[1]));
      likelihood[t,2] = exp(normal_lpdf(y[1]| mu[2]+mu[1], sigma[2]));
      marginal[t] = p11*likelihood[t,1]*init_pred1
                      + p21*likelihood[t,1]*init_pred2
                      + p12*likelihood[t,2]*init_pred1
                      + p22*likelihood[t,2]*init_pred2;

      predict[t,1] = p11*init_pred1+p21*init_pred2;
      predict[t,2] = p12*init_pred1+p22*init_pred2;
      
      filter[t,1] = likelihood[t,1]*predict[t,1]/marginal[t];
      filter[t,2] = likelihood[t,2]*predict[t,2]/marginal[t];
    }else{
      likelihood[t,1] = exp(normal_lpdf(y[t]| mu[1], sigma[1]));
      likelihood[t,2] = exp(normal_lpdf(y[t]| mu[2]+mu[1], sigma[2]));
      marginal[t] = p11*likelihood[t,1]*filter[t-1,1]
                        + p21*likelihood[t,1]*filter[t-1,2]
                        + p12*likelihood[t,2]*filter[t-1,1]
                        + p22*likelihood[t,2]*filter[t-1,2];
    
      predict[t,1] = p11*filter[t-1,1]+p21*filter[t-1,2];
      predict[t,2] = p12*filter[t-1,1]+p22*filter[t-1,2];
      
      filter[t,1] = likelihood[t,1]*predict[t,1]/marginal[t];
      filter[t,2] = likelihood[t,2]*predict[t,2]/marginal[t];
    }
  }
}
model {
  mu ~ normal(kmeans, 1);
  sigma ~ cauchy(0, 1);
  p11 ~ beta(1, 1);
  p22 ~ beta(1, 1);
  target += sum(log(marginal));
}
generated quantities{
  vector[2] smooth[T];  
  vector[2] endSmooth;
  real ypred;
  int j;
  // Kim smoothing algorithm
  smooth[T,1] = filter[T,1];
  smooth[T,2] = filter[T,2];
  for(t in 1:T-1){
    j=T-t;
    smooth[j,1] = filter[j,1]*(p11*smooth[j+1,1]/predict[j+1,1]+p12*smooth[j+1,2]/predict[j+1,2]);
    smooth[j,2] = filter[j,2]*(p21*smooth[j+1,1]/predict[j+1,1]+p22*smooth[j+1,2]/predict[j+1,2]);
  }
  ypred=0;
  for(i in 1:2){
    ypred += smooth[T, i] * normal_rng(mu[i],sigma[i]);
  }
  endSmooth = smooth[T,];
}

