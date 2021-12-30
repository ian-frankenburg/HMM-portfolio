# HMMbayes Overview
This repository contains an implementation of a Bayesian 2-state multivariate Hidden Markov Model (HMM). Further functionality to implement a $K$-state multivariate HMM will be added in the future. Source code is located within the ```src``` folder, with all methods entirely self-contained within a single ```.cpp``` file. At the expense of a small computational overhead, parallel computing is supported within the ```HMM_parallel``` file to allow for simultaneous MCMC chains to be run concurrently on separate threads. Code correctness and test cases are provided in the ```simulation.Rmd``` file located within the ```test``` directory. A real-world example is demonstrated in the ```portfolio.Rmd``` file, modeling returns of an all-ETF portfolio.

# Functionality
These instructions will assume installation and use of RStudio. I futher assume the installation and use of the $\texttt{C++}$ header libraries ```RcppArmadillo```, ```RcppDist```, and ```RcppThread```, if parallel processeing is desired. After the ```.cpp``` is compiled, a single R function labeled ```gibbs``` or ```gibbs_parallel``` will be available within the workspace. Simply provide the requisite arguments that include the data and prior hyperparameters. 

# Example
Consider a latent Markovian dynamical system $S_t\in\{0,1\}$ for $t\in\{0,\ldots,T\}$ that determine the multivariate Gaussian sampling distribution at time $t$. To visualize, see the following simulated latent stochastic process:

<p align="center">
<img src="img/states.png" width="50%">
</p>

Based on these latent "regimes", the sampling model for a multivariate outcome $y_t\in\mathbb R^{p}$ becomes $y_t\sim \mathcal N_p(\boldsymbol \mu_{S_t}, \boldsymbol \Sigma_{S_t})$. Let's assume $p=3$ for simplicity and generate an observation process based upon these latent states:

<p align="center">
<img src="img/outcome.png" width="50%">
</p>

To fit a single-threaded HMM to this example series $y_{1:1000}$, call the function 
```
gibbs(niter = 5000, burnin = 2500, y = y, mu0 = mu00, Sigma0 = Sigma00, v0=v00, S0 = S00, h=10)
```
for specified function arguments. To briefly explain the necessary priors

- ```mu0``` is the prior mean for each regime, i.e., $\mu_{S_t} \sim \mathcal N(\mu_{0}, \boldsymbol \Sigma_{0})$
- ```v0``` and ```S0``` are the inverse Wishart hyperparameters for the prior on $\boldsymbol \Sigma_{S_t}$, i.e., $\Sigma_{S_t}\sim \texttt{Inverse-Wishart}(v_0, S_0)$
- ```h``` is the desired forcast length for the posterior predictive distribution

Further methodolical details are provided in the ```.Rmd``` files located in the ```test``` directory.

