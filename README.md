# HMMbayes Overview
This repository contains a Bayesian implementation of a 2-state multivariate Hidden Markov Model (HMM). Code correctness is tested within the ```test``` folder, with a simulated example demonstrating correct parameter recovery with a fixed ground truth. For a real-world example, a simple mean-variance allocation scheme is implementated to demonstrate rebalancing an all-ETF portfolio during different market regimes. Further functionality to implement a $K$-state multivariate HMM will be added in the future.

Source code is located within the ```src``` folder, with all methods entirely self-contained within a single ```.cpp``` file. At the expense of a small computational overhead, parallel computing is supported within the ```HMM_parallel``` file to allow for simultaneous MCMC chains to be run concurrently on separate threads.

# Functionality
These instructions will assume installation and use of RStudio. After the ```.cpp``` is compiled, a single R function labeled ```gibbs``` or ```gibbs_parallel``` will be available within the workspace. Simply provide the requisite arguments that include the data and prior fixed hyperparameters. 

# Example
Past the test cases within the ```testing``` directory, there is an application to portfolio allocation by way of calculating the posterior predictive density of the returns for a collection of assets.Given this distribution, various optimization methodology can be employed to dynamically allocate to the various assets. 



