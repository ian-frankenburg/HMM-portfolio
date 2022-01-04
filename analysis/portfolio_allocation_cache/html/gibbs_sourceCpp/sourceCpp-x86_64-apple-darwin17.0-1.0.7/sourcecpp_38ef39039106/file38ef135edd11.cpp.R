`.sourceCpp_1_DLLInfo` <- dyn.load('/Users/ianfrankenburg/Desktop/HMMbayes/analysis/portfolio_allocation_cache/html/gibbs_sourceCpp/sourceCpp-x86_64-apple-darwin17.0-1.0.7/sourcecpp_38ef39039106/sourceCpp_2.so')

gibbs <- Rcpp:::sourceCppFunction(function(niter, burnin, y, Sigma0, v0, mu0, S0, h, nugget, m) {}, FALSE, `.sourceCpp_1_DLLInfo`, 'sourceCpp_1_gibbs')

rm(`.sourceCpp_1_DLLInfo`)
