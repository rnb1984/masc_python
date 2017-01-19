"""
Part of Machine Learning Labs
- Bayesian Regression for the Olympic Data
"""

import numpy as np
import pylab as plt

# Data
data = np.loadtxt('data100m.csv',delimiter=',')
years = data[:,0][:,None]
times = data[:,1]
years = ( years - 1896 )/4
sigma_sq = 0.05

test_years = np.arange(0,40,0.1)[:,None]

############################
# 	Create X and X_test
############################

X = np.ones_like( years )
X_test = np.ones_like( test_years )
for i in range(k):
    X = np.hstack((X, years **(i+1)))
    X_test = np.hstack(( X_test, test_years **(i+1)))

# plot sample w's
plt.figure()
plt.plot( years , times,'gx')

for i in range(10):
    prior_w_samp = np.random.multivariate_normal( prior_mean.flatten(), prior_covariance )
    plt.plot(test_years ,np.dot( X_test, prior_w_samp ),'b')

###########################
# 	Compute the posterior
############################
posterior_covariance = np.linalg.inv( np.linalg.inv( prior_covariance ) + ( 1.0/sigma_sq )*np.dot( X.T, X))
posterior_mean = ( 1.0/sigma_sq )*np.dot( np.dot(posterior_covariance, X.T ), times )


###############################
# 	Compute the predictions
################################

pred_mean = np.dot( X_test, posterior_mean )
pred_var = np.zeros( ( len( test_years ),1))
for i in range( len( test_years )):
    pv = sigma_sq + np.dot( np.dot(X_test[i,:], posterior_covariance ), X_test[i,:].T )
    pred_var[i] = pv

# Plot predictions
plt.figure()
plt.plot( years , times,'bx')
plt.errorbar(test_years, pred_mean, pred_var)