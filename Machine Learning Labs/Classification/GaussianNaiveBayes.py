"""
Part of Machine Learning Labs
- Naive-Bayes classifiers
"""

import numpy as np
import pylab as plt
%matplotlib inline
train_x = np.loadtxt('trainx.csv',delimiter=',')
test_x = np.loadtxt('testx.csv',delimiter=',')
train_t = np.hstack((np.zeros(50,),np.ones(50,)))
test_t  = np.hstack((np.zeros(200,),np.ones(200,)))

############
# Plot

plt.plot( train_x[:50,0], train_x[:50,1],'ro')
plt.plot( train_x[50:,0], train_x[50:,1],'bo')

########################################################
# Fit a Gaussian to each classes dimension
########################################################

parameters = {}
for cl in range(2):
    data_pos = np.where( train_t ==cl)[0]
    class_pars = {}
    class_pars['mean'] = train_x[ data_pos,:].mean(axis=0)
    class_pars['vars'] = train_x[ data_pos,:].var(axis=0)
    class_pars['prior'] = 1.0*len( data_pos )/len( train_x )
    parameters[cl] = class_pars

########################################################
#  Loop through the test points,
#  computing their likelihood in each class
#  and multiplying by the prior
########################################################

predictions = np.zeros((400,))
for j,tx in enumerate( test_x ):
    un_norm_prob = np.zeros((2,))
    for cl in parameters:
        un_norm_prob[cl] = parameters[cl]['prior']
        for i,m in enumerate(parameters[cl]['mean']):
            vari = parameters[cl]['vars'][i]
            un_norm_prob[cl] *= 1.0/np.sqrt(2.0*np.pi*vari)
            un_norm_prob[cl] *= np.exp((-0.5/var)*(tx[i]-m)**2)
    norm_prob = un_norm_prob/un_norm_prob.sum()
    predictions[j] = norm_prob.argmax()

accuracy = ( predictions == test_t ).mean()
print accuracy