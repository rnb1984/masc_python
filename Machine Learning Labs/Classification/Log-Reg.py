"""
Part of Machine Learning Labs
- Logistic Regression
- - MAP, Laplace and MH
"""

import numpy as np
import pylab as plt

#################################
#       Data
#################################
trainx = np.loadtxt('trainx.csv',delimiter=',')
N = len(trainx)
traint = np.hstack((np.zeros(50),np.ones(50)))[:,None]
# Add a column of 1s, for a bias term
trainx = np.hstack((trainx,np.ones_like(traint)))


# plot orginal data
plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')

# define mean and cov of w
mu0 = np.zeros((3,1))
cov0 = np.eye(3)

################################################################
#       Newton-Raphson to find the MAP estimate for w
################################################################

w = np.zeros((3,1))
dx = np.diag(np.dot(trainx,trainx.T))[:,None]
n_its = 20
allw = np.zeros((3,20))
for i in range(20):
    allw[:,i] = w.flatten()
    P = 1.0/(1.0 + np.exp(-np.dot(trainx,w)))
    gw = -w + np.sum(trainx*np.tile(traint-P,(1,3)),axis=0)[:,None]
    temp = trainx*np.tile(P*(1-P),(1,3))
    hw = -np.eye(3) - np.dot(temp.T,trainx)
    w = w - np.dot(np.linalg.inv(hw),gw)

########################
# Plots 2 in a grid

xvals = np.arange(-3,6,0.1)
Ngrid = len(xvals)
gridpred = np.zeros((Ngrid,Ngrid))
for i in range(len(xvals)):
    for j in range(len(xvals)):
        pp = np.hstack((xvals[i],xvals[j],1))[:,None]
        gridpred[i][j] = 1/(1+np.exp(-np.dot(w.T,pp)))

plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')
A = plt.contour(xvals,xvals,gridpred.T,linewidths=3)
plt.clabel(A, inline=1, fontsize=25)

################################################
#       Laplace Approximation
################################################

# define Laplace mean and cov
lap_mean = w
lap_cov = -np.linalg.inv(hw)

# Samples
n_samps = 1000
w_samps = np.random.multivariate_normal(lap_mean.flatten(),lap_cov,n_samps)

# Plots
plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')
xlims = np.array([-3,6])
for i in range(100):
    this_w = w_samps[i,:]
    ylims = (-this_w[2] - this_w[0]*xlims)/this_w[1]
    plt.plot(xlims,ylims,'k',alpha=0.4)
    
plt.ylim((-3,6))


# Average over Contours
gridpred = np.zeros((Ngrid,Ngrid))
for i in range(len(xvals)):
    for j in range(len(xvals)):
        pp = np.hstack((xvals[i],xvals[j],1))[:,None]
        gridpred[i][j] = (1/(1+np.exp(-np.dot(w_samps,pp)))).mean()

# Plot
plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')
A = plt.contour(xvals,xvals,gridpred.T,linewidths=3)
plt.clabel(A, inline=1, fontsize=25)


######################################################
#       Metropolis-Hastings
######################################################

w = np.zeros_like(w)
n_samps = 10000
w_samps = np.zeros((n_samps,3))
old_like = -0.5*np.dot(np.dot(w.T,np.eye(3)),w)
old_like += (traint*np.log(1.0/(1+np.exp(np.dot(-trainx,w))))).sum()
old_like += ((1-traint)*np.log(1 - 1.0/(1+np.exp(np.dot(-trainx,w))))).sum()
for s in range(n_samps):
    # Propose a value
    w_new = w + np.random.normal(0,1,(3,1))
    new_like = -0.5*np.dot(np.dot(w_new.T,np.eye(3)),w_new)
    new_like += (traint*np.log(1.0/(1+np.exp(np.dot(-trainx,w_new))))).sum()
    new_like += ((1-traint)*np.log(1 - 1.0/(1+np.exp(np.dot(-trainx,w_new))))).sum()
    if np.random.rand() <= np.exp(new_like - old_like):
        w = w_new
        old_like = new_like
    w_samps[s,:] = w.T

# Plot MH
plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')
xlims = np.array([-3,6])
for i in range(100):
    this_w = w_samps[np.random.randint(n_samps),:]
    ylims = (-this_w[2] - this_w[0]*xlims)/this_w[1]
    plt.plot(xlims,ylims,'k',alpha=0.4)
    
plt.ylim((-3,6))

# Plot Average Grid
gridpred = np.zeros((Ngrid,Ngrid))
for i in range(len(xvals)):
    for j in range(len(xvals)):
        pp = np.hstack((xvals[i],xvals[j],1))[:,None]
        gridpred[i][j] = (1/(1+np.exp(-np.dot(w_samps,pp)))).mean()

plt.figure()
pos0 = np.where(traint==0)[0]
pos1 = np.where(traint==1)[0]
plt.plot(trainx[pos0,0],trainx[pos0,1],'ro')
plt.plot(trainx[pos1,0],trainx[pos1,1],'bo')
A = plt.contour(xvals,xvals,gridpred.T,linewidths=3)
plt.clabel(A, inline=1, fontsize=25)