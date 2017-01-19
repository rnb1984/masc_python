"""
Part of Machine Learning Labs
- Classification with K-Nearest Neighbours
"""

import numpy as np
import pylab as plt

################################
#   KNN Algorithm
################################
def KNN_predict( X, t, test, K):
    dis_all = []
    for i,x in enumerate( X ):
        dis = (( x-test )**2).sum()
        dis_all.append((dis, t[i]))
    dis_all = sorted( dis_all, key=lambda di:dis_all[0] )
    vote_a = 0
    vote_b = 0
    for k in range( K ):
        if dis_all[k][1] == 0:
            vote_a += 1
        else:
            vote_b += 1
    if vote_a >= vote_b:
        return 0
    else:
        return 1

################################
#   Generate Data to test
################################
N = 100
D = 2

x_1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],50)
x_2 = np.random.multivariate_normal([2,2],[[1,0],[0,1]],50)
x = np.vstack((x_1 ,x_2))

plt.figure()
plt.plot( x_1[:,0], x_1[:,1],'ro')
plt.plot(x_2[:,0],x_2[:,1],'bo')

test_x_1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],200)
test_x_2 = np.random.multivariate_normal([2,2],[[1,0],[0,1]],200)

test_x = np.vstack(( test_x_1, test_x_2))

np.savetxt('trainx.csv',x,delimiter=',')
np.savetxt('testx.csv',test_x,delimiter=',')

train_t  = np.hstack((np.zeros((50,)),np.ones((50,))))
test_t  = np.hstack((np.zeros((200,)),np.ones((200,))))


################################
# Find best K's
################################
max_k = 50
accuracy = np.zeros((max_k,))
for k in range(max_k):
    test_predict = np.zeros((400,))
    for i,tx in enumerate( test_x ):
        test_predict[i] = KNN_predict(x, train_t ,tx,k)

    accuracy[k] = (test_predict == test_t ).mean()

#  Plot K's accuracy
plt.figure()
plt.plot(range(max_k),accuracy)