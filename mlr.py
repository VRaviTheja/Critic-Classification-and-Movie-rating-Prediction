
import numpy as np
import discovery as ds

weights = []
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
    


def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    return theta,cost


def predict(d,weights):
    res = []
    g1 = weights[0]
    g2 = weights[1]
    g3 = weights[2]
    g4 = weights[3]
    print(d)
    for g in range(len(d)):
        sum_f = []
        sum = 0
        i = 1
        for temp in d[g]:
            sum = sum + temp*g1[0][i]
            i = i+1
        sum = sum + 1*g1[0][0]
        sum_f.append(sum)
        #print(sum)
        
        sum = 0
        i = 1
        for temp in d[g]:
            sum = sum + temp*g2[0][i]
            i = i+1
        sum = sum + 1*g2[0][0]
        sum_f.append(sum)
        #print(sum)
        
        sum = 0
        i = 1
        for temp in d[g]:
            sum = sum + temp*g3[0][i]
            i = i+1
        sum = sum + 1*g3[0][0]
        sum_f.append(sum)
        #print(sum)
        
        sum = 0
        i = 1
        for temp in d[g]:
            sum = sum + temp*g4[0][i]
            i = i+1
        sum = sum + 1*g4[0][0]
        sum_f.append(sum)
        #print(sum)
        res.append(sum_f.index(max(sum_f)) + 1)

    return res


''' ADDING '''
def multi_svm(X, y1,y2,y3,y4, X_train_o):
    global mins, maxs, rng
    X, mins, maxs, rng = ds.scaling_x(X)
    
    ones = np.ones([X.shape[0],1])
    X = np.concatenate((X_train_o,X),axis=1)
    X = np.concatenate((ones,X),axis=1)
    
    theta = np.zeros([1,len(X[0])])
    
    #set hyper parameters
    alpha = 0.01
    iters = 2000
    
    g1,cost1 = gradientDescent(X,y1,theta,iters,alpha)
    print(g1)
    g2,cost2 = gradientDescent(X,y2,theta,iters,alpha)
    print(g2)
    g3,cost3 = gradientDescent(X,y3,theta,iters,alpha)
    print(g3)
    g4,cost4 = gradientDescent(X,y4,theta,iters,alpha)
    print(g4)
    global weights
    weights.append(g1)
    weights.append(g2)
    weights.append(g3)
    weights.append(g4)
    return weights, mins, maxs, rng
    
