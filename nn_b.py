# -*- coding: utf-8 -*-
"""
Created on Mon May  7 00:07:33 2018

@author: Ravi Theja
"""
import numpy as np
import pandas as pd
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def train_nn(X, y, num_iterations, D_s):
    #Variable initialization
    epoch=500 #Setting training iterations
    lr=0.1 #Setting learning rate
    inputlayer_neurons = X.shape[1] #number of features in data set
    hiddenlayer_neurons = 10 #number of hidden layers neurons
    output_neurons = 4 #number of neurons at output layer
    
    #weight and bias initialization
    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(size=(1,output_neurons))
    
    for i in range(epoch):
    
    #Forward Propogation
        hidden_layer_input1=np.dot(X,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        output = sigmoid(output_layer_input)
        print(output[:5])
        #Backpropagation
        E = y-output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) *lr
        bout += np.sum(d_output, axis=0,keepdims=True) *lr
        wh += X.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    return output, wh, bh, wout, bout
