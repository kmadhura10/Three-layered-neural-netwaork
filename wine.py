#importing libraries

import pandas as pd 
import numpy as num

#package
import matplotlib
import matplotlib.pyplot as plt
import sklearn

# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

file = pd.read_csv('E:/NYIT/Python practice/wine.csv')
file.head()
        
class Wine_NN():
            

    #forward propogatio function
    def forwad_prop(model, a0):

       #load parameters from model
        w1, b1, w2, b2, w3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
           
       #first stage of propogation
        z1= a0.dot(w1) + b1
       
       #putting 1st activation function
        a1 = num.tanh (z1)
       
       #2nd linear step
        z2 = a1.dot(w2) + b2
       
       #2nd activation
        a2 = num.tanh(z2)
       
       #3rd linear step
        z3 = a2.dot(w3) + b3
       
       #3rd activation
        a3 = softmax(z3)
   
       # store all result in cache
        cache = {'a0' : a0 , 'z1' : z1 , 'a1' : a1 , 'z2' : z2 , 'a2' : a2 , 'a3' : a3 , 'z3' : z3 }
        return cache
    
    #backward propagation fuction
    def backward_prop(model, cache, y):
    
       #load parametrs from model
        w1, b1, w2, b2, w3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
       
        #load forward propagation result
        a0, a1, a2, a3 = cache ['a0'], cache ['a1'], cache ['a2'], cache ['a3']
       
       #calculate loss derivate wrt output
        dz3 = loss_derivative(y=y, y_hat = a3)
       
       #calculate loss derivative wrt 2nd layer of weights
        dw3 = 1/m *(a2.T). dot(dz3)
       
       #calculate loss derivative wrt 2nd layer bias
        db3 = 1/m * num.sum(dz3, axis = 0)
       
       #calculte loss derivative wrt 1st layer
        dz2 = num.multiply(dz3.dot(w3.T) , anh_derivative(a2))
       
       #calculate loss derivative wrt 1st layer weights
        dw2 = 1/m * num.dot(a1.T, dz2)
       
       #calculate loss derivative wrt 1st layer bias
        db2 = 1/m * num.sum(dz2, axis = 0)
       
       #calculate loss derivation wrt start oth layer
        dz1 = num.multiply(dz2.dot(w2.T),tanh_derivative(a1))
       
        dW1 = 1/m*num.dot(a0.T,dz1)
       
        db1 = 1/m*num.sum(dz1,axis=0)
       
       #store gradients...
       
        gradients = { 'dw3' : dw3 ,'db3' : db3, 'dw2' : dw2 ,'db2' : db2, 'dw1' : dw1 ,'db1' : db1}
        return gradients

    def loss_derivative(y, y_hat):
        return y_hat - y
    
    def tanh_derivative(x):
        return (1 - num.power(x,2))
    
    def softmax (z):
        #calculate exponent term first
        exp_score = num.exp(z)
        return exp_score / num.sum(exp_score, axis=1, keepdims=True)
    
    def softmax_loss(y,y_hat):
        minval = 0.000000000001
        #get  number of samples
        m = y.shape[0]
        loss = -1/m * num.sum(y * num.log(y_hat.clip(min=minval)))
        return loss


    
    def update_parameters(model,grads,learning_rate):
        w1, b1, w2, b2, w3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        
        #update parameters
        W1 -= learning_rate * grads['dw1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dw2']
        b2 -= learning_rate * grads['db2']
        W3 -= learning_rate * grads['dw3']
        b3 -= learning_rate * grads['db3']
        
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,'w3':w3,'b3':b3}
        return model
        
    def predict(model, x):
        # Do forward pass
        c = forward_prop(model,x)
        #get y_hat
        y_hat = num.argmax(c['a3'], axis=1)
        return y_hat
        
    def calc_accuracy(model,x,y):
        # Get total number of examples
        m = y.shape[0]
        # Do a prediction with the model
        pred = predict(model,x)
        # Ensure prediction and truth vector y have the same shape
        pred = pred.reshape(y.shape)
        # Calculate the number of wrong examples
        error = num.sum(num.abs(pred-y))
        # Calculate accuracy
        return (m - error)/m * 100
    
#training phase
    
    def train(model,x,y,learning_rate, epochs=20000, print_loss=False):
        #Gradient descent. Loop over epochs
        for i in range(0, epochs):
        
        # Forward propagation
         cache = forward_prop(model,x)
        
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
         grads = backward_prop(model,cache,y)
        
        # Gradient descent parameter update
        # Assign new parameters to the model
         model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
        
        # Printing loss & accuracy every 100 iterations
        if print_loss and i % 100 == 0:
            a3 = cache['a3']
            print('Loss after iteration',i,':',softmax_loss(y,a3))
            y_hat = predict(model,x)
            y_true = y.argmax(axis=1)
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            losses.append(accuracy_score(y_pred=y_hat,y_true=y_true)*100)
            return model
        
        
    def initialize_parameters(nn_ipdim= 13,nn_hdim= 5, nn_opdim= 3):
        
        # First layer weights
        w1 = 2 *num.random.randn(nn_ipdim, nn_hdim) - 1
        
        # First layer bias
        b1 = num.zeros((1, nn_hdim))
        
        # Second layer weights
        w2 = 2 *num.random.randn(nn_hdim , nn_hdim) - 1
        
        # second layer bias
        b2 = num.zeros((1, nn_hdim))
        
        W3 = 2 * num.random.rand(nn_hdim, nn_opdim) - 1
        
        b3 = np.zeros((1,nn_opdim))
        
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,'w3':w3,'b3':b3}
        
        return model

    if __name__ == "__main__":
    
        wine = Wine_NN()
        
        
            # Get the wine labels
        y = file[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
            
            # Get inputs; we define our x and y here.
        x = file.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)
            
            # Print shapes just to check
        x.shape
        y.shape 
        x = x.values
            
            
        num.random.seed(0)
        
       #this is what returns at end 
    
        model = wine.initialize_parameters(nn_ipdim= 13,nn_hdim= 5, nn_opdim= 3)
        
        model = wine.train(model,x,y,learning_rate=0.7,epochs=4500,print_loss=True)
        
        plt.plot(model)
