import numpy as np
import random

def sigmoid(Z):
    sgd= 1/(1 + np.exp(-Z))
    return sgd

def forward_pass(W1,W2,B1,B2,X):

    Z1= W1 @ X + B1
    A1= sigmoid(Z1)

    Z2= W2 @ A1 + B2
    A2= sigmoid(Z2)

    return A1,A2


def backward_pass(W2,A1,A2,X,Y,m):

    dz2= A2 - Y
    dw2= 1/m * (dz2 @ A1.T)
    db2= 1/m * (dz2)

    dz1= (W2.T @ dz2) * A1 * (1-A1)
    dw1= 1/m * (dz1 @ X.T)
    db1= 1/m * dz1

    return dw1,db1,dw2,db2


def loss(Y,A2):
    loss= -(Y * np.log(A2) + (1-Y)*np.log(1-A2))
    return loss

def steps(dw1,db1,dw2,db2,W1,W2,B1,B2,lr):
    W1 -= lr * dw1
    W2 -= lr * dw2
    B1 -= lr * db1
    B2 -= lr * db2

    return W1,W2,B1,B2


X= np.array([1,2,1,4,2]).reshape(5,1)
y= np.array([1]).reshape(1,1)
W1= np.array([0.01*random.random() for _ in range(15)]).reshape(3,5)
W2= np.array([0.01*random.random() for _ in range(3)]).reshape(1,3)
B1= np.zeros((3,1))
B2= np.zeros((1,1))
m= X.shape[1]

for _ in range(100):
    A1,A2= forward_pass(W1,W2,B1,B2,X)
    losscalc= loss(y,A2)

    dw1,db1,dw2,db2= backward_pass(W2,A1,A2,X,y,m)
    steps(dw1,db1,dw2,db2,W1,W2,B1,B2,0.01)

    print(losscalc)




    




    







    



