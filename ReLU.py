import numpy as np
import random


def init_params(n_x,n_h,seed=0):

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2/n_x), size=(n_h, n_x))
    W2 = rng.normal(0, np.sqrt(2/n_h), size=(1, n_h))
    B1= np.zeros((n_h,1))
    B2= np.zeros((1,1))

    return W1,W2,B1,B2

def ReLU(Z):
    
        return np.maximum(0,Z)

    
def sigmoid(Z):
    sgd= 1/(1 + np.exp(-Z))
    return sgd

def forward_pass(W1,W2,B1,B2,X):

    Z1= W1 @ X + B1
    A1= ReLU(Z1)

    Z2= W2 @ A1 + B2
    A2= sigmoid(Z2)

    return A1,A2,Z1,Z2

def compute_loss(Y,A2,eps=1e-12):
    A2= np.clip(A2,eps,1-eps)
    loss= -(Y * np.log(A2) + (1-Y) * np.log(1-A2))
    return loss

def backward(W1,W2,B1,B2,Z1,X,A1,A2,Y,m):

    dz2= A2 - Y
    dw2= 1/m * (dz2 @ A1.T) 
    db2= 1/m * np.sum(dz2, axis=1, keepdims=True )

    dz1= (W2.T @ dz2) * (Z1 > 0)
    dw1= 1/m * (dz1 @ X.T)
    db1= 1/m * np.sum(dz1, axis=1, keepdims=True)

    return dw1,db1,dw2,db2

def update(W1,W2,B1,B2,dw1,db1,dw2,db2,lr):

    W1 -= lr * dw1
    W2 -= lr * dw2
    B1 -= lr * db1
    B2 -= lr * db2


def train(X, Y, n_h, lr=0.1, epochs=5000, batch_size=4, seed=0):

    n_x= X.shape[0]
    W1,W2,B1,B2= init_params(n_x,n_h)

    for i in range(epochs):

        for start in range(0,m,batch_size):

            end= start + batch_size

            X_batch= X[:, start:end]
            Y_batch= Y[:, start:end]

            A1,A2,Z1,Z2= forward_pass(W1,W2,B1,B2,X_batch)
            loss= compute_loss(Y_batch,A2)

            m_batch= X_batch.shape[1]

            dw1,db1,dw2,db2= backward(W1,W2,B1,B2,Z1,X_batch,A1,A2,Y_batch,m_batch)
            update(W1,W2,B1,B2,dw1,db1,dw2,db2,lr)

            print(f"epoch {i}, batch {start//batch_size}: {np.mean(loss)}")


X= np.random.randn(10,4)
Y= np.array([1,0,0,1]).reshape(1,4)

m= X.shape[1]



train(X, Y, 6, lr=0.1, epochs=5000, batch_size=2, seed=0)

        

















