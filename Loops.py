import numpy as np
import random

def sigmoid(z):
    sgd= 1/(1 + np.exp(-z))
    return sgd

def forward_pass(W_in,W_out,B_in,B_out,x):
    n= len(W_in)
    z=[0] * n
    a=[0] * n

    for i in range(len(W_in)):
        z[i]= W_in[i] * x + B_in[i]
        a[i]= sigmoid(z[i])

    z3= sum(W_out[i] * a[i] for i in range(n)) + B_out[0]
    a3= sigmoid(z3)

    cache= {"a":a, "a_out": a3 }
    return cache

def backward_pass(W_in,W_out,cache,x,y):

    n= len(W_in)
    grads_in=[0] * n
    grads_out=[0] * n
    bgrads_in=[0] * n
    bgrads_out=[0]

    dl_dy= cache["a_out"]-y

    for i in range(n):
        grads_in[i]= dl_dy * W_out[i] * (cache["a"][i] * (1 - cache["a"][i]))  * x
        grads_out[i]= dl_dy * cache['a'][i]
        bgrads_in[i]= dl_dy * W_out[i] * (cache["a"][i] * (1 - cache["a"][i])) 
    
    bgrads_out = dl_dy

    return grads_in,grads_out,bgrads_in,bgrads_out

def step(W_in,W_out,B_in,B_out,grads_in,grads_out,bgrads_in,bgrads_out,lr):
    for i in range(len(W_in)):
        W_in[i] -= lr * grads_in[i]
        W_out[i] -= lr * grads_out[i]
        B_in[i] -= lr * bgrads_in[i]
    B_out[0] -= lr * bgrads_out

    return W_in,W_out,B_in,B_out

def loss(y,y_hat,eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1-eps)
    loss= -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))
    return loss

x= 2
lr=0.01

W_in= [0.01*random.random() for _ in range(3)] 
W_out= [0.01*random.random() for _ in range(3)]
B_in= [0.1,0.1,0.1]
B_out= [0.1]

steps=1000

for t in range(steps):
    cache= forward_pass(W_in,W_out,B_in,B_out,x)
    yhat= cache["a_out"]

    Lo= loss(1,yhat)

    grads_in,grads_out,bgrads_in,bgrads_out= backward_pass(W_in,W_out,cache,x,1)

    step(W_in,W_out,B_in,B_out,grads_in,grads_out,bgrads_in,bgrads_out,lr)

    print(f"step {t:4d} | yhat={yhat:.6f} | loss={Lo:.6f}")









    

