import numpy as np

y=1

W = {
  "w1": 0.8, "w2": 0.7, "w3": 0.5,
  "w4": 0.6, "w5": 0.77, "w6": 0.67
}

B= {
    "b1": 0.1, "b2": 0.1, "b3": 0.1, "b4": 0.1
}

#activation function
def sigmoid(z):
    sgd= 1/( 1 + np.exp(-z))

    return sgd

#forward pass
def forward_pass(W, x,B):
    # for hidden layer 
    z1= W["w1"] * x + B["b1"]
    z2= W["w2"] * x + B["b2"]
    z3= W["w3"] * x + B["b3"]

    a1= sigmoid(z1)
    a2= sigmoid(z2)
    a3= sigmoid(z3)

    # for output layer
    z4= ((W["w4"] * a1 ) + (W["w5"] * a2 ) + (W["w6"] * a3)) + B["b4"]
    a4= sigmoid(z4)

    cache = {"a1": a1, "a2": a2, "a3": a3, "a4": a4}
    return cache


def backward_pass(W, cache, x):

    a1,a2,a3,a4= cache["a1"], cache["a2"], cache["a3"], cache["a4"]

    grads= {}
    bgrads= {}

    # gradient for 1st layer - w1,w2 and w3 
    # path w1-z1-a1-z4-a4-L : derivative backward for gradients. repeat for all Ws
    grads["w1"]= (a4-y) * W["w4"] * (a1*(1-a1)) * x
    grads["w2"]= (a4-y) * W["w5"] * (a2*(1-a2)) * x
    grads["w3"]= (a4-y) * W["w6"] * (a3*(1-a3)) * x

    # gradient for output layer- w4,w5,w6 
    grads["w4"]= (a4-y) * a1
    grads["w5"]= (a4-y) * a2
    grads["w6"]= (a4-y) * a3
    
    # gradients for biases b1,b2,b3
    bgrads["b1"]= (a4-y) * W["w4"] * (a1*(1-a1))
    bgrads["b2"]= (a4-y) * W["w5"] * (a2*(1-a2))
    bgrads["b3"]= (a4-y) * W["w6"] * (a3*(1-a3))

    #gradients for bias b4(output layer)
    bgrads["b4"]= (a4-y)

    return grads,bgrads



def step(W,B,grads,bgrads, lr):
    for k in W:
        W[k] -= lr * grads[k]
    for i in B:
        B[i] -= lr * bgrads[i]

x= 2
lr= 0.01

cache= forward_pass(W,x,B)
y_hat= cache["a4"]

def loss_calc(y,y_hat):
  loss= -(y*np.log(y_hat)+ (1-y)*np.log(1-y_hat))
  return loss

old_loss= loss_calc(y,y_hat)

W_old= W.copy()
B_old= B.copy()

grads,bgrads= backward_pass(W,cache,x)
step(W,B,grads,bgrads,lr)

cache_new= forward_pass(W,x,B)
new_yhat = cache_new["a4"]

new_loss= loss_calc(y,new_yhat)

print("old W,B and loss:", W_old,B_old, old_loss)
print("new W,B and loss:", W,B, new_loss)


