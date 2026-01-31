import numpy as np

target1=5
target2=9

W = {
  "w1": 0.8, "w2": 0.7, "w3": 0.5,
  "w4": 0.6, "w5": 0.77, "w6": 0.67,
  "w7": 0.88, "w8": 0.17, "w9": 0.45
}


#forward pass
def forward_pass(W, x):
    
    h1= W["w1"] * x
    h2= W["w2"] * x
    h3= W["w3"] * x

    y1= (W["w4"] * h1) + (W["w5"] * h2) + (W["w6"] * h3)
    y2= (W["w7"] * h1) + (W["w8"] * h2) + (W["w9"] * h3)

    cache = {"h1": h1, "h2": h2, "h3": h3, "y1": y1, "y2": y2}

    return cache


def backward_pass(W, cache, x, target1, target2):

    h1,h2,h3= cache["h1"], cache["h2"], cache["h3"]
    y1, y2= cache["y1"], cache["y2"]

    dl_dy1= 2 * (y1 - target1)
    dl_dy2= 2 * (y2 - target2)

    grads= {}

    # gradient for 1st layer - w1,w2 and w3
    grads["w1"]= (dl_dy1 * W["w4"] * x) + (dl_dy2 * W["w7"] * x)
    grads["w2"]= (dl_dy1 * W["w5"] * x) + (dl_dy2 * W["w8"] * x)
    grads["w3"]= (dl_dy1 * W["w6"] * x) + (dl_dy2 * W["w9"] * x)

    # gradient for output layer- w4,w5,w6,w7,w8,w9
    grads["w4"]= dl_dy1 * h1
    grads["w5"]= dl_dy1 * h2
    grads["w6"]= dl_dy1 * h3
    grads["w7"]= dl_dy2 * h1
    grads["w8"]= dl_dy2 * h2
    grads["w9"]= dl_dy2 * h3
    
    return grads



def step(W, grads, lr):
    for k in W:
        W[k] -= lr * grads[k]

x= 1
lr= 0.1

cache= forward_pass(W,1)
y1,y2= cache["y1"], cache["y2"]

loss1= (y1- target1) ** 2 
loss2= (y2 - target2) ** 2
loss= loss1 + loss2

W_old= W.copy()

grads= backward_pass(W,cache,x, target1, target2)
step(W,grads,lr)

cache_new= forward_pass(W,x)
newy1,newy2= cache_new["y1"],cache_new["y2"]

new_loss1= (newy1 - target1) ** 2
new_loss2= (newy2 - target2) ** 2
new_loss= new_loss1 + new_loss2

print("old W and loss:", W_old, loss)
print("new W and loss:", W, new_loss)


