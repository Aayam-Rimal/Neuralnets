import numpy as np

target1=5
target2=9

W = {
  "w1": 0.8, "w2": 0.7, "w3": 0.5,
  "w4": 0.6, "w5": 0.77, "w6": 0.67,
  "w7": 0.88, "w8": 0.17, "w9": 0.45
}

B= {
    "b1": 0.1, "b2": 0.1, "b3": 0.1, "b4": 0.1, "b5": 0.1
}


#forward pass
def forward_pass(W, x,B):
    
    h1= W["w1"] * x + B["b1"]
    h2= W["w2"] * x + B["b2"]
    h3= W["w3"] * x + B["b3"]

    y1= ((W["w4"] * h1 ) + (W["w5"] * h2 ) + (W["w6"] * h3)) + B["b4"]
    y2= ((W["w7"] * h1 ) + (W["w8"] * h2 ) + (W["w9"] * h3)) + B["b5"]

    cache = {"h1": h1, "h2": h2, "h3": h3, "y1": y1, "y2": y2}

    return cache


def backward_pass(W, cache, x, target1, target2):

    h1,h2,h3= cache["h1"], cache["h2"], cache["h3"]
    y1, y2= cache["y1"], cache["y2"]

    dl_dy1= 2 * (y1 - target1)
    dl_dy2= 2 * (y2 - target2)

    grads= {}
    bgrads= {}

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
    
    # gradients for biases b1,b2,b3
    bgrads["b1"]= (dl_dy1 * W["w4"]) + (dl_dy2 * W["w7"])
    bgrads["b2"]= (dl_dy1 * W["w5"]) + (dl_dy2 * W["w8"])
    bgrads["b3"]= (dl_dy1 * W["w6"]) + (dl_dy2 * W["w9"])

    #gradients for biases b4 and b5
    bgrads["b4"]= dl_dy1
    bgrads["b5"]= dl_dy2

    return grads,bgrads



def step(W,B,grads,bgrads, lr):
    for k in W:
        W[k] -= lr * grads[k]
    for i in B:
        B[i] -= lr * bgrads[i]

x= 1
lr= 0.001

cache= forward_pass(W,x,B)
y1,y2= cache["y1"], cache["y2"]

loss1= (y1- target1) ** 2 
loss2= (y2 - target2) ** 2
loss= loss1 + loss2

W_old= W.copy()
B_old= B.copy()

grads,bgrads= backward_pass(W,cache,x, target1, target2)
step(W,B,grads,bgrads,lr)

cache_new= forward_pass(W,x,B)
newy1,newy2= cache_new["y1"],cache_new["y2"]

new_loss1= (newy1 - target1) ** 2
new_loss2= (newy2 - target2) ** 2
new_loss= new_loss1 + new_loss2

print("old W,B and loss:", W_old,B_old, loss)
print("new W,B and loss:", W,B, new_loss)


