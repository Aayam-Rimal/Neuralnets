import torch
import torch.nn as nn

class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1= nn.Linear(5,10)
        self.relu= nn.ReLU()
        self.layer2= nn.Linear(10, 5)
        self.relu= nn.ReLU()
        self.layer3= nn.Linear(5, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):

        x= self.layer1(x)
        x= self.relu(x)
        x= self.layer2(x)
        x= self.relu(x)
        x= self.layer3(x)
        x= self.sigmoid(x)

        return x
    

model= BiggerNet()

criterion= nn.BCELoss()

optimizer= torch.optim.SGD(model.parameters(), lr=0.1)

torch.manual_seed(42)

X_train= torch.randn(100,5)
Y_train= torch.randint(0, 2, (100, 1)).float() # note: this is not a real relationship , Learning is useless here 

for epoch in range(1000):

    output= model(X_train)
    loss= criterion(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
      print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


model.eval()
with torch.no_grad():
    X_input= torch.randn(10, 5)
    output= model(X_input)
    print(f"input : {X_input} | prediction : {output}")





