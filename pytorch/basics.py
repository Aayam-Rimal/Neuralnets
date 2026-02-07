import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.layer1= nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.layer2= nn.Linear(hidden_size, output_size)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x= self.layer1(x)
        x= self.relu(x)
        x= self.layer2(x)
        x= self.sigmoid(x)

        return x
    
model= NeuralNet(1,3,1)

criterion= nn.BCELoss()

optimizer= torch.optim.SGD(model.parameters(), lr=0.01)

X_train= torch.tensor([[1.0], [2.0], [4.0], [5.0]])
Y_train= torch.tensor([[0.0], [1.0], [1.0], [0.0]])



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
    test_input = torch.tensor([[1.5]])
    prediction = model(test_input)
    print(f"\nInput: {test_input.item():.1f}, Prediction: {prediction.item():.4f}")


    
    

