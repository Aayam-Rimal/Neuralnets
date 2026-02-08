import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

X= torch.randn(100, 5)
Y= (X.sum(dim=1)>0).float().unsqueeze(1) # unsqueeze sets dimension 1 as 1 here (100, 1)

x_train, y_train= X[:80], Y[:80]
x_test,  y_test=  X[80:], Y[80:]

train_dataset= TensorDataset(x_train,y_train)
train_loader= DataLoader(train_dataset, batch_size=16, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1= nn.Linear(5, 10)
        self.relu1= nn.ReLU()
        self.layer2= nn.Linear(10, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x= self.relu1(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
    
model= Classifier()

criterion= nn.BCELoss()

optimizer= torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    epoch_loss=0

    for X_batch,Y_batch in train_loader:
        outputs= model(X_batch)
        loss= criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+= loss.item()


    if epoch % 10 == 0:
     print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")


#inference

model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_preds = (test_outputs > 0.5).float()
    accuracy = (test_preds == y_test).float().mean()
    
print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")
 
