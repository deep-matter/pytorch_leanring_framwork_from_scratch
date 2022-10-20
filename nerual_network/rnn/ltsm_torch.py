import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
import torchvision.datasets as datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## loading the dataset 
training_set = datasets.MNIST(
    root='dataset/', train=True, transform=Transforms.ToTensor(), download=False)
testing_set = datasets.MNIST(
    root='dataset/', train=False, transform=Transforms.ToTensor(), download=False)
training_laoding = DataLoader(
    dataset=training_set, batch_size=64, shuffle=True, num_workers=1)
training_loading = DataLoader(
    dataset=testing_set, batch_size=64, shuffle=True, num_workers=1)
### intialze the Hpyerparamter of model 
input_size = 28
hidden_size = 250 
sequence_length = 28 
hidden_layer = 6
num_classes = 10 

### modeling the neural network RNN 


class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, hidden_layer,sequence_length, num_classes):
        super(RNN,self).__init__() 
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.ltsm= nn.LSTM(input_size,hidden_size, hidden_layer, batch_first=True)  
        self.dense = nn.Linear(hidden_size*sequence_length, num_classes)  

    def forward(self, x):
        # inrialize hidden state 
        h0 = torch.zeros(hidden_layer, x.size(0),self.hidden_size).to(device=device)
        c0 = torch.zeros(hidden_layer, x.size(0),self.hidden_size).to(device=device) 
        #here nly we care about comes from hidden state  
        out, _ = self.ltsm( x, (h0 , c0)) 
        out  = out.reshape(out.shape[0],-1)
        out = self.dense(out)

        return out 

Epochs = 2
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
#batch_size = 64
#### intialzie the model 
model = RNN(input_size, hidden_size, hidden_layer, sequence_length, num_classes).to(device=device)
optimizer= optim.Adam(model.parameters(),lr = learning_rate)
# x = torch.rand(64,1,28,28).squeeze(1).to(device=device)
# print(model(x))
if torch.cuda.is_available():
    for epoch in range(Epochs):
        for idx ,(images , labels) in enumerate(training_loading):
            num_correct = 0
            num_samples = 0 
            images = images.squeeze(1).to(device=device)
            labels = labels.to(device = device )

            ## here we done't need to reshape the tensoer to (64x784) 
            #images = images.reshape(images.shope[0],-1)
            score = model(images)
            loss = criterion(score , labels)
            _,pred_class= score.max(1)
            num_correct += (pred_class == labels).sum()
            num_samples += pred_class.size(0)
            accuracy = num_correct/num_samples
            print(f"the loss tracking {loss} , the accuracy performe {accuracy}")
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()







