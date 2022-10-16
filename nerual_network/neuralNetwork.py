import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#####################
# intialization the model class Neural of input_size and nmuber classes
#####################


class Neural(nn.Module):

    def __init__(self, input_size, num_classes):

        super(Neural, self).__init__()
        self.layer_1 = nn.Linear(input_size, 50)
        self.outputs = nn.Linear(50, num_classes)

    def forward(self, x):

        x = F.relu(self.layer_1(x))
        x = self.outputs(x)
        return x


# model=Neural(764,10)
# x=torch.rand(64,764)
# print(model(x).shape)
# devices = " cuda" GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
input_size = 784
num_classes = 10
batch_size = 64
num_workers=1
learning_rate = 0.001
epochs = 2

# loading the dataser

train_set = datasets.MNIST(root='dataset/', train=True,
                           transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST(root='dataset/', train=False,
                          transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# intialize the model

model = Neural(input_size, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Loop training model

for epoch in range(epochs):
    for index_batchs , (img ,labels) in enumerate(train_loader):
        img = img.to(device=device)
        #print(img.shape[0])
        labels = labels.to(device=device) ## pass data to available GOU in Computer 
        img = img.reshape(img.shape[0],-1)
        #print(img.shape) ## flatten images dim to 764 input_size
        score = model(img)
        loss = criterion(score,labels)
        print(loss)
        ### backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0 
    model.eval()
    with torch.no_grad():
        for x , y in loader :
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            score = model(x)
            _,pred_class= score.max(1)
            num_correct += (pred_class == y).sum()
            num_samples += pred_class.size(0)
    model.train()

    return num_correct/num_samples

# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")        


        
