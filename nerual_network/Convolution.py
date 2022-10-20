import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
import torchvision.datasets as datasets


# modeling the neural network

# in convolution Net we have demissios for 64x1x28x28 [batchs , in_channels,size_x,size_y]


class CNN_Net(nn.Module):
    def __init__(self, in_channel, out_channel, num_class):
        super(CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.dense = nn.Linear(36*4*4, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)

        return x

# test input shape of the Tensor
model = CNN_Net(in_channel=1,out_channel=8,num_class=10)

x = torch.randn(64,1,28,28)


print(model(x).shape)
exit()
 # loading the dataset
training_set = datasets.MNIST(
    root='dataset/', train=True, transform=Transforms.ToTensor(), download=False)
testing_set = datasets.MNIST(
    root='dataset/', train=False, transform=Transforms.ToTensor(), download=False)
training_laoding = DataLoader(
    dataset=training_set, batch_size=64, shuffle=True, num_workers=1)
training_loading = DataLoader(
    dataset=testing_set, batch_size=64, shuffle=True, num_workers=1)

###### Hyperparamters 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_iter = iter(training_laoding)
images , labels = next(data_iter )
print(images.squeeze().shape)
# x = torch.randn(4,1,4)
# print(x.ndimension())
# y = x.unsqueeze(0)
# print(y.shape)


exit()
Epochs = 2
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
batch_size = 64
#### intialzie the model 
model = CNN_Net(in_channel=1, out_channel=8, num_class=10).to(device=device)
optimizer= optim.Adam(model.parameters(),lr = learning_rate)


if torch.cuda.is_available():
    for epoch in range(Epochs):
        for idx ,(images , labels) in enumerate(training_loading):
            num_correct = 0
            num_samples = 0 
            images = images.to(device=device)
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





