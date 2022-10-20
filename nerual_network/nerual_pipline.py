import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
###############
# input shapoe of images is set inot [batchz_size = 1 , input_channel = 3,32 , 32 ]
Grayscale_channel = 3
Transforms = transforms.Compose([
    transforms.Resize((28, 28)),  # resize the imagge into 28x28
    transforms.Grayscale(num_output_channels=Grayscale_channel),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)
# loading the dataste

train = datasets.CIFAR10(root="CIFAR20", train=True,
                         download=False, transform=Transforms)
test = datasets.CIFAR10(root="CIFAR20", train=False,
                        download=False, transform=Transforms)
train_loading = DataLoader(dataset=train, batch_size=1,
                           shuffle=True, num_workers=1)
test_loding = DataLoader(dataset=test, batch_size=1,
                         shuffle=True, num_workers=1)

############## modeling the convolution NEt 

class Convolutions(nn.Module):
    def __init__(self, in_channel ,out_channel,num_classes ):

        super(Convolutions,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=16,kernel_size=(3,3))
        self.full_conncted = nn.Linear(36*4*4 , num_classes) #16 * 7*7

    def forward(self,input_tensor):

        x = self.conv1(input_tensor)   
        x = F.relu(x) 
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        output = x.reshape(x.shape[0],-1)
        output = self.full_conncted(output)

        return output
########## testing model tensors 

#sample = torch.rand(1,3,28,28)
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print(model(sample).shape )

############# Hyperamaters 
Epochs = 2 
lr = 0.001
criterion = nn.CrossEntropyLoss()
model = Convolutions(3 ,8,num_classes=len(classes))
optimizer = optim.Adam(model.parameters(), lr = lr )

def Train(epochs):
    if torch.cuda.is_available():

        ########## loading dataset from Dataloader class 
       sample_1 = iter(train_loading)
        #images , labels = sample_1.next()
        
       for step in range(epochs):
            current_state = 0 
            sample_out = 0
            running_loss = 0.0

            for idx , (image,label) in enumerate(sample_1):
                
                output = model(image)
                loss   = criterion(output , label)
                _ , precdicted_class = output.max(1)
                current_state += (precdicted_class == label).sum()
                sample_out += precdicted_class.size(0)
                accuracy = current_state/sample_out
                running_loss += loss.item()
                if idx % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (step + 1, idx + 1, running_loss / 2000))
                    print(f"the accuracy performe {accuracy}")
                running_loss = 0.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

if __name__ == "__main__":

    run_train = Train(Epochs)
    run_train
 








         


