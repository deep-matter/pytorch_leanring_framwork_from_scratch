import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from ulits import * 

########## intialze the model 


class RNN(nn.Module):

    def __init__(self,input_size , hidden_size , output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.Hidden_0 = nn.Linear(input_size + hidden_size, hidden_size)
        self.Hidden_1 = nn.L1Loss(input_size + hidden_size, output_size)
        self.activation = nn.LogSoftmax(dim=1) ## [1.57] ===> we want ot hAVE SIZE OF TENSOR.SIZE[57]

    def forward(self,input_tensor , hidden_stat):
        combined_state = torch.cat((input_tensor,hidden_stat),1) 

        hidden_0 = self.Hidden_0(combined_state)  
        output_state = self.Hidden_1(combined_state) 
        output_state = self.activation(output_state)

        return output_state , hidden_0

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)    

###### test the model 

category_line , all_categories = loading_data()
n_categories = len(all_categories)

n_hidden = 182 ## hyperparamters 

model = RNN(num_letters, n_hidden, n_categories)

# One step forward 

input_tensor = letter_to_tensor("A")
hidden_tensor = torch.zeros(1,n_hidden)   

output ,next_hidden = model(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())

