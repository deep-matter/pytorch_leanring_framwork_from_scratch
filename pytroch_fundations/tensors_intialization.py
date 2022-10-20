import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tonsor = torch.tensor([[1, 2, 3], [5, 6, 8]],
                         dtype=torch.float32,
                         device='cpu', requires_grad=True)
x=my_tonsor[0].detach().numpy()  
y=my_tonsor[1].detach().numpy() 

#fig , ax =plt.subplot()
plt.scatter(x,y)
plt.show()
print(my_tonsor,my_tonsor.type,my_tonsor.device)
print(x,y)

#########################
# Tensors with different intialization 
#########################


my_tonsor_i = torch.empty(size=(3,3))
print("tensor empyt",my_tonsor_i)
my_tonsor_i = torch.zeros((3,3))
print("tensor zeros",my_tonsor_i)
my_tonsor_i = torch.ones((3,3))
print("tensor ones",my_tonsor_i)
my_tonsor_i = torch.rand((3,3))
print("tensor rand",my_tonsor_i)
my_tonsor_i = torch.eye(3,3)
print("tensor eye",my_tonsor_i)
my_tonsor_i = torch.linspace(start=3, end=2,steps=10)
print("tensor linespace",my_tonsor_i)
my_tonsor_i = torch.empty(3,3).uniform_(0,1)
print("tensor mean/std",my_tonsor_i)
####################################
## data type of tensor 
####################################
tenosr_np =torch.zeros(5,5)
print(tenosr_np)
np_array= tenosr_np.numpy()
print(np_array)
tensor_back=torch.from_numpy(np_array)
print(tensor_back)




