import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmptyOut:

    def __init__(self,dim ,outs):

        self.empty_1D = torch.empty(size=dim,out=outs)
          
        return print(f"final outpist of empyt list :{self.empty_1D}")
   

    def dim_n(self,dim):

        empty_2D=torch.empty(size=dim)

        return print(f"results of matrix size dim {empty_2D}")



# # additional two 1D matrix
# tensor_x = torch.rand(64,1,28,28)
# outs_neural=tensor_x.unsqueeze(1)
# print(tensor_x.shape)
# print(outs_neural.shape)

x = torch.rand(2,5) #n*m
y=torch.rand(1,5) #m*p
z=torch.mul(x,y) #====> n*p
print(z)

#tensor_y = torch.rand(2,2)

# outs=torch.mul(tensor_x,tensor_y).sum()
# outs_x=torch.mul(tensor_x,tensor_y)
# outs_y=outs_x.reshape(outs_x.shape[0],-1)
# re_size_shape=outs_x.view(2,-1)
# print(outs_y)
# print(re_size_shape.shape)
# empty_tensor=EmptyOut(outs)

# # multiplicatio 1D tensor
# outs = torch.mul(tensor_x, tensor_y)
# empty_tensor = EmptyOut(outs)

# power_tensor=tensor_x.pow(2)
# empty_tensor=EmptyOut(power_tensor)

# # ## inplace tensor 

# inplace = tensor_x.add_(tensor_x)
# print(inplace)
# ## other method 
# tensor_x += tensor_x
# empty_tensor=EmptyOut(tensor_x)


# ####################
# # matrices multiplications 
# ###################

# matrix_x=torch.rand(3,3)
# matrix_y=torch.empty(3,3).uniform_(1,2)
# mul_matrix=torch.mm(matrix_x,matrix_y)
# results_add=EmptyOut((3,3),mul_matrix)
# results_add.dim_n((3,3))

##########
# exponotional matrix 
##########
# dot_product=torch.dot(tensor_x,tensor_y)
# print(dot_product)
# power_matrix = torch.matrix_power(matrix_x, 2)

# results_pow=EmptyOut((3,3),power_matrix)
# results_pow.dim_n((3,3))

#print(results)
################
## batch mul matrices 
#################
# batchs=2
# n=2
# m=4
# p=2
# ax =plt.axes(projection='3d')
# batch_1=torch.rand(batchs,n,m)####[32,12,36]
# batch_2=torch.rand(batchs,n,m)
# z = np.sin(torch.add(batch_1,batch_2))

# X ,Y = np.meshgrid(batch_1,batch_2)

# Z = np.sin(X) * np.cos(Y) 
# z =torch.bmm(batch_1, batch_2)  
# print(z.shape)

# ax.plot_surface(X,Y,Z,color='red',alpha=0.5)
# plt.show()

#z =torch.bmm(batch_1, batch_2)  
# print(z.size())
# z.size()
##################
# Other operations
###################

#sum_x_and_y=torch.sum(batch_1,dim=0)
# sum_col=torch.mean(tensor_x.float(),dim=0)
# sorted_tensor,indices=torch.sort(tensor_x,dim=0 ,descending=False)
# clamp_t=torch.clamp(tensor_x,min=2,max=3)
# print(clamp_t)
# print(sorted_tensor)
# print(sum_col)
# value, index = torch.argmax(batch_2,dim=0)
# print(value, index )