import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



class EmptyOut:

    def __init__(self,outs):

        self.empty_1D = torch.empty(3,out=outs)
          
        return print(f"final outpist of empyt list :{self.empty_1D}")
   

    def dim_n(self,dim):

        empty_2D=torch.empty(size=dim)

        return print(f"results of matrix size dim {empty_2D}")



# additional two 1D matrix
tensor_x = torch.tensor([1, 2, 3])
tensor_y = torch.tensor([1, 5, 4])

outs=torch.add(tensor_x,tensor_y)
empty_tensor=EmptyOut(outs)

# multiplicatio 1D tensor
outs = torch.mul(tensor_x, tensor_y)
empty_tensor = EmptyOut(outs)

power_tensor=tensor_x.pow(2)
empty_tensor=EmptyOut(power_tensor)

# ## inplace tensor 

inplace = tensor_x.add_(tensor_x)
print(inplace)
## other method 
tensor_x += tensor_x
empty_tensor=EmptyOut(tensor_x)


####################
# matrices multiplications 
###################

matrix_x=torch.rand(3,3)
matrix_y=torch.empty(3,3).uniform_(1,2)

mul_matrix=torch.mm(matrix_x,matrix_y)

results=EmptyOut(mul_matrix)
results.dim_n((3,3))

#print(results)


