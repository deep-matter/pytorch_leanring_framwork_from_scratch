import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


tensor_x = torch.tensor([1,2,3])
tensor_y = torch.tensor([1,5,4])


empty_tensor=torch.empty(3)
print(torch.add(tensor_x,tensor_y))

