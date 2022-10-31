import torch
import matplotlib.pyplot as plt

# tensor multiplicatio wiht equiivqlnet shape
# a = torch.rand(2, 2)
# b = torch.rand(2, 2)
# c = torch.zeros(2, 2)
# print(c)
# old_id = id(c)
# d = torch.matmul(a, b, out=c)
# try:
#     assert c is d
#     assert id(c), old_id
#     print(c)
#     print(d)
# except AssertionError as msg:
#     print(msg)

######## change some indicies values inside the tenosrs

# m = torch.ones(2,2)
# print(m)
# h = m 
# m[0][1] = 12

# print(h)
# print(torch.eq(h,m))

## copy the data between the object
# a = torch.rand(3,3)
# b = a.clone()
# assert b is not a 
# print(torch.eq(a,b))

## copy the data that turn on autgrade wieghts 
a = torch.rand(2,2 , requires_grad=True )
print(a)
b = a.clone()
print(b)
c = a.detach().clone()
print(c)
print(a)
