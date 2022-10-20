import torch 


batch_size =3
features = 3
x = torch.tensor([[1,2,3],[3,5,6],[5,9,9]],device='cuda')
z=torch.rand(12,24,14)
y=torch.randint(2, (5,5))
print(z.ndimension())

# print(x[0:3,1:2])
# print(x[x.remainder(2)==0])
#print(x)

#print(x[0:, 0:13])

# x1 = torch.rand(3,4)
# row = torch.tensor([1,2])
# col =torch.tensor([2,3])

# print(x1[row,col].shape)