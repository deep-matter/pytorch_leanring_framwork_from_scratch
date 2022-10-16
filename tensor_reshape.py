import torch 


# x= torch.arange(9)
# x_v=x.view(3,3)
# x_r = x.reshape(3,3)
# # print(x_r.shape)
# # print(x_r)

# transpose= x_v.t()
# print(transpose)
# print(x_r.contiguous().view(9))

x=torch.rand(3,5)
y= torch.rand(3,5)
conta=torch.cat((x,y),dim=1)
print(conta.view(-1).squeeze(0).shape)
## flatten to one dimessionality 
print(conta.ndimension())
# batch=5
# m= torch.rand(batch,2,3)
# seq=m.squeeze(1)
# print(seq.view(-1))
# print(m.view(batch,-1).shape)
# print(m.permute(0,2,1).shape)

