import torch 


# x= torch.arange(9)
# x_v=x.view(3,3)
# x_r = x.reshape(3,3)
# # print(x_r.shape)
# # print(x_r)

# transpose= x_v.t()
# print(transpose)
# print(x_r.contiguous().view(9))

x=torch.rand(3,5,6)
out = x.reshape(x.shape[0],-1)
out1 = x.size(3,5) # tensore [1d .2d , 3d ] ==> 
#y= torch.rand(3,5)

print(out.shape,out1.shape)
#print(x[1::1,1])
# conta=torch.cat((x,y),dim=0)
# print(conta.unsqueeze(-1).shape)
# ## flatten to one dimessionality 
# print(conta.ndimension())
# batch=5
# m= torch.rand(batch,2,3)
# seq=m.squeeze(1)
# print(seq.view(-1))
# print(m.view(batch,-1).shape)
# print(m.permute(0,2,1).shape)
# img= torch.rand(64,1,28,28)
# img1=img.reshape(img.shape[0],-1).squeeze(-1)
# print(img1.shape)
