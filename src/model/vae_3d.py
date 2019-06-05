import torch
import torch.nn as nn
import numpy as np

class SimpleVAE3D(nn.Module):
  
  def __init__(self,aspect_ratio = [8,10,3],dim = 220):
    super().__init__()
    self.aspect_ratio = aspect_ratio
    self.dim_before_reparam = np.prod(self.aspect_ratio)
    self.enc = nn.Sequential(nn.Conv3d(1,16,5,stride = 2,padding = 2),nn.ReLU(),nn.Conv3d(16,32,5,stride = 2,padding = 2),nn.ReLU(),
                             nn.Conv3d(32,64,5,stride = 2,padding = 2),nn.ReLU(),nn.Conv3d(64,64,5,stride = 2,padding = 2),nn.ReLU())
    self.dec = nn.Sequential(nn.ConvTranspose3d(64,64,4,stride = 2,padding = 1),nn.ReLU(),
                              nn.ConvTranspose3d(64,32,4,stride = 2,padding = 1),nn.ReLU(),
                              nn.ConvTranspose3d(32,16,4,stride = 2,padding = 1),nn.ReLU(),
                              nn.ConvTranspose3d(16,1,4,stride = 2,padding = 1),nn.Sigmoid())
    self.mu_net = nn.Linear(64 * self.dim_before_reparam,dim)
    self.var_net = nn.Linear(64 * self.dim_before_reparam,dim)
    self.decode_net = nn.Linear(dim,64 * self.dim_before_reparam)
    #self.reparameterize = 
  def forward(self,x):
    x = x.permute(0,3,1,2)
    x = x[:,None,:,:,:]
    out = self.enc(x)
    out = out.view(-1, 64 * self.dim_before_reparam)
    #print(out)
    mu_out = self.mu_net(out)
    var_out = self.var_net(out)
    z = self.reparameterize(mu_out,var_out)
    #print(z)
    decoded = self.decode_net(z)
    decoded = decoded.view(-1,64,self.aspect_ratio[-1],*self.aspect_ratio[:-1])
    decoded = self.dec(decoded)
    decoded = torch.squeeze(decoded,dim = 1).permute(0,2,3,1)
    return decoded,mu_out,var_out
  
  def reparameterize(self,mu,log_var):
    std = log_var.mul(.5).exp_()
    if torch.cuda.is_available():
      eps = torch.cuda.FloatTensor(std.size()).normal_()
    else:
      eps = torch.FloatTensor(std.size()).normal_()      
    return eps.mul(std).add_(mu)
  
