import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

""" Loss-PM """
class Contrastive_loss(nn.Module):
    def __init__(self,tau):
        super(Contrastive_loss,self).__init__()
        self.tau = tau

    def sim(self,z1:torch.Tensor,z2:torch.Tensor):

        z1 = F.normalize(z1) # 640*128
        z2 = F.normalize(z2) # video: 640*32*128 x 640*32*128
        leng = len(z1.shape)

        if leng == 2:
            
            smi = torch.mm(z1,z2.t())
        else:
            smi = torch.mm(torch.mean(z1,dim=1), (torch.mean(z2,dim=1)).t())
        

        return smi  # torch.mm 矩阵乘法
    
    def semi_loss(self,z1:torch.Tensor,z2:torch.Tensor):

        f=lambda x: torch.exp(x/self.tau)

        """ Return the expression torch.exp(x / self.tau) """

        refl_sim = f(self.sim(z1,z2))
        between_sim=f(self.sim(z1,z2))

        # Used to calculate the negative log-likelihood loss: refl_sim.sum(1) sums each row of the refl_sim matrix.

        return -torch.log(between_sim.diag() / (refl_sim.sum(1)+between_sim.sum(1)-refl_sim.diag()))
    
    def forward(self, z1:torch.Tensor, z2:torch.Tensor, mean:bool=True):

        l1=self.semi_loss(z1,z2)
        l2=self.semi_loss(z2,z1)
        ret=(l1+l2)*0.5
        ret=ret.mean() if mean else ret.sum()

        return ret

def reparameterise(mu, logvar):

        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)  # Generate random numbers with the same shape as the standard deviation.
        sampler = epsilon * std  # Generate sampling results.
    
        return mu + sampler