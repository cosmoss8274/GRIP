import torch
from torch import nn
from torch.nn import functional as F
import basic
import math

# m : N L 32  
class TriOut(nn.Module):
    def __init__(self,z_dim,c=32):
        super(TriOut,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)

    def forward(self,z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
        o = torch.einsum('oilc,ojlc->oijc',a,b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o

class TriIn(nn.Module):
    def __init__(self,z_dim,c=32):
        super(TriIn,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)

    def forward(self,z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
        o = torch.einsum('olic,oljc->oijc',a,b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o


class TriAttStart(nn.Module):
    def __init__(self,z_dim,N_head=4,c=8):
        super(TriAttStart,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)

    def forward(self,z_, mask= True):#z:b,l,l, z_dim
        B, L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(B,L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('oblhc,obkhc->oblkh',q,k)*self.sq_c + b[:,None,:,:,:]#o is added batch dim
        print("The attention shape for pair is:", att.shape)
        #mask
        if mask is not None:
            mask_condition = (z.sum(dim=-1) == 0).unsqueeze(-1).unsqueeze(-1) #b, l, l, h
            att = att.masked_fill(mask_condition == 0, float('-1e9'))

       
        att = F.softmax(att,dim=3)

        o = torch.einsum('oblkh,obkhc->oblhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(B,L1,L2,self.N_head,self.c)) * o).reshape(B,L1,L2,-1)
        o = self.olinear(o)
        return o

class TriAttEnd(nn.Module):
    def __init__(self,z_dim,N_head=4,c=8):
        super(TriAttEnd,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)

    def forward(self,z_,mask=True):
        B,L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(B,L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('oblhc,okbhc->oblkh',q,k)*self.sq_c + b[:,None,:,:,:].permute(0,1,3,2,4)
        print(att.shape)

         #mask
        if mask is not None:
            mask_condition = (z.sum(dim=-1) == 0).unsqueeze(-1) #b, l, l, 1
            att = att.masked_fill(mask_condition == 0, float('-1e9'))


        att = F.softmax(att,dim=3)
        o = torch.einsum('oblkh,oklhc->oblhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(B,L1,L2,self.N_head,self.c)) * o).reshape(B,L1,L2,-1)
        o = self.olinear(o)
        return o
    def forward2(self,z_):
        z = z_.permute(0,2,1,3)
        B,L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(B,L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(B,L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('oblhc,obkhc->oblkh',q,k)*self.sq_c + b[:,None,:,:,:]
        att = F.softmax(att,dim=3)
        o = torch.einsum('oblkh,obkhc->oblhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(B,L1,L2,self.N_head,self.c)) * o).reshape(B,L1,L2,-1)
        o = self.olinear(o)
        o = o.permute(0,2,1,3)
        return o
class PairTrans(nn.Module):
    def __init__(self,z_dim,c_expand=2):
        super(PairTrans,self).__init__()
        self.z_dim=z_dim
        self.c_expand=c_expand
        self.norm = nn.LayerNorm(z_dim)
        self.linear1=basic.Linear(z_dim,z_dim*c_expand)
        self.linear2=basic.Linear(z_dim*c_expand,z_dim)
    def forward(self,z):
        a = self.linear1(self.norm(z))
        a = self.linear2(F.relu(a))
        return a 

