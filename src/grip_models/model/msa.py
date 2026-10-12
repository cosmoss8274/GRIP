"""MSA attention, transition and outer-product layers."""

import logging
import math

from numpy import select
import torch
from torch import nn
from torch.nn import functional as F

from . import primitives


logger = logging.getLogger(__name__)


class MSARow(nn.Module):
    def __init__(self,m_dim,z_dim,N_head=8,c=8):
        super(MSARow,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = primitives.LinearNoBias(m_dim,N_head*c)
        self.klinear = primitives.LinearNoBias(m_dim,N_head*c)
        self.vlinear = primitives.LinearNoBias(m_dim,N_head*c)
        self.norm_z  = nn.LayerNorm(z_dim)
        self.zlinear = primitives.LinearNoBias(z_dim,N_head)
        self.glinear = primitives.Linear(m_dim,N_head*c)
        self.olinear = primitives.Linear(N_head*c,m_dim)

    def forward(self,m,z,mask= True):
        # m : N L 32  
        B,N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(B,N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(B,N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(B,N,L,self.N_head,self.c)
        bias = self.zlinear(self.norm_z(z))
        #print("bias shape:", bias.shape)
        #print("q shape:", q.shape) 
        g = torch.sigmoid(self.glinear(m)).reshape(B, N,L,self.N_head,self.c)
        att=torch.einsum('bnqhc,bnvhc->bnqvh',q,k) * (self.sq_c) + bias[:,None,:,:,:] # rq rv h #[N, L, L, H]
 
        #print("m shape:", m.shape)
        #print("att shape:", att.shape)

 
        att=F.softmax(att,dim=3)
        #print("The attention for MSA row is:", {att})
        o = torch.einsum('bnqvh,bnvhc->bnqhc',att,v) * g
        m_ = self.olinear(o.reshape(B, N,L,-1))

        if torch.isnan(m_).any():
            logger.debug("NaN detected in msarow")

        return m_

class MSACol(nn.Module):
    def __init__(self,m_dim,N_head=8,c=8):
        super(MSACol,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = primitives.LinearNoBias(m_dim,N_head*c)
        self.klinear = primitives.LinearNoBias(m_dim,N_head*c)
        self.vlinear = primitives.LinearNoBias(m_dim,N_head*c)

        self.glinear = primitives.Linear(m_dim,N_head*c)
        self.olinear = primitives.Linear(N_head*c,m_dim)

    def forward(self,m):
        # m : N L 32  
        B,N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(B,N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(B,N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(B,N,L,self.N_head,self.c)

        g = torch.sigmoid(self.glinear(m)).reshape(B,N,L,self.N_head,self.c)

        att=torch.einsum('bslhc,btlhc->bstlh',q,k) * (self.sq_c)  # rq rv h

        #print("The attention shape for MSA col is:", {att.shape})

        att=F.softmax(att,dim=2)
        o = torch.einsum('bstlh,btlhc->bslhc',att,v) * g
        m_ = self.olinear(o.reshape(B,N,L,-1))

        return m_

class MSATrans(nn.Module):
    def __init__(self,m_dim,c_expand=2):
        super(MSATrans,self).__init__()
        self.c_expand=4
        self.m_dim=m_dim
        self.norm=nn.LayerNorm(m_dim)
        self.linear1 = primitives.Linear(m_dim,m_dim*c_expand)
        self.linear2 = primitives.Linear(m_dim*c_expand,m_dim)
    def forward(self,m):
        m = self.norm(m)
        m = self.linear1(m)
        m = self.linear2(F.relu(m))

        if torch.isnan(m).any():
            logger.debug("NaN detected in msatrans")
        return m

class MSAOPM(nn.Module):
    def __init__(self,m_dim,z_dim,c=12):
        super(MSAOPM,self).__init__()
        self.m_dim=m_dim
        self.c=c
        self.norm=nn.LayerNorm(m_dim)
        self.linear1=primitives.Linear(m_dim,c)
        self.linear2=primitives.Linear(m_dim,c)
        self.linear3=primitives.Linear(c*c,z_dim)
    def forward(self,m):
        B,N,L,D=m.shape
        o=self.norm(m)
        a=self.linear2(o)
        b=self.linear1(o)
        o = torch.einsum('knia,knjb->knijab',a,b).mean(dim=1)# in n dim get mean. k is added batch dim
        o = self.linear3(o.reshape(B,L,L,-1))

        return o


        




