import torch
from torch import nn
from torch.nn import functional as F
from . import pair, msa


class EvoBlock(nn.Module):

    def __init__(self, m_dim, z_dim, docheck=False):
        super(EvoBlock, self).__init__()
        self.msa_row = msa.MSARow(m_dim, z_dim)
        self.msa_trans = msa.MSATrans(m_dim)
        self.msa_opm = msa.MSAOPM(m_dim, z_dim)
        self.pair_triout = pair.TriOut(z_dim)
        self.pair_triin = pair.TriIn(z_dim)
        self.pair_trans = pair.PairTrans(z_dim)
        self.docheck = docheck

    def layerfunc_msa_row(self, m, z):
        return self.msa_row(m, z) + m

    def layerfunc_msa_trans(self, m):
        return self.msa_trans(m) + m

    def layerfunc_msa_opm(self, m, z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self, z):
        return self.pair_triout(z) + z

    def layerfunc_pair_triin(self, z):
        return self.pair_triin(z) + z

    def layerfunc_pair_trans(self, z):
        return self.pair_trans(z) + z

    def forward(self, m, z):
        m = m + self.msa_row(m, z)
        m = m + self.msa_trans(m)
        z = z + self.msa_opm(m)
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_trans(z)
        return (m, z)


class Evoformer(nn.Module):

    def __init__(self, m_dim, z_dim, docheck=False):
        super(Evoformer, self).__init__()
        self.num_layers = 4
        self.docheck = docheck
        if docheck:
            pass
        self.evos = nn.ModuleList(
            [EvoBlock(m_dim, z_dim, True) for _ in range(self.num_layers)]
        )

    def layerfunc(self, layermodule, m, z):
        m_, z_ = layermodule(m, z)
        return (m_, z_)

    def forward(self, m, z):
        for i in range(self.num_layers):
            m, z = self.evos[i](m, z)
        return (m, z)
