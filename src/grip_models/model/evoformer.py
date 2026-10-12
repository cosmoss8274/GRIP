from torch import nn
from . import pair, msa


class EvoBlock(nn.Module):

    def __init__(self, m_dim, z_dim):
        super(EvoBlock, self).__init__()
        self.msa_row = msa.MSARow(m_dim, z_dim)
        self.msa_trans = msa.MSATrans(m_dim)
        self.msa_opm = msa.MSAOPM(m_dim, z_dim)
        self.pair_triout = pair.TriOut(z_dim)
        self.pair_triin = pair.TriIn(z_dim)
        self.pair_trans = pair.PairTrans(z_dim)

    def forward(self, m, z):
        m = m + self.msa_row(m, z)
        m = m + self.msa_trans(m)
        z = z + self.msa_opm(m)
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_trans(z)
        return (m, z)


class Evoformer(nn.Module):

    def __init__(self, m_dim, z_dim, num_layers=4):
        super(Evoformer, self).__init__()
        self.num_layers = num_layers
        self.evos = nn.ModuleList(
            [EvoBlock(m_dim, z_dim) for _ in range(self.num_layers)]
        )

    def forward(self, m, z):
        for i in range(self.num_layers):
            m, z = self.evos[i](m, z)
        return (m, z)
