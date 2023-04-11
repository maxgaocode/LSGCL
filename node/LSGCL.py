
from time import perf_counter as t
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
import torch
import torch.nn as nn
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
class List_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, bias=True, **kwargs):
        super(List_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        list_mat = []
        list_mat.append(x)

        # D^(-0.5)AD^(-0.5)
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        for i in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm, size=None)
            list_mat.append(x)
        return list_mat




class LSGCL(nn.Module):
    def __init__(self, nfeat, eachdim, num_projection, tau, layer_num,  activation="relu",**kwargs):
        super(LSGCL, self).__init__()

        self.numhidden = eachdim
        self.tau = tau
        self.layer_num = layer_num
        self.lin1 = nn.ModuleList([nn.Linear(nfeat, self.numhidden) for _ in range(layer_num+1)])#there
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

        self.edge_weight = None

        self.prop1 = List_prop(self.layer_num)
        self.layers = nn.ModuleList()

        self.fc1 = nn.Linear(num_projection, num_projection)
        self.fc2 = nn.Linear(num_projection, num_projection)

    def forward(self, h, edge_index, Norm):

        list_mat = self.prop1(h, edge_index)

        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.lin1[ind](mat)
            if Norm==1:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)

        return final_mat

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def semi_loss(self, z1):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))

        return torch.log(refl_sim.sum(1))

    def loss(self, z1, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        if batch_size == 0:
            l1 = self.semi_loss(h1)
        else:
            l1 = self.batched_semi_loss(h1, batch_size)

        ret = l1
        ret = ret.mean() if mean else ret.sum()
        return ret



