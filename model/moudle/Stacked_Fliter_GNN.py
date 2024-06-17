import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
import torch.nn.functional as F


class Stacked_Filter(nn.Module):

    def __init__(self, args):
        super(Stacked_Filter, self).__init__()
        self.args = args
        self.K = args.K
        # self.residual_alpha = args.residual_alpha

        self.residual_alpha = Parameter(torch.FloatTensor([1]))

        self.a_list = Parameter(torch.Tensor(args.K))
        self.b_list = Parameter(torch.Tensor(args.K))
        self.c_list = Parameter(torch.Tensor(args.K))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.a_list, a=0.0, b=2.0)
        torch.nn.init.uniform_(self.b_list, a=0.0, b=2.0)
        torch.nn.init.uniform_(self.c_list, a=-1.0, b=1.0)


    def forward(self, data):
        batch_x = data.x
        batch_edge_index = data.edge_index
        batch_L = data.batch_L
        batch_Λ = data.batch_Λ
        batch_U = data.batch_U

        # filter_list = []
        for i in range(self.K):
            if self.args.plan == 'eig':
                filter_rep = (self.a_list[i] * torch.eye(batch_Λ.shape[0], device=batch_Λ.device) - batch_Λ)\
                             @ (batch_Λ - self.b_list[i] * torch.eye(batch_Λ.shape[0], device=batch_Λ.device))
                filter_rep_up = F.relu(filter_rep)
                filter_i = self.c_list[i]/((self.a_list[i]-self.b_list[i])*(self.a_list[i]-self.b_list[i])/4) * batch_U\
                           @ filter_rep_up \
                           @ batch_U.permute(1, 0) \
                           @ batch_x
            elif self.args.plan == 'inv':
                print("请将args.plan改成eig")
            # filter_list.append(filter_i)
            if i == 0:
                out = self.residual_alpha * batch_x + filter_i
            else:
                out = out + filter_i

        out_data = Data(x=out, edge_index=batch_edge_index, batch_L=batch_L, batch_Λ=batch_Λ, batch_U=batch_U)

        return out_data


class Stacked_Filter_GNN(nn.Module):
    def __init__(self, args):
        super(Stacked_Filter_GNN, self).__init__()

        self.args = args
        self.Stacked_Filter = Stacked_Filter(args)

    def forward(self, data):
        """
        Args:
            data:

        Returns:
            out_x=(batch_size, node_num, lag)

        """
        for i in range(self.args.layers):
            data = self.Stacked_Filter(data)
        filter_x = data.x.view(self.args.batch_size, self.args.node_num, self.args.lag)
        out_x = layer_normalize(filter_x)
        # (batch_size, node_num, lag)

        return out_x

def layer_normalize(batch_x):
    """

    Args:
        batch_x: (batch_size, node_num, lag)

    Returns:

    """
    # 逐 node_num
    min_val, _ = torch.min(torch.min(batch_x, dim=0)[0], dim=1, keepdim=True)
    # (node_num, 1)
    min = min_val.repeat(batch_x.shape[0], 1, batch_x.shape[2])
    # (batch_size, node_num, lag)
    max_val, _ = torch.max(torch.max(batch_x, dim=0)[0], dim=1, keepdim=True)
    # (1, node_num, 1)
    max = max_val.repeat(batch_x.shape[0], 1, batch_x.shape[2])
    # (batch_size, node_num, lag)
    batch_x_norm = (batch_x - min) / (max - min + 1e-8)
    # (batch_size, node_num, lag)

    return batch_x_norm
