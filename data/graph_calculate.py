import math
import os

import pandas as pd
import pytorch_lightning as L
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_undirected


def get_laplace_square_matrix(edge_index, node_num):
    """
    Args:
        edge_index: （2，edge_num）
        node_num:

    Returns:
        （node_num，node_num）

    """
    edge_num = edge_index.shape[1]
    row, col = edge_index
    data = torch.ones(edge_num, device=edge_index.device)
    adj = torch.sparse.FloatTensor(edge_index, data, torch.Size([node_num, node_num]))
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    degree_mat_inv_sqrt = torch.diag(degree_inv_sqrt)
    laplace_matrix = torch.eye(node_num, device=edge_index.device) - \
                     torch.mm(torch.mm(degree_mat_inv_sqrt, adj.to_dense()), degree_mat_inv_sqrt)

    return laplace_matrix


def get_batch_edge_index(one_edge_index, batch_num, node_num):
    """
    Args:
        one_edge_index: （2，edge_num）
        batch_num:
        node_num:

    Returns:
        （2，batch_num*edge_num）

    """
    edge_index = one_edge_index.clone().detach()
    edge_num = one_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


def eigenvalue_decomposition(L, eigvalue_if_norm=False):
    """
    L=U * Λ * U.T

    Args:
        L: （node_num，node_num）
        eigvalue_if_norm:

    Returns:
        （node_num）, U（node_num，node_num）
    """
    eigenvalue, eigenvector_square = torch.linalg.eigh(L)
    if eigvalue_if_norm:
        indices = eigenvalue.argsort()
        indices2 = indices.argsort()
        eigenvalue = 2 / (eigenvalue.shape[0] - 1) * indices2
    eigenvalue_square = torch.diag_embed(eigenvalue)

    return eigenvalue, eigenvector_square, eigenvalue_square


class Graph_calculate(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.edge_topK = args.edge_topK
        self.last_L_sym_square = None
        self.L_sym_square = None
        self.selfloop_edge_index = None
        self.pyG_batch_edge_index = None
        self.batch_L_sym_square = None
        self.batch_U = None
        self.batch_Λ = None


    def forward(self, batch_x, last_edge_index):
        """

        Args:
            batch_x: (batch_num, lag, node_num)

        Returns:
            contains:
                Data:@pyG_batch_data:   [x=(batch_num*node_num, lag),
                                        edge_index=(2, batch_num*(edge_num*2+node_num)),
                                        batch_L=(batch_num*node_num, batch_num*node_num)
                                        batch_Λ=(batch_num*node_num, batch_num*node_num)
                                        batch_U=(batch_num*node_num, batch_num*node_num)
                Tensor:@selfloop_edge_index: (2, edge_num*2+node_num)
                Tensor:@batch_L_sym_square: (batch_num*node_num, batch_num*node_num)
                Tensor:@L_sym_square: (node_num, node_num)
                Tensor:@batch_x_T: (batch_num, node_num, lag)

        """
        if self.L_sym_square is None:
            batch_num, lag, node_num = batch_x.shape
            data = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path,
                                            '{}_train.csv'.format(self.args.data_name)),
                               sep=',', index_col=False)
            cau_data = torch.from_numpy(data.values[:self.args.graph_ca_lag, :]).to(batch_x.device)
            cau_data = (cau_data - cau_data.min(dim=0)[0]) / (cau_data.max(dim=0)[0] - cau_data.min(dim=0)[0] + 1e-8)
            if self.args.graph_ca_meth == 'Euc':
                cos_ji_mat = torch.ones((node_num, node_num), device=cau_data.device)
                for i in range(node_num):
                    for j in range(i + 1, node_num):
                        cos_ji_mat[i, j] = torch.sqrt(torch.sum((cau_data[:, i] - cau_data[:, j]) ** 2))
                        cos_ji_mat[j, i] = cos_ji_mat[i, j]
                cos_ji_mat = -cos_ji_mat
                cos_ji_mat = cos_ji_mat + (torch.diag(cos_ji_mat.min(dim=0)[0]) - 2)
            else:
                cau_data_T = cau_data.permute(1, 0).contiguous()
                cos_ji_mat = torch.matmul(cau_data_T, cau_data)
                normed_mat = torch.matmul(cau_data_T.norm(dim=-1).view(-1, 1), cau_data_T.norm(dim=-1).view(1, -1))
                cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)
                cos_ji_mat = cos_ji_mat - torch.eye(node_num, device=cos_ji_mat.device)
            topk_indices_ji = torch.topk(cos_ji_mat, self.edge_topK, dim=-1)[1]
            gated_i = torch.arange(0, node_num).to(batch_x.device).T.unsqueeze(1)\
                                                .repeat(1, self.edge_topK).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_i, gated_j), dim=0)
            undirected_edge_index = to_undirected(gated_edge_index)
            selfloop_edge_index, _ = add_self_loops(undirected_edge_index)
            L_sym_square = get_laplace_square_matrix(selfloop_edge_index, node_num)
            pyG_batch_edge_index = get_batch_edge_index(selfloop_edge_index, batch_num, node_num)
            batch_L_sym_square = get_laplace_square_matrix(pyG_batch_edge_index, batch_num * node_num)
            if self.args.plan == 'eig':
                eigenvalue, U, Λ = eigenvalue_decomposition(L_sym_square, self.args.eigvalue_if_norm)
                batch_eigenvalue = torch.zeros((batch_num * node_num), device=batch_x.device)
                batch_U = torch.zeros((batch_num * node_num, batch_num * node_num), device=batch_x.device)
                batch_Λ = torch.zeros((batch_num * node_num, batch_num * node_num), device=batch_x.device)
                for i in range(batch_num):
                    batch_eigenvalue[i * node_num: (i + 1) * node_num] = eigenvalue
                    batch_U[i * node_num: (i + 1) * node_num, i * node_num: (i + 1) * node_num] = U
                    batch_Λ[i * node_num: (i + 1) * node_num, i * node_num: (i + 1) * node_num] = Λ
            elif self.args.plan == 'inv':
                batch_eigenvalue = "No_need"
                batch_U = "No_need"
                batch_Λ = "No_need"
            self.L_sym_square = L_sym_square
            self.selfloop_edge_index = selfloop_edge_index
            self.pyG_batch_edge_index = pyG_batch_edge_index
            self.batch_L_sym_square = batch_L_sym_square
            self.batch_U = batch_U
            self.batch_Λ = batch_Λ
        else:
            L_sym_square = self.L_sym_square
            selfloop_edge_index = self.selfloop_edge_index
            pyG_batch_edge_index = self.pyG_batch_edge_index
            batch_L_sym_square = self.batch_L_sym_square
            batch_U = self.batch_U
            batch_Λ = self.batch_Λ
        batch_num, lag, node_num = batch_x.shape
        batch_x_T = batch_x.permute(0, 2, 1).contiguous()
        pyG_batch_x = batch_x_T.view(batch_num * node_num, lag)
        pyG_batch_data = Data(x=pyG_batch_x, edge_index=pyG_batch_edge_index, batch_L=batch_L_sym_square,
                              batch_Λ=batch_Λ, batch_U=batch_U)

        return pyG_batch_data, selfloop_edge_index, batch_L_sym_square, L_sym_square, batch_x_T

