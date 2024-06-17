from torch import nn
from model.moudle.Stacked_Fliter_GNN import Stacked_Filter_GNN


class GNN_spatial_block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.spatial_method == 'Stacked_Filter_GNN':
            self.Spatial_Net = Stacked_Filter_GNN(args)

    def forward(self, data):
        """
        图滤波

        Args:
            data:

        Returns:
            out_x=(batch_size, node_num, lag)

        """
        return self.Spatial_Net(data)



