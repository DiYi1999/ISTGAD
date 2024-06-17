import torch.nn as nn
from model.block.spatial_block import GNN_spatial_block
from model.block.temporal_block import TCN_temporal_block, dilated_temporal_block, GRU_temporal_block, MLP_temporal_block


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.z_dim = args.z_dim
        self.args.LeakyReLU_slope = args.LeakyReLU_slope
        self.args.map_block_dropout = args.dropout

        self.spatial_block = GNN_spatial_block(args)

        if args.temporal_method == 'TCN':
            self.temporal_block = TCN_temporal_block(args.node_num, args.E_layers_channels, args.dilated_kernel_size, args.dropout)
            # if args.E_layers_channels[-1] != 1:
            #     self.map_block_pre = nn.Sequential(
            #         nn.Linear(args.E_layers_channels[-1], 1),
            #         nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
            #         nn.Dropout(p=args.map_block_dropout)
            #     )

        elif args.temporal_method == 'dilated_convolution':
            self.temporal_block = dilated_temporal_block(args.node_num, args.E_layers_channels, args.dilated_kernel_size, args.dropout)
            # if args.E_layers_channels[-1] != 1:
            #     self.map_block_pre = nn.Sequential(
            #         nn.Linear(args.E_layers_channels[-1], 1),
            #         nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
            #         nn.Dropout(p=args.map_block_dropout)
            #     )

        elif args.temporal_method == 'GRU':
            self.temporal_block = GRU_temporal_block(args.node_num, args.GRU_hidden_num, args.GRU_layers, args.dropout)
            # if args.GRU_hidden_num != 1:
            #     self.map_block_pre = nn.Sequential(
            #         nn.Linear(args.GRU_hidden_num, 1),
            #         nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
            #         nn.Dropout(p=args.map_block_dropout)
            #     )

        elif args.temporal_method == 'MLP':
            self.temporal_block = MLP_temporal_block(args.node_num, args.E_layers_channels, args.dropout)
            #### 这里dropout用的是模型的args.dropout，默认为0的那种，要是训练部不起来，可以改为用args.map_block_dropout，那个默认0.2
            # if args.E_layers_channels[-1] != 1:
            #     self.map_block_pre = nn.Sequential(
            #         nn.Linear(args.E_layers_channels[-1], 1),
            #         nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
            #         nn.Dropout(p=args.map_block_dropout)
            #     )

        # 定义个这最终映射MLP
        if args.temporal_method == 'GRU':
            self.map_block = nn.Sequential(
                nn.Linear(args.GRU_hidden_num * args.lag, args.z_dim),
                nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
                nn.Dropout(p=args.map_block_dropout)
            )
        else:
            self.map_block = nn.Sequential(
                nn.Linear(args.E_layers_channels[-1] * args.lag, args.z_dim),
                nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
                nn.Dropout(p=args.map_block_dropout)
            )
        # self.map_block = nn.Sequential(
        #     nn.Linear(args.lag, args.z_dim),
        #     nn.LeakyReLU(args.LeakyReLU_slope, inplace=True),
        #     nn.Dropout(p=args.map_block_dropout)
        # )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, data):
        out1 = self.spatial_block(data)

        out2 = self.temporal_block(out1)

        out2 = out2.contiguous().view(self.args.batch_size, -1)
        z = self.map_block(out2)

        return z


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        layers = []
        for i in range(len(args.G_layers_channels)):
            in_channels = args.z_dim if i == 0 else args.G_layers_channels[i-1] * args.lag
            out_channels = args.G_layers_channels[i] * args.lag
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=args.map_block_dropout))
        if args.G_layers_channels[-1] != args.node_num:
            layers.append(nn.Linear(args.G_layers_channels[-1] * args.lag, args.node_num * args.lag))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=args.map_block_dropout))
        self.G_layers = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, z):
        x_hat = self.G_layers(z)
        x_hat = x_hat.view(self.args.batch_size, self.args.node_num, self.args.lag)

        return x_hat


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        layers = []
        if args.if_D_GNN_Filter == True:
            self.pre_D_layers = GNN_spatial_block(args)
        for i in range(len(args.D_layers_channels)):
            in_channels = args.node_num if i == 0 else args.D_layers_channels[i-1]
            out_channels = args.D_layers_channels[i]
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=args.map_block_dropout))
        self.D_layers = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, data):
        if self.args.if_D_GNN_Filter:
            x_filter = self.pre_D_layers(data)
            x_filter_2D = x_filter.view(self.args.batch_size * self.args.node_num, self.args.lag)
        else:
            x_filter_2D = data.x.view(self.args.batch_size, self.args.node_num, self.args.lag)

        if self.args.D_focus_on_fre:
            batch_U = data.batch_U
            fre = batch_U @ x_filter_2D
            feature = fre.view(self.args.batch_size, self.args.node_num, self.args.lag)
        else:
            feature = x_filter_2D.view(self.args.batch_size, self.args.node_num, self.args.lag)

        feature = feature.permute(0, 2, 1)
        validity = self.D_layers(feature)
        validity = validity.view(self.args.batch_size, self.args.lag)

        return validity


