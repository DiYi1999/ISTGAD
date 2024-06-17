import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np


### TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, padding_mode='replicate', dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, padding_mode='replicate', dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_temporal_block(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_temporal_block, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        # (batch, node_num, lag) -> (batch, E_layers_channels[-1], lag)

        return out


### 膨胀卷积
class dilated_temporal_block(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(dilated_temporal_block, self).__init__()
        self.input_size = input_size
        # self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Create a list of convolutional layers with dilation
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation_size,
                                 padding='same',
                                 padding_mode='replicate'),
                       nn.LeakyReLU(0.01, inplace=True),
                       nn.Dropout(p=dropout)]

        self.network = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        # Pass the input through the convolutional layers
        out = self.network(x)
        # (batch, node_num, lag) -> (batch, E_layers_channels[-1], lag)

        return out


class GRU_temporal_block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        """
        GRU

        Args:
            input_size:
            hidden_size:
            num_layers:
            dropout:
        """
        super(GRU_temporal_block, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # (batch, node_num, lag) -> (batch, lag, node_num)

        # Forward propagate RNN
        out, _ = self.gru(x)
        # (batch, lag, hidden_size)

        out = out.permute(0, 2, 1)
        # (batch, lag, hidden_size) -> (batch, hidden_size, lag)

        return out


### MLP
class MLP_temporal_block(nn.Module):
    def __init__(self, input_size, num_channels, dropout=0.2):
        """
        普通几层MLP+全连接层+dropout完成时间信息提取

        Args:
            input_size:
            num_channels:
            dropout:The probability of dropping out a neuron in the dropout layer.default=0.2
        """
        super(MLP_temporal_block, self).__init__()
        self.input_size = input_size
        # self.output_size = output_size
        self.num_channels = num_channels
        self.dropout = dropout

        # Create a list of convolutional layers with dilation
        layers = []
        for i in range(len(num_channels)):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Linear(in_channels, out_channels),
                       nn.LeakyReLU(0.001, inplace=True),
                       nn.Dropout(p=dropout)]
        self.network = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = x
        # (batch, node_num, lag)
        x = x.permute(0, 2, 1)
        # (batch, lag, node_num)
        out = self.network(x)
        # (batch, lag, E_layers_channels[-1])
        out = out.permute(0, 2, 1)
        # (batch, E_layers_channels[-1], lag)

        return out
