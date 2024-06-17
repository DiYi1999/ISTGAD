from torch import nn
import torch


class anomaly_score(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x_real, x_reco, validity_real):
        x_real = x_real.view(self.args.batch_size, self.args.node_num, self.args.lag)
        x_reco = x_reco.view(self.args.batch_size, self.args.node_num, self.args.lag)
        reco_score = torch.abs(x_real - x_reco)

        disc_score = torch.mean(validity_real, dim=1)
        disc_score = torch.unsqueeze(disc_score, 1)
        disc_score = torch.unsqueeze(disc_score, 2)
        disc_score = disc_score.repeat(1, reco_score.shape[1], reco_score.shape[2])

        return reco_score, disc_score

