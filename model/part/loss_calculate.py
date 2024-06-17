import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Parameter



class Loss_calculate(nn.Module):
    def __init__(self, args):
        super(Loss_calculate, self).__init__()
        self.args = args
        self.reco_loss_median = None
        self.reco_loss_iqr = None
        self.disc_loss_median = None
        self.disc_loss_iqr = None


    def GRL_EGD_loss(self, x, x_fake_from_real_z, validity_fake1, validity_real1, validity_fake2, validity_real2):
        x = x.view(validity_real1.shape[0], -1, x.shape[1])
        x_fake_from_real_z = x_fake_from_real_z.view(validity_real1.shape[0], -1, x_fake_from_real_z.shape[1])
        reco_loss = torch.abs(x - x_fake_from_real_z)
        reco_loss = torch.mean(reco_loss, dim=(1, 2))
        reco_loss = torch.mean(reco_loss)
        disc_loss1 = self.onebyone_D_loss(validity_fake1, validity_real1)
        disc_loss2 = self.onebyone_D_loss(validity_fake2, validity_real2)
        EGD_loss = self.args.reco_loss_alpha * reco_loss \
                   + disc_loss1 + disc_loss2
        return EGD_loss


    def twopart_D_loss(self, validity_fake1, validity_real1, validity_fake2, validity_real2):
        D_loss = self.onebyone_D_loss(validity_fake1, validity_real1) \
                 + self.onebyone_D_loss(validity_fake2, validity_real2)
        return D_loss

    def twopart_EG_loss(self, x, x_fake_from_real_z, validity_fake, validity_real):
        EG_loss = self.onebyone_E_loss(x, x_fake_from_real_z, validity_fake, validity_real)
        return EG_loss


    def onebyone_D_loss(self, validity_fake, validity_real):
        D_loss = - torch.abs(torch.mean(validity_real, 1) - torch.mean(validity_fake, 1)) \
                 + 0.5 * self.args.diff_loss_gama * F.relu(torch.mean(validity_real, 1) - torch.mean(validity_fake, 1))
        D_loss = torch.mean(D_loss)
        return D_loss

    def onebyone_E_loss(self, x, x_fake_from_real_z, validity_fake, validity_real):
        x = x.view(validity_real.shape[0], -1, x.shape[1])
        x_fake_from_real_z = x_fake_from_real_z.view(validity_real.shape[0], -1, x_fake_from_real_z.shape[1])
        reco_loss = torch.abs(x - x_fake_from_real_z)
        reco_loss = torch.mean(reco_loss, dim=(1, 2))
        reco_loss = torch.mean(reco_loss)
        E_loss = self.args.reco_loss_alpha * reco_loss - self.onebyone_D_loss(validity_fake, validity_real)
        return E_loss

    def onebyone_G_loss(self, x, x_fake_from_real_z, validity_fake, validity_real):
        G_loss = self.onebyone_E_loss(x, x_fake_from_real_z, validity_fake, validity_real)
        return G_loss