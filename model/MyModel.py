import os
import pandas as pd
import pytorch_lightning as L
from utils.plot import my_plot
from model.part.E_G_D import Encoder, Generator, Discriminator
from data.graph_calculate import Graph_calculate
from model.part.Gradient_Reversal_Layer import GRL
from model.part.loss_calculate import Loss_calculate
from torch_geometric.data import Data
from utils.score import anomaly_score
from utils.evaluate import *


class MyModel_GRL(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.Graph_calculate = Graph_calculate(args)
        self.edge_index = torch.Tensor()
        self.pyG_batch_edge_index = torch.Tensor()
        self.batch_L = torch.Tensor()
        self.batch_Λ = torch.Tensor()
        self.batch_U = torch.Tensor()

        self.Loss_calculate = Loss_calculate(args)
        self.E = Encoder(args)
        self.G = Generator(args)
        self.D = Discriminator(args)
        self.GRL = GRL()
        self.anomaly_score = anomaly_score(args)

        self.early_stop_loss = 0

        self.z_tensor = torch.Tensor()

        self.E_residual_alpha = torch.Tensor()
        self.E_a_list_final = torch.Tensor()
        self.E_b_list_final = torch.Tensor()
        self.E_c_list_final = torch.Tensor()

        self.D_residual_alpha = torch.Tensor()
        self.D_a_list_final = torch.Tensor()
        self.D_b_list_final = torch.Tensor()
        self.D_c_list_final = torch.Tensor()

        self.orig_tensor = torch.Tensor()
        self.missed_orig_tensor = torch.Tensor()
        self.fixmiss_orig_tensor = torch.Tensor()

        self.reco_tensor = torch.Tensor()
        self.missed_reco_tensor = torch.Tensor()

        self.reco_anomaly_score_tensor = torch.Tensor()
        self.disc_anomaly_score_tensor = torch.Tensor()
        self.anomaly_score_tensor = torch.Tensor()
        self.anomaly_detect_tensor = torch.Tensor()
        self.anomaly_label_tensor = torch.Tensor()
        self.anomaly_label_vector = torch.Tensor()

        self.exam_result = {}

        self.configure_optimizers()

    def configure_optimizers(self):
        optimizer_EGD = self.args.optimizer([
            {'params': self.G.parameters(), 'lr': self.args.EGD_lr},
            {'params': self.E.parameters(), 'lr': self.args.EGD_lr},
            {'params': self.D.parameters(), 'lr': self.args.EGD_lr}
        ])
        if self.args.scheduler == 'ReduceLROnPlateau':
            scheduler_EGD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_EGD,
                                                                       mode='min', factor=0.5,
                                                                       patience=5, verbose=True,
                                                                       threshold=0.0001, threshold_mode='rel',
                                                                       cooldown=0, min_lr=0,
                                                                       eps=1e-07)
        elif self.args.scheduler == 'StepLR':
            scheduler_EGD = torch.optim.lr_scheduler.StepLR(optimizer_EGD, step_size=10, gamma=0.5)
        elif self.args.scheduler == 'ExponentialLR':
            scheduler_EGD = torch.optim.lr_scheduler.ExponentialLR(optimizer_EGD, gamma=0.9)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler_EGD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_EGD, T_max=10, eta_min=0)
        elif self.args.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler_EGD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_EGD, T_0=10, T_mult=1, eta_min=0)
        else:
            scheduler_EGD = None

        return [optimizer_EGD], [scheduler_EGD]

    def training_step(self, batch, batch_idx):
        missed_batch, mask_batch, datetime_batch, full_batch = batch
        # (batch_size, lag, node_num), (batch_size, lag, node_num), (batch_size, lag, 6), (batch_size, lag, node_num)

        full_x_3D = missed_batch
        # (batch_size, lag, node_num)

        full_data, edge_index, _, _, _ = self.Graph_calculate(full_x_3D, self.edge_index)
        self.edge_index = edge_index
        self.pyG_batch_edge_index = full_data.edge_index
        self.batch_L = full_data.batch_L
        self.batch_Λ = full_data.batch_Λ
        self.batch_U = full_data.batch_U

        optimizer_EGD = self.optimizers()

        self.toggle_optimizer(optimizer_EGD)
        optimizer_EGD.zero_grad()

        z_random = torch.randn(self.args.batch_size, self.args.z_dim)
        z_random = z_random.type_as(full_data.x)
        x_fake_from_random_z = self.G(z_random)
        x_fake_from_random_z = x_fake_from_random_z.contiguous().view(self.args.batch_size * self.args.node_num, self.args.lag)

        x_fake_from_random_z_GRL = self.GRL(x_fake_from_random_z)
        data_fake_from_random_z_GRL = Data(x=x_fake_from_random_z_GRL, edge_index=full_data.edge_index, batch_L=full_data.batch_L, batch_Λ=full_data.batch_Λ, batch_U=full_data.batch_U)

        full_x_2D_GRL = self.GRL(full_data.x)
        full_data_GRL = Data(x=full_x_2D_GRL, edge_index=full_data.edge_index, batch_L=full_data.batch_L, batch_Λ=full_data.batch_Λ, batch_U=full_data.batch_U)

        validity_fake1 = self.D(data_fake_from_random_z_GRL)
        validity_real1 = self.D(full_data_GRL)



        z_real = self.E(full_data)
        x_fake_from_real_z = self.G(z_real)
        x_fake_from_real_z = x_fake_from_real_z.contiguous().view(self.args.batch_size * self.args.node_num, self.args.lag)

        x_fake_from_real_z_GRL = self.GRL(x_fake_from_real_z)
        data_fake_from_real_z_GRL = Data(x=x_fake_from_real_z_GRL, edge_index=full_data.edge_index, batch_L=full_data.batch_L, batch_Λ=full_data.batch_Λ, batch_U=full_data.batch_U)

        validity_fake2 = self.D(data_fake_from_real_z_GRL)
        validity_real2 = validity_real1

        EGD_loss = self.Loss_calculate.GRL_EGD_loss(full_data.x, x_fake_from_real_z, validity_fake1, validity_real1, validity_fake2, validity_real2)

        self.manual_backward(EGD_loss)
        optimizer_EGD.step()
        self.untoggle_optimizer(optimizer_EGD)

        self.E_residual_alpha = \
            self.E.state_dict()['spatial_block.Spatial_Net.Stacked_Filter.residual_alpha'].cpu().detach().tolist()
        self.E_a_list_final = \
            self.E.state_dict()['spatial_block.Spatial_Net.Stacked_Filter.a_list'].cpu().detach().tolist()
        self.E_b_list_final = \
            self.E.state_dict()['spatial_block.Spatial_Net.Stacked_Filter.b_list'].cpu().detach().tolist()
        self.E_c_list_final = \
            self.E.state_dict()['spatial_block.Spatial_Net.Stacked_Filter.c_list'].cpu().detach().tolist()

        self.D_residual_alpha = \
            self.D.state_dict()['pre_D_layers.Spatial_Net.Stacked_Filter.residual_alpha'].cpu().detach().tolist()
        self.D_a_list_final = \
            self.D.state_dict()['pre_D_layers.Spatial_Net.Stacked_Filter.a_list'].cpu().detach().tolist()
        self.D_b_list_final = \
            self.D.state_dict()['pre_D_layers.Spatial_Net.Stacked_Filter.b_list'].cpu().detach().tolist()
        self.D_c_list_final = \
            self.D.state_dict()['pre_D_layers.Spatial_Net.Stacked_Filter.c_list'].cpu().detach().tolist()

        self.log("early_stop_loss", EGD_loss, prog_bar=True)
        self.early_stop_loss = EGD_loss
        return EGD_loss


    def on_train_epoch_end(self) -> None:
        scheduler_EGD = self.lr_schedulers()

        if self.args.scheduler == 'ReduceLROnPlateau':
            scheduler_EGD.step(self.trainer.callback_metrics["early_stop_loss"])
        else:
            scheduler_EGD.step()


    def forward(self,data):
        z_real = self.E(data)
        x_fake_from_real_z = self.G(z_real)
        x_fake_from_real_z = x_fake_from_real_z.contiguous().view(self.args.batch_size * self.args.node_num, self.args.lag)
        data_fake_from_real_z = Data(x=x_fake_from_real_z, edge_index=data.edge_index, batch_L=data.batch_L,
                                     batch_Λ=data.batch_Λ, batch_U=data.batch_U)
        validity_real = self.D(data)

        reco_anomaly_score, disc_anomaly_score = self.anomaly_score(data.x, x_fake_from_real_z, validity_real)
        x_fake_from_real_z = x_fake_from_real_z.view(self.args.batch_size, self.args.node_num, self.args.lag)
        return reco_anomaly_score, disc_anomaly_score, x_fake_from_real_z, data_fake_from_real_z, z_real


    def test_step(self, batch, batch_idx):
        missed_batch, mask_batch, datetime_batch, full_batch, anomaly_label, anomaly_all_label = batch
        self.anomaly_label_vector = anomaly_label[0]
        self.anomaly_label_tensor = anomaly_all_label[0]

        full_x_3D = missed_batch.permute(0, 2, 1)
        full_x_2D = full_x_3D.contiguous().view(self.args.batch_size*self.args.node_num, self.args.lag)
        full_data = Data(x=full_x_2D, edge_index=self.pyG_batch_edge_index, batch_L=self.batch_L, batch_Λ=self.batch_Λ, batch_U=self.batch_U)

        reco_anomaly_score, disc_anomaly_score, reco_x, reco_data, z_real = self.forward(full_data)


        orig_batch_tensor = full_batch.type_as(full_x_3D)
        missed_batch_tensor = missed_batch.type_as(full_x_3D)
        fixmiss_batch_tensor = full_x_3D.permute(0, 2, 1)
        reco_batch_tensor = reco_x.permute(0, 2, 1)
        missreco_batch_tensor = torch.mul(reco_batch_tensor, mask_batch)
        reco_anomaly_score_batch_tensor = reco_anomaly_score.permute(0, 2, 1)
        disc_anomaly_score_batch_tensor = disc_anomaly_score.permute(0, 2, 1)
        z_batch_tensor = z_real

        if batch_idx == 0:
            self.orig_tensor = orig_batch_tensor[0, :, :]
            self.missed_orig_tensor = missed_batch_tensor[0, :, :]
            self.fixmiss_orig_tensor = fixmiss_batch_tensor[0, :, :]
            self.reco_tensor = reco_batch_tensor[0, :, :]
            self.missed_reco_tensor = missreco_batch_tensor[0, :, :]
            self.reco_anomaly_score_tensor = reco_anomaly_score_batch_tensor[0, :, :]
            self.disc_anomaly_score_tensor = disc_anomaly_score_batch_tensor[0, :, :]
            self.z_tensor = z_batch_tensor[0, :].unsqueeze(0)
            for n in range(1, self.args.batch_size):
                self.orig_tensor = torch.cat((self.orig_tensor[:self.orig_tensor.shape[0]-orig_batch_tensor.shape[1]+1, :]
                                              , orig_batch_tensor[n, :, :]), dim=0)
                self.missed_orig_tensor = torch.cat((self.missed_orig_tensor[:self.missed_orig_tensor.shape[0]-missed_batch_tensor.shape[1]+1, :]
                                                     , missed_batch_tensor[n, :, :]), dim=0)
                self.fixmiss_orig_tensor = torch.cat((self.fixmiss_orig_tensor[:self.fixmiss_orig_tensor.shape[0]-fixmiss_batch_tensor.shape[1]+1, :]
                                                      , fixmiss_batch_tensor[n, :, :]), dim=0)
                self.reco_tensor = torch.cat((self.reco_tensor[:self.reco_tensor.shape[0]-reco_batch_tensor.shape[1]+1, :]
                                              , reco_batch_tensor[n, :, :]), dim=0)
                self.missed_reco_tensor = torch.cat((self.missed_reco_tensor[:self.missed_reco_tensor.shape[0]-missreco_batch_tensor.shape[1]+1, :]
                                                     , missreco_batch_tensor[n, :, :]), dim=0)
                self.reco_anomaly_score_tensor = torch.cat((self.reco_anomaly_score_tensor[:self.reco_anomaly_score_tensor.shape[0]-reco_anomaly_score_batch_tensor.shape[1]+1, :]
                                                            , self.reco_anomaly_score_tensor[self.reco_anomaly_score_tensor.shape[0]-reco_anomaly_score_batch_tensor.shape[1]+1:, :]
                                                            + reco_anomaly_score_batch_tensor[n, :-1, :]
                                                            , reco_anomaly_score_batch_tensor[n, -1, :].view(1, -1)), dim=0)
                self.disc_anomaly_score_tensor = torch.cat((self.disc_anomaly_score_tensor[:self.disc_anomaly_score_tensor.shape[0]-disc_anomaly_score_batch_tensor.shape[1]+1, :]
                                                            , self.disc_anomaly_score_tensor[self.disc_anomaly_score_tensor.shape[0]-disc_anomaly_score_batch_tensor.shape[1]+1:, :]
                                                            + disc_anomaly_score_batch_tensor[n, :-1, :]
                                                            , disc_anomaly_score_batch_tensor[n, -1, :].view(1, -1)), dim=0)
                self.z_tensor = torch.cat((self.z_tensor, z_batch_tensor[n].unsqueeze(0)), dim=0)
        else:
            for n in range(self.args.batch_size):
                self.orig_tensor = torch.cat((self.orig_tensor[:self.orig_tensor.shape[0]-orig_batch_tensor.shape[1]+1, :]
                                              , orig_batch_tensor[n, :, :]), dim=0)
                self.missed_orig_tensor = torch.cat((self.missed_orig_tensor[:self.missed_orig_tensor.shape[0]-missed_batch_tensor.shape[1]+1, :]
                                                     , missed_batch_tensor[n, :, :]), dim=0)
                self.fixmiss_orig_tensor = torch.cat((self.fixmiss_orig_tensor[:self.fixmiss_orig_tensor.shape[0]-fixmiss_batch_tensor.shape[1]+1, :]
                                                      , fixmiss_batch_tensor[n, :, :]), dim=0)
                self.reco_tensor = torch.cat((self.reco_tensor[:self.reco_tensor.shape[0]-reco_batch_tensor.shape[1]+1, :]
                                              , reco_batch_tensor[n, :, :]), dim=0)
                self.missed_reco_tensor = torch.cat((self.missed_reco_tensor[:self.missed_reco_tensor.shape[0]-missreco_batch_tensor.shape[1]+1, :]
                                                     , missreco_batch_tensor[n, :, :]), dim=0)
                self.reco_anomaly_score_tensor = torch.cat((self.reco_anomaly_score_tensor[:self.reco_anomaly_score_tensor.shape[0]-reco_anomaly_score_batch_tensor.shape[1]+1, :]
                                                            , self.reco_anomaly_score_tensor[self.reco_anomaly_score_tensor.shape[0]-reco_anomaly_score_batch_tensor.shape[1]+1:, :]
                                                            + reco_anomaly_score_batch_tensor[n, :-1, :]
                                                            , reco_anomaly_score_batch_tensor[n, -1, :].view(1, -1)), dim=0)
                self.disc_anomaly_score_tensor = torch.cat((self.disc_anomaly_score_tensor[:self.disc_anomaly_score_tensor.shape[0]-disc_anomaly_score_batch_tensor.shape[1]+1, :]
                                                            , self.disc_anomaly_score_tensor[self.disc_anomaly_score_tensor.shape[0]-disc_anomaly_score_batch_tensor.shape[1]+1:, :]
                                                            + disc_anomaly_score_batch_tensor[n, :-1, :]
                                                            , disc_anomaly_score_batch_tensor[n, -1, :].view(1, -1)), dim=0)
                self.z_tensor = torch.cat((self.z_tensor, z_batch_tensor[n].unsqueeze(0)), dim=0)

        return self.reco_anomaly_score_tensor, self.disc_anomaly_score_tensor, self.z_tensor


    def on_test_epoch_end(self):
        self.orig_tensor = self.orig_tensor[self.args.lag:-self.args.lag, :]
        self.missed_orig_tensor = self.missed_orig_tensor[self.args.lag:-self.args.lag, :]
        self.fixmiss_orig_tensor = self.fixmiss_orig_tensor[self.args.lag:-self.args.lag, :]

        self.reco_tensor = self.reco_tensor[self.args.lag:-self.args.lag, :]
        self.missed_reco_tensor = self.missed_reco_tensor[self.args.lag:-self.args.lag, :]

        self.reco_anomaly_score_tensor = self.reco_anomaly_score_tensor[self.args.lag:-self.args.lag, :]
        self.disc_anomaly_score_tensor = self.disc_anomaly_score_tensor[self.args.lag:-self.args.lag, :]

        self.reco_anomaly_score_tensor = all_normalize(self.reco_anomaly_score_tensor)
        self.disc_anomaly_score_tensor = all_normalize(self.disc_anomaly_score_tensor)

        self.anomaly_score_tensor = self.reco_anomaly_score_tensor + self.args.disc_score_delta * self.disc_anomaly_score_tensor

        self.anomaly_score_tensor = moving_average(self.anomaly_score_tensor, self.args.moving_average_window)

        self.z_tensor = self.z_tensor[self.args.lag:-self.args.lag, :]

        self.anomaly_label_tensor = self.anomaly_label_tensor[self.args.lag:self.args.lag+self.anomaly_score_tensor.shape[0], :]
        self.anomaly_label_vector = self.anomaly_label_vector[self.args.lag:self.args.lag+self.anomaly_score_tensor.shape[0]]

        top1_bestF1_result = get_best_performance_data(self.anomaly_score_tensor.cpu().numpy(),
                                                       self.anomaly_label_vector.cpu().numpy(),
                                                       topk=1, focus_on=self.args.focus_on,
                                                       thresold=self.args.threshold)
        if self.args.focus_on == 'F1':
            print(f'F1 score: {top1_bestF1_result[0]}')
        else:
            print(f'F1 score: {top1_bestF1_result[0] - top1_bestF1_result[2]}')
        print(f'accuracy: {top1_bestF1_result[1]}')
        print(f'precision: {top1_bestF1_result[2]}')
        print(f'recall: {top1_bestF1_result[3]}')
        print(f'AUC: {top1_bestF1_result[4]}')
        print(f'threshold: {top1_bestF1_result[5]}')
        mae, mse, rmse, mape, mspe = metric(self.orig_tensor.cpu().numpy(), self.reco_tensor.cpu().numpy())
        print('mse:{}, mae:{}'.format(mse, mae))

        self.exam_result = {'F1': top1_bestF1_result[0],
                            'precision': top1_bestF1_result[2],
                            'recall': top1_bestF1_result[3],
                            'AUC': top1_bestF1_result[4],
                            'accuracy': top1_bestF1_result[1],
                            'threshold': str(top1_bestF1_result[5]),
                            'mse': mse,
                            'mae': mae
                            }


        self.log("early_stop_loss", self.early_stop_loss, prog_bar=True)
        self.log('test_F1', top1_bestF1_result[0], prog_bar=True)
        self.log('test_precision', top1_bestF1_result[2], prog_bar=True)
        self.log('test_recall', top1_bestF1_result[3], prog_bar=True)
        self.log('test_AUC', top1_bestF1_result[4], prog_bar=True)
        self.log('test_accuracy', top1_bestF1_result[1], prog_bar=True)
        self.log('test_mse', mse, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)

        dirname_path = self.args.result_save_path+'/'
        if not os.path.exists(dirname_path + self.args.result_dirname + '.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_dict = vars(self.args)
            args_dict = {k: str(v) for k, v in args_dict.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.result_save_path + '/' + self.args.result_dirname+'.csv', index=False, mode='a', header=True)

        else:
            args_dict = vars(self.args)
            args_dict = {k: str(v) for k, v in args_dict.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.result_save_path + '/' + self.args.result_dirname+'.csv', index=False, mode='a', header=False)


        self.anomaly_detect_tensor = torch.where(torch.tensor(self.anomaly_score_tensor >
                                                              torch.tensor([top1_bestF1_result[5]]).to(
                                                                  self.anomaly_score_tensor.device)),
                                                 torch.tensor([1]).to(self.anomaly_score_tensor.device),
                                                 torch.tensor([0]).to(self.anomaly_score_tensor.device))
        freq_tensor = self.batch_U[:self.args.node_num, :self.args.node_num].permute(1, 0) \
                      @ (self.orig_tensor.permute(1, 0))
        freq_eig_vector = torch.diag(self.batch_Λ[:self.args.node_num, :self.args.node_num])
        my_plot(self.args,
                self.orig_tensor, self.reco_tensor, self.anomaly_label_tensor,
                self.anomaly_detect_tensor, self.anomaly_score_tensor, self.anomaly_label_vector,
                self.E_residual_alpha, self.E_a_list_final, self.E_b_list_final, self.E_c_list_final,
                self.D_residual_alpha, self.D_a_list_final, self.D_b_list_final, self.D_c_list_final,
                freq_tensor, freq_eig_vector,
                self.z_tensor,
                self.exam_result, args_dict)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            print('this experiment finished')


