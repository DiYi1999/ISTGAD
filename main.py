from argparse import ArgumentParser
import math
from pytorch_lightning.strategies import DDPStrategy
from data.MyDataset import *
from model.mycallback import MyEarlyStopping
from model.MyModel import *
import torch
import pytorch_lightning as L
from pytorch_lightning import Trainer, seed_everything
from data.lightingdata import MyLigDataModule
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.accelerators import find_usable_cuda_devices


if __name__ == '__main__':
    parser = ArgumentParser()

    ### 实验名称
    parser.add_argument('--ray_exp_name', type=str, default='V8.0_BIRDS_10', help='ray实验的名字')

    ### 数据导入
    parser.add_argument('--data_name', type=str, default='BIRDS_10sensor',
                        help='MSL/SMAP/BIRDS_10sensor')
    parser.add_argument('--root_path', type=str, default='/home/data/DATA',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='BIRDS_1U_CubeSat/BIRDS_6535part_10sensor',
                        help='MSL/MSL or SMAP/SMAP or BIRDS_1U_CubeSat/BIRDS_6535part_10sensor')

    ### 路径相关
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0/log',
                        help='检查点缓存路径，同时也是日志保存路径')
    # parser.add_argument('--ray_path', type=str, default='/data1/CODE/Adversarial_Stacked_XGNN2023.4/log',
    #                     help='ray_tune相关checkpoint保存路径')
    parser.add_argument('--result_save_path', type=str, default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0/table',
                        help='调试结果的保存路径')
    parser.add_argument('--result_dirname', type=str, default='V8.0_BIRDS_10', help='调试结果的CSV保存文件名')
    parser.add_argument('--plot_save_path', type=str,
                        default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0/plot',
                        help='画图的保存路径')

    # ### 实验名称
    # parser.add_argument('--ray_exp_name', type=str, default='V8.0_SMAP', help='ray实验的名字')
    #
    # ### 数据导入
    # parser.add_argument('--data_name', type=str, default='SMAP',
    #                     help='MSL/SMAP/BIRDS_10_alllen')
    # parser.add_argument('--root_path', type=str, default='/home/data/DATA',
    #                     help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='SMAP/SMAP',
    #                     help='MSL/MSL or SMAP/SMAP or BIRDS_1U_CubeSat/BIRDS_10_alllen')
    #
    # ### 路径相关
    # parser.add_argument('--ckpt_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_SMAP/log',
    #                     help='检查点缓存路径，同时也是日志保存路径')
    # # parser.add_argument('--ray_path', type=str, default='/data1/CODE/Adversarial_Stacked_XGNN2023.4/log',
    # #                     help='ray_tune相关checkpoint保存路径')
    # parser.add_argument('--result_save_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_SMAP/table',
    #                     help='调试结果的保存路径')
    # parser.add_argument('--result_dirname', type=str, default='V8.0_SMAP', help='调试结果的CSV保存文件名')
    # parser.add_argument('--plot_save_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_SMAP/plot',
    #                     help='画图的保存路径')

    # ### 实验名称
    # parser.add_argument('--ray_exp_name', type=str, default='V8.0_MSL', help='ray实验的名字')
    #
    # ### 数据导入
    # parser.add_argument('--data_name', type=str, default='MSL',
    #                     help='MSL/SMAP/BIRDS_10sensor')
    # parser.add_argument('--root_path', type=str, default='/home/data/DATA',
    #                     help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='MSL/MSL',
    #                     help='MSL/MSL or SMAP/SMAP or BIRDS_1U_CubeSat/BIRDS_6535part_10sensor')
    #
    # ### 路径相关
    # parser.add_argument('--ckpt_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_MSL/log',
    #                     help='检查点缓存路径，同时也是日志保存路径')
    # # parser.add_argument('--ray_path', type=str, default='/data1/CODE/Adversarial_Stacked_XGNN2023.4/log',
    # #                     help='ray_tune相关checkpoint保存路径')
    # parser.add_argument('--result_save_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_MSL/table',
    #                     help='调试结果的保存路径')
    # parser.add_argument('--result_dirname', type=str, default='V8.0_MSL', help='调试结果的CSV保存文件名')
    # parser.add_argument('--plot_save_path', type=str,
    #                     default='/home/data/MyWorks_Results/Adversarial_Stacked_XGNN/V8.0_MSL/plot',
    #                     help='画图的保存路径')

    if parser.parse_known_args()[0].data_name == 'MSL':
        node_num = 55
    elif parser.parse_known_args()[0].data_name == 'SMAP':
        node_num = 25
    elif parser.parse_known_args()[0].data_name == 'BIRDS':
        node_num = 18
    elif parser.parse_known_args()[0].data_name == 'BIRDS_10sensor':
        node_num = 10
    else:
        raise ValueError('node_num is not defined， 请在main.py中定义node_num')

    ### 数据
    parser.add_argument('--Dataset', default=NASA_Anomaly,
                        help='NASA_Anomaly，BIRDS_Anomaly')

    parser.add_argument('--node_num', type=int, default=node_num, help='node_num')
    parser.add_argument('--lag', type=int, default=200)
    parser.add_argument('--missing_rate', type=float, default=0)  #
    parser.add_argument('--missvalue', default=np.nan)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')

    ### 预处理
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--preIDW', type=bool, default=True)
    parser.add_argument('--preMA', type=bool, default=True)
    parser.add_argument('--preMA_win', type=int, default=50)

    ### 图结构
    parser.add_argument('--graph_ca_lag', type=int, default=1000)
    parser.add_argument('--graph_ca_meth', type=str, default='Euc', help='Euc，Cos')
    parser.add_argument('--edge_topK', type=int, default=4)
    parser.add_argument('--eigvalue_if_norm', type=bool, default=True)
    parser.add_argument('--embedding_len', type=int, default=50)


    ### 模型
    # parser.add_argument('--model_name', type=str, default='Adversial_Stacked_Filter_GNN', help='model')
    parser.add_argument('--train_format', type=str, default='GRL', help='onebyone, twopart, GRL')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--LeakyReLU_slope', type=float, default=1e-3)

    ### Encoder
    parser.add_argument('--z_dim', type=int, default=500, help='dim of z')

    # Spatial_block
    parser.add_argument('--spatial_method', type=str, default='Stacked_Filter_GNN')
    parser.add_argument('--plan', type=str, default='eig')
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--residual_alpha', type=float, default=0)

    # Temporal_block
    parser.add_argument('--temporal_method', type=str, default='GRU', help='TCN/dilated_convolution/GRU/MLP')
    parser.add_argument('--E_layers_channels', type=list, default=[8, 4, 1])
    parser.add_argument('--dilated_kernel_size', type=int, default=2)
    parser.add_argument('--GRU_layers', type=int, default=1, help='how many GRU')
    parser.add_argument('--GRU_hidden_num', type=int, default=node_num)

    # map_block
    parser.add_argument('--map_block_dropout', type=float, default=0)

    ### Generator
    parser.add_argument('--G_layers_channels', type=list, default=[8, 10])

    ### Discriminator
    parser.add_argument('--if_D_GNN_Filter', type=bool, default=True)
    parser.add_argument('--D_focus_on_fre', type=bool, default=True)
    parser.add_argument('--D_layers_channels', type=list)

    ### 异常分数计算
    parser.add_argument('--moving_average_window', type=int, default=3)

    ### loss function
    parser.add_argument('--reco_loss_alpha', type=float, default='2.5')
    parser.add_argument('--diff_loss_gama', type=float, default='1.0')
    parser.add_argument('--disc_score_delta', type=float, default='2.0')

    ### 优化器
    parser.add_argument('--optimizer', default=torch.optim.Adam)
    parser.add_argument('--EGD_lr', default=0.001, type=float)
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str,
                        help= 'ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts')
    # parser.add_argument('--Graph_calculate_lr', type=float, default=0.0001)
    # parser.add_argument('--E_lr', default=0.0001, type=float)
    # parser.add_argument('--G_lr', default=0.0004, type=float)
    # parser.add_argument('--D_lr', default=0.00002, type=float)

    ### 训练配置
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--threshold', type=float, default=2.813)
    parser.add_argument('--focus_on', type=str, default='F1', help='F1 pr F1+Pre')

    ### Ray Tune
    # parser.add_argument('--max_trail', type=int, default=500)
    parser.add_argument('--trail_grace_period', type=int, default=1)
    # parser.add_argument('--trail_time_out', type=int, default=600)
    parser.add_argument('--trail_reduction_factor', type=int, default=3)
    parser.add_argument('--grid_num_samples', type=int, default=-1)

    args = parser.parse_args()



def light_main(config, args):
    if config is not None:
        for k, v in config.items():
            setattr(args, k, v)

    print("请确认节点数目即通道数是否是：", args.node_num)

    datamodule = MyLigDataModule(args)
    if args.train_format == 'GRL':
        model = MyModel_GRL(args)
    # elif args.train_format == 'twopart':
    #     model = MyModel_twopart(args)
    # elif args.train_format == 'onebyone':
    #     model = MyModel_onebyone(args)

    trainer = Trainer(# strategy="auto",
                      # devices="auto",
                      max_epochs=args.max_epoch,
                      enable_progress_bar=False,
                      logger=TensorBoardLogger(save_dir=args.ckpt_path, name="lightning_logs"),
                      callbacks=[
                          TuneReportCallback(
                              {'loss': "early_stop_loss", 'F1': 'test_F1', 'precision': 'test_precision',
                               'recall': 'test_recall', 'AUC': 'test_AUC', 'accuracy': 'test_accuracy',
                               'mse': 'test_mse'}, on="test_end"),
                          EarlyStopping(monitor="early_stop_loss", patience=args.patience, check_on_train_epoch_end=True,
                                        mode="min")
                      ],
                      deterministic="warn",
                      )

    seed_everything(args.random_seed, workers=True)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    trainer.predict(model, datamodule=datamodule)


def ray_tune_run(args, config, everyexp_gpu, exp_num):
    config = config

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=args.max_epoch,
        grace_period=args.trail_grace_period,
        reduction_factor=args.trail_reduction_factor)

    algo = None

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=['loss', 'F1', 'precision', 'recall', 'AUC', 'accuracy', 'mse'])

    trainable_func = tune.with_parameters(light_main, args=args)

    resources_per_trial = {"cpu": 96//exp_num, "gpu": everyexp_gpu}
    trainable_func = tune.with_resources(trainable_func, resources_per_trial)

    tuner = tune.Tuner(
        trainable_func,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=args.grid_num_samples,
            max_concurrent_trials = 0,
            reuse_actors=False,
        ),
        run_config=air.RunConfig(
            name=args.ray_exp_name,
            local_dir=args.ckpt_path+'/'+'ray_result',
            progress_reporter=reporter,
        ),
        param_space=config,
    )

    if os.path.exists(args.plot_save_path):
        tuner = tune.Tuner.restore(
            path=args.ckpt_path+'/'+'ray_result'+'/'+args.ray_exp_name,
            trainable=trainable_func,
            resume_unfinished=True,
            resume_errored=True,
            param_space=config,
        )

    result = tuner.fit()
    print("Best hyperparameters found were: ", result.get_best_result().config)
    return result






BIRDS_simple_config = {
    "Dataset": tune.choice([BIRDS_Anomaly]),
    "lag": tune.choice([200, 300, 500, 800]),
    # "missing_rate": tune.choice([0, 0.1, 0.2, 0.3]),
    "missing_rate": tune.choice([0, 0.3]),
    "preMA_win": tune.choice([50]),
    "graph_ca_meth": tune.choice(['Euc']),
    # "eigvalue_if_norm": tune.choice([True, False]),
    "eigvalue_if_norm": tune.choice([True]),
    # "graph_ca_lag": tune.choice([2000, 5000]),
    # "edge_topK": tune.choice([4, 5, 6, 7, 8]),
    "edge_topK": tune.choice([4, 5, 8]),
    # "embedding_len": tune.choice([30, 50, 100, 150, 200, 300, 500]),
    # "train_format": tune.choice(['GRL', 'twopart', 'onebyone']),
    # "dropout": tune.choice([0, 0.2]),
    # "z_dim": tune.choice([50, 100, 150, 200, 300, 500]),
    "z_dim": tune.choice([50, 150, 300, 500]),
    # "spatial_method": tune.choice(['Stacked_Filter_GNN', 'BernNet', 'GPRNet', 'ARMANet', 'ChebNet']),
    # "spatial_method": tune.choice(['Stacked_Filter_GNN']),
    "K": tune.choice([3, 4, 5, 8, 10, 15, 20]),
    # "layers": tune.choice([1, 2]),
    # "residual_alpha": tune.choice([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    # 这里上面就0就行，因为滤波器自己会有个常数项
    # "temporal_method": tune.choice(['TCN', 'dilated_convolution', 'GRU', 'MLP']),
    "temporal_method": tune.choice(['GRU']),
    # "E_layers_channels": tune.choice([[8, 4, 1], [10, 8, 4, 1], [8, 1], [1]]),
    "E_layers_channels": tune.choice([[10, 1], [8, 1], [1]]),
    # "dilated_kernel_size": tune.choice([2, 5, 8]),
    "dilated_kernel_size": tune.choice([2]),
    # "map_block_dropout": tune.choice([0, 0.2]),
    # "G_layers_channels": tune.choice([[5, 8, 10], [8, 10], [10]]),
    "G_layers_channels": tune.choice([[8, 10], [10]]),
    # "if_D_GNN_Filter": tune.choice([True, False]),
    "D_focus_on_fre": tune.choice([True, False]),
    # "D_focus_on_fre": tune.choice([True]),
    # "D_layers_channels": tune.choice([[10, 8, 1], [8, 1], [1]]),
    "D_layers_channels": tune.choice([[8, 1], [1]]),
    # "moving_average_window": tune.choice([3, 5]),
    "reco_loss_alpha": tune.quniform(0, 2, 0.2),
    # "disc_loss_beta": tune.choice([1, 0.5, 1.5, 2]),
    "diff_loss_gama": tune.choice([0.2, 0.5, 1, 2]),
    "disc_score_delta": tune.quniform(0, 2, 0.2),

    # "EGD_lr": tune.loguniform(1e-5, 1e-1),
    # "EGD_lr": tune.qloguniform(1e-5, 1e-2, 5e-6),
    "EGD_lr": tune.choice([0.001, 0.0001]),
    "scheduler": tune.choice(['ReduceLROnPlateau', 'CosineAnnealingLR', 'StepLR', 'ExponentialLR']),

    "batch_size": tune.choice([32, 64, 128]),
    # "patience": tune.choice([5, 10, 20]),
    # "max_epoch": tune.choice([100, 200, 500]),
    # "focus_on": tune.choice(['F1', 'F1+Pre']),
    "threshold": tune.quniform(2.8, 2.9, 0.00000001),
    "focus_on": tune.choice(['F1']),

    # "trail_grace_period": tune.choice([1, 2, 3]),
    # "trail_reduction_factor": tune.choice([2, 3, 4]),
    # "grid_num_samples": tune.choice([1, -1, 20]),
}

devices = "2,3"
# devices = "0,1,2"
# devices = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = devices
all_exp_num = 2

ray_tune_run(args, BIRDS_simple_config, everyexp_gpu=(math.ceil(len(devices)/2)/all_exp_num), exp_num=all_exp_num)















