import pytorch_lightning as pl
from data.MyDataset import *
import torch
from torch.utils.data import DataLoader
# from pytorch_lightning import LightningDataModule
import os
import time
import warnings
warnings.filterwarnings('ignore')


class MyLigDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MyLigDataModule, self).__init__()
        self.data_name = args.data_name
        self.shuffle_flag = True
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.drop_last = True
        self.args = args
        self.ready_dataset_module()

    def ready_dataset_module(self):
        self.DataSet = self.args.Dataset

    def train_dataloader(self):
        args = self.args
        data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.missing_rate,
            missvalue=args.missvalue,
            flag='train',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale
        )
        print('train', len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def val_dataloader(self):
        args = self.args
        data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.miss_rate,
            missvalue=args.missvalue,
            flag='val',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale
        )
        print('val', len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def test_dataloader(self):
        self.shuffle_flag = False;
        args = self.args
        data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.missing_rate,
            missvalue=args.missvalue,
            flag='test',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale
        )
        print('test', len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def predict_dataloader(self):
        data_loader = self.test_dataloader()
        return data_loader
