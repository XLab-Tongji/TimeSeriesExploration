import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl 
import json

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.wrappers.datamodule import SpatioTemporalDataModule
from dataset.wrappers.imputation_dataset import GraphImputationDataset,ImputationDataset
from dataset.bay_dataset import PemsBay,MissingValuesPemsBay
from dataset.la_dataset import MetrLA,MissingValuesMetrLA
from config.config_bay import Config_bay

from model.gril_net import GRILNet
from model.utils.metrics import *
from model.utils.metric_base import MaskedMetric,Metric
from model.network import GrilPlNetwork

from utils.parser_utils import str_to_bool
from utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe,get_args_from_json


# default_conf=Config_bay()
# conf=default_conf.get_config()
current_path=os.path.abspath(__file__) 
parent_path=os.path.abspath(os.path.dirname(current_path))

def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--default_conf', type=str, default='config/pems_bay.json')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--model-name", type=str, default='gril')
    parser.add_argument("--dataset-name", type=str, default='pems-bay')
    parser.add_argument('--logdir', type=str, default='logs')
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)


    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.default_conf is not None:
        with open(os.path.join(parent_path,args.default_conf),'r') as f:
            conf=json.load(f)
        for arg in conf:
            setattr(args, arg, conf[arg])

    return args

def dataset_loader(args):
    if args.dataset_name=='pemsbay':
        raw_dataset=MissingValuesPemsBay()
    elif args.dataset_name=='metr-la':
        raw_dataset=MissingValuesMetrLA()
    elif args.dataset_name=='pemsbay-noise':
        raw_dataset=MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif args.dataset_name=='pemsbay-noise':
        raw_dataset=MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    # get adjacency matrix
    adj = raw_dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    dataset=GraphImputationDataset(*raw_dataset.numpy(return_idx=True),
                                    mask=raw_dataset.training_mask,
                                    eval_mask=raw_dataset.eval_mask,
                                    window=args.window,
                                    stride=args.stride)
    split_conf = parser_utils.filter_function_args(args, raw_dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = raw_dataset.splitter(dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dataset_module = SpatioTemporalDataModule(dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dataset_module.setup()
    return dataset_module,adj

def model_loader(dataset,adj,args):
    if args.model_name=="gril":
        model_cls=GRILNet
    #basic net config
    model_hparams = dict(adj=adj, d_in=dataset.d_in, n_nodes=dataset.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)
    #loss function settings
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})
    #metric settings
    metrics=dict(
                mae=MaskedMAE(compute_on_step=False),
                mape=MaskedMAPE(compute_on_step=False),
                mse=MaskedMSE(compute_on_step=False),
                mre=MaskedMRE(compute_on_step=False)
                )
    scheduler_cls=CosineAnnealingLR
    #settings for pl module
    pl_network_hparams=dict(model_class=model_cls,
                            model_kwargs=model_kwargs,
                            optim_class=torch.optim.Adam,
                            optim_kwargs={'lr': args.lr,
                                          'weight_decay': args.l2_reg},
                            loss_fn=loss_fn,
                            metrics=metrics,
                            scheduler_class=scheduler_cls,
                            scheduler_kwargs={
                            'eta_min': 0.0001,
                            'T_max': args.epochs},
                            alpha=args.alpha,
                            hint_rate=args.hint_rate,
                            g_train_freq=args.g_train_freq,
                            d_train_freq=args.d_train_freq)
    
    pl_kwargs=parser_utils.filter_args(
                                    args={**vars(args), **pl_network_hparams},
                                    target_cls=GrilPlNetwork,
                                    return_dict=True)
    pl_module=GrilPlNetwork(**pl_kwargs)
    return pl_module

def train(dataset,network,args):
    #logging config
    logdir=os.path.join(args.logdir,args.model_name,args.dataset_name)
    
    #traning settings
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(network, datamodule=dataset)  

if __name__==('__main__'):
    args=parse_args()

    #load dataset after wrapping 
    dataset_module,adj=dataset_loader(args)
    network=model_loader(dataset_module,adj,args)
    train(dataset_module,network,args)