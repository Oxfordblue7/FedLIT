import time

import pandas as pd
import torch

from ..utils.util import accuracy_dgl, loss_ce, loss_rmse, metric_rmsle, metric_mae, _to_wandb


class CentralDevice_basic():
    def __init__(self, model, data, optimizer, args):
        self.model = model
        self.data = data.to(args.device)   # a DGLGraph
        self.optimizer = optimizer
        self.args = args
        if self.args.task == 'classification':
            self.loss_func = loss_ce
            self.metric_func = {'acc': accuracy_dgl}
        if self.args.task == 'regression':
            self.loss_func = loss_rmse
            self.metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

        self.trainhistory = pd.DataFrame()

    def train(self, epoch, prefix):
        for e in range(epoch):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(self.data)
            loss = self.loss_func(logits[self.data.ndata['train_mask']], self.data.ndata['y'][self.data.ndata['train_mask']])
            loss.backward()
            self.optimizer.step()

            loss_train, metrics_train = self.evaluate('train_mask')

            self.trainhistory.loc[e, 'train_loss'] = loss_train
            for mname, metric in metrics_train.items():
                self.trainhistory.loc[e, f'train_{mname}'] = metric

            # _to_wandb(f'{prefix}_train', e, loss_train, metrics_train)

            if 'val_mask' in self.data.ndata:
                loss_val, metrics_val = self.evaluate('val_mask')
                self.trainhistory.loc[e, 'val_loss'] = loss_val
                for mname, metric in metrics_val.items():
                    self.trainhistory.loc[e, f'val_{mname}'] = metric

                # _to_wandb(f'{prefix}_val', e, loss_val, metrics_val)


    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)
            loss = self.loss_func(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]]).item()
            metrics = {}
            for mname, mfunc in self.metric_func.items():
                metrics[mname] = mfunc(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]])
        return loss, metrics


class FL_client():
    def __init__(self, id, model, data, optimizer, args):
        self.id = id
        self.model = model
        self.data = data[0].to(args.device)
        self.train_size = data[1]
        self.optimizer = optimizer
        self.args = args
        if self.args.task == 'classification':
            self.loss_func = loss_ce
            self.metric_func = {'acc': accuracy_dgl}
        if self.args.task == 'regression':
            self.loss_func = loss_rmse
            self.metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

        self.Ws = {k: v for k, v in self.model.named_parameters()}

    def download_from_server(self, server):
        for k in server.Ws:
            self.Ws[k].data = server.Ws[k].data.clone()

    def local_train(self, local_epoch):
        for e in range(local_epoch):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(self.data)
            loss = self.loss_func(pred[self.data.ndata['train_mask']], self.data.ndata['y'][self.data.ndata['train_mask']])
            loss.backward()
            self.optimizer.step()

        loss_train, metrics_train = self.evaluate('train_mask')
        if 'val_mask' in self.data.ndata:
            loss_val, metrics_val = self.evaluate('val_mask')
        else:
            loss_val, metrics_val = None, None

        return (loss_train, metrics_train, loss_val, metrics_val)

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)
            loss = self.loss_func(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]]).item()
            metrics = {}
            for mname, mfunc in self.metric_func.items():
                metrics[mname] = mfunc(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]])

        return loss, metrics


class FL_server():
    def __init__(self, model, task, device):
        self.model = model
        self.task = task
        self.device = device
        if self.task == 'classification':
            self.loss_func = loss_ce
            self.metric_func = {'acc': accuracy_dgl}
        if self.task == 'regression':
            self.loss_func = loss_rmse
            self.metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

        self.Ws = {k: v for k, v in self.model.named_parameters()}

    def aggregate(self, clients):
        total_size = 0
        for client in clients:
            total_size += client.train_size
        for k in self.Ws:
            self.Ws[k].data = torch.div(
                torch.sum(torch.stack([torch.mul(client.Ws[k].data, client.train_size) for client in clients]), dim=0),
                total_size).clone()

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data)
            loss = self.loss_func(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]]).item()
            metrics = {}
            for mname, mfunc in self.metric_func.items():
                metrics[mname] = mfunc(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]])

        return loss, metrics
