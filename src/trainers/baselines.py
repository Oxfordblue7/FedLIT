import os
import argparse
import random
from pathlib import Path
import time

import numpy as np

import dgl
import wandb

from ..models.models import GCN
from ..models.baseline_devices import CentralDevice_basic, FL_client, FL_server
from ..models.client import _copy_shared_layer, _copy_branch
from ..utils.setup_devices import *
from ..utils.utils import loss_ce, accuracy_dgl, loss_rmse, metric_rmsle, metric_mae, _output_results, _to_wandb


def run_GCN(data, foldk, epoch, outpath):
    """ run on the global graph with a basic GCN model """
    print("> Run GCN baseline ...")
    model = GCN(args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    central_device = CentralDevice_basic(model, data, optimizer, args)
    central_device.train(epoch, 'global')
    outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_GCN.csv')
    central_device.trainhistory.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

    test_loss, test_metrics = central_device.evaluate('test_mask')
    outfile = os.path.join(outpath, f'{foldk}_result_GCN.csv')
    df = pd.DataFrame()
    _output_results(df, 'GCN', test_loss, test_metrics, 'test')
    _to_wandb(f'global_test', None, test_loss, test_metrics)
    df.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

def run_local_GCN(clientData, foldk, epoch, outpath):
    """ run on local graphs with a basic GCN model """
    print("> Run local_GCN baseline ...")
    df_test_local = pd.DataFrame()
    for cid, data in clientData.items():
        model = GCN(args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        central_device = CentralDevice_basic(model, data[0], optimizer, args)
        central_device.train(epoch, f'client{cid}')

        outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_local_GCN_client{cid}.csv')
        central_device.trainhistory.to_csv(outfile, header=True, index=True)
        print(f"Wrote to: {outfile}")

        # evaluating
        test_loss, test_metrics = central_device.evaluate('test_mask')
        # df_test_local.loc[linktype, ['test_loss', 'test_acc']] = [test_loss, test_acc]
        _output_results(df_test_local, cid, test_loss, test_metrics, 'test')
        _to_wandb(f'client{cid}_test', None, test_loss, test_metrics)

    outfile = os.path.join(outpath, f'{foldk}_result_local_GCN.csv')
    df_test_local.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

def run_cGCN(data, foldk, epoch, outpath):
    """ run on the global graph with a cGCN """
    print("> Run cGCN baseline ...")
    model = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout,
                             args.seed).to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    central_device = Client_NC(0, model, (data, None), optimizer, args)

    df_trainhistory = pd.DataFrame()
    for e in range(epoch):
        if e % 20 == 0:
            print(f'  round {e}')
        central_device.train(1)
        (train_loss, train_metrics, val_loss, val_metrics) = central_device.evaluate('train_val_mask')
        # df_trainhistory.loc[e, ['train_loss', 'train_acc', 'val_loss', 'val_acc']] = [train_loss, train_acc, val_loss, val_acc]
        _output_results(df_trainhistory, e, train_loss, train_metrics, 'train')
        _to_wandb('global_train', e, train_loss, train_metrics)
        _output_results(df_trainhistory, e, val_loss, val_metrics, 'val')
        _to_wandb('global_val', e, val_loss, val_metrics)
    outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_cGCN.csv')
    df_trainhistory.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

    (test_loss, test_metrics) = central_device.evaluate('test_mask')
    outfile = os.path.join(outpath, f'{foldk}_result_cGCN.csv')
    df = pd.DataFrame()
    _output_results(df, 'cGCN', test_loss, test_metrics, 'test')
    _to_wandb('global_test', None, test_loss, test_metrics)
    df.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")


def run_mGCN(data, foldk, linktypes, epoch, task, outpath):
    """ run on the oracle graph (linktype known) with a mGCN """
    print("> Run mGCN baseline ...")
    model = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout,
                             args.seed).to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    if task == 'classification':
        loss_func = loss_ce
        metric_func = {'acc': accuracy_dgl}#, 'f1micro': f1_micro}
    if task == 'regression':
        loss_func = loss_rmse
        metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

    df_trainhistory = pd.DataFrame()
    subgraphs = None
    metrics_train = {}
    for e in range(epoch):
        if e % 20 == 0:
            print(f'  round {e}')
        model.train()
        node_embs_all = model.feature_projection(data.ndata['x'])
        if e == 0:
            subgraphs = _subgraph_byLinktype(data.cpu(), data.edata['edge_type'], linktypes, args.device)
        for subg in subgraphs:
            subg.ndata['x'] = node_embs_all[subg.ndata[dgl.NID]].clone()

        optimizer.zero_grad()
        out = model.split_forward(subgraphs, data.num_nodes(), args.device)
        pred = model.classify(out)
        loss_train = loss_func(pred[data.ndata['train_mask']], data.ndata['y'][data.ndata['train_mask']])
        for mname, mfunc in metric_func.items():
            metrics_train[mname] = mfunc(pred[data.ndata['train_mask']], data.ndata['y'][data.ndata['train_mask']])

        loss_train.backward()
        optimizer.step()

        loss_val, metrics_val = _evaluate_mGCN(model, data, subgraphs, 'val_mask', loss_func, metric_func, args.device)
        _output_results(df_trainhistory, e, loss_train.item(), metrics_train, 'train')
        _to_wandb('global_train', e, loss_train.item(), metrics_train)
        _output_results(df_trainhistory, e, loss_val, metrics_val, 'val')
        _to_wandb('global_val', e, loss_val, metrics_val)

    outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_mGCN.csv')
    df_trainhistory.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

    # evaluate
    loss_test, metrics_test = _evaluate_mGCN(model, data, subgraphs, 'test_mask', loss_func, metric_func, args.device)
    outfile = os.path.join(outpath, f'{foldk}_result_mGCN.csv')
    df = pd.DataFrame()
    _output_results(df, 'mGCN', loss_test, metrics_test, 'test')
    _to_wandb('global_test', None, loss_test, metrics_test)
    df.to_csv(outfile, header=True, index=True)
    print(f"Wrote to: {outfile}")

def _subgraph_byLinktype(data, clusters, linktypes, device):
    """ for DGL data """
    subgraphs = []
    for c in linktypes:
        subg = dgl.edge_subgraph(data, clusters==c)
        subgraphs.append(subg.to(device))
    return subgraphs

def _evaluate_mGCN(model, data, subgraphs, mask, loss_func, metric_func, device):
    model.eval()
    with torch.no_grad():
        node_embs_all = model.feature_projection(data.ndata['x']).detach()
        for subg in subgraphs:
            subg.ndata['x'] = node_embs_all[subg.ndata[dgl.NID]].clone()
        out = model.split_forward(subgraphs, data.num_nodes(), device)
        pred = model.classify(out)

        loss = loss_func(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]]).item()
        metrics = {}
        for mname, mfunc in metric_func.items():
            metrics[mname] = mfunc(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]])
    return loss, metrics

def run_FedmGCN(globaldata, clientData, foldk, linktypes, num_round, local_epoch, task, outpath):
    """ run FL with mGCN on oracle graphs """
    print('> Run Fed-mGCN baseline ...')
    start = time.time()
    clients = []
    for cid_ltype, data in clientData.items():
        cmodel = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout,
                                  args.seed).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        clients.append(Client_NC(cid_ltype, cmodel, data, optimizer, args))

    smodel = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
    server = Server(smodel, args.nlinktype, args.edge_batchsize, args.task, args.device)

    if task == 'classification':
        loss_func = loss_ce
        metric_func = {'acc': accuracy_dgl}
    if task == 'regression':
        loss_func = loss_rmse
        metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

    for client in clients:
        client.download_from_server(server)

    trainhistory_local = {}
    df_trainhistory_global = pd.DataFrame()
    subgraphs = {client.id: None for client in clients}
    subgraphs_global = None
    ids_clients = {client.id: client for client in clients}
    for r in range(num_round):
        print(f" round {r}")
        for client in clients:
            client.model.train()
            subgraphs[client.id], loss_train, metrics_train = _train_FedmGCN(r, local_epoch, client, linktypes, subgraphs[client.id], loss_func, metric_func)
            # store training history (in local level)
            if client.id not in trainhistory_local:
                trainhistory_local[client.id] = pd.DataFrame()
            _output_results(trainhistory_local[client.id], r, loss_train.item(), metrics_train, 'train')
            _to_wandb(f'client{client.id}_train', r, loss_train.item(), metrics_train)

        if r == 0:  # for the first time
            # assign server's groups
            server.groups = {}
            for client in clients:
                for idx_cluster in range(args.nlinktype):
                    server.groups[(client.id, idx_cluster)] = idx_cluster
            # subgraph the global data
            subgraphs_global = _subgraph_byLinktype(globaldata.cpu(), globaldata.edata['edge_type'], linktypes,
                                             args.device)

        # aggregate local models
        server.aggregate(ids_clients)

        for client in clients:
            _download_from_server(server, client)
            # evaluate local validation data
            loss_val, metrics_val = _evaluate_mGCN(client.model, client.data, subgraphs[client.id], 'val_mask', loss_func, metric_func, args.device)
            # store training history (in local level)
            _output_results(trainhistory_local[client.id], r, loss_val, metrics_val, 'val')
            _to_wandb(f'client{client.id}_val', r, loss_val, metrics_val)

        # evaluate on the global val data
        val_loss, metrics_val = _evaluate_mGCN(server.model, globaldata, subgraphs_global, 'val_mask', loss_func, metric_func, args.device)
        _output_results(df_trainhistory_global, r, val_loss, metrics_val, 'val')
        _to_wandb(f'global_val', r, val_loss, metrics_val)

    # write to files
    for client_id, df in trainhistory_local.items():
        outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_FedmGCN_local_client{client_id}.csv')
        df.to_csv(outfile, header=True, index=True)
        print(f"Wrote to {outfile}")

    outfile = os.path.join(outpath, f'{foldk}_trainhistory_val_FedmGCN_global.csv')
    df_trainhistory_global.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    # evaluate on the test data
    df_test_local = pd.DataFrame()
    for client in clients:
        test_loss, metrics_test = _evaluate_mGCN(client.model, client.data, subgraphs[client.id], 'test_mask', loss_func, metric_func, args.device)
        _output_results(df_test_local, client.id, test_loss, metrics_test, 'test')
        _to_wandb(f'client{client.id}_test', None, test_loss, metrics_test)
    outfile = os.path.join(outpath, f'{foldk}_result_FedmGCN_local.csv')
    df_test_local.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    df_test_global = pd.DataFrame()
    test_loss, metrics_test = _evaluate_mGCN(server.model, globaldata, subgraphs_global, 'test_mask', loss_func, metric_func, args.device)
    _output_results(df_test_global, 'FedmGCN', test_loss, metrics_test, 'test')
    _to_wandb(f'global_test', None, test_loss, metrics_test)
    outfile = os.path.join(outpath, f'{foldk}_result_FedmGCN_global.csv')
    df_test_global.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    print("Total time:", time.time()-start)

def _train_FedmGCN(r, local_epoch, client, linktypes, subgraphs, loss_func, metric_func):
    metrics_train = {}
    for e in range(local_epoch):
        client.node_embs_all = client.model.feature_projection(client.data.ndata['x'])
        if r == 0 and e == 0:  # for the first time
            # subgraphing
            subgraphs = _subgraph_byLinktype(client.data.cpu(), client.data.edata['edge_type'], linktypes,
                                             args.device)
            client.cluster_train_size = []
            for subg in subgraphs:
                client.cluster_train_size.append(len(subg.ndata['train_mask'].nonzero()))

        for subg in subgraphs:
            subg.ndata['x'] = client.node_embs_all[subg.ndata[dgl.NID]].clone()

        client.optimizer.zero_grad()
        out = client.model.split_forward(subgraphs, client.data.num_nodes(), args.device)
        pred = client.model.classify(out)
        loss_train = loss_func(pred[client.data.ndata['train_mask']],
                             client.data.ndata['y'][client.data.ndata['train_mask']])
        for mname, mfunc in metric_func.items():
            metrics_train[mname] = mfunc(pred[client.data.ndata['train_mask']],
                                    client.data.ndata['y'][client.data.ndata['train_mask']])
        loss_train.backward()
        client.optimizer.step()

    return subgraphs, loss_train, metrics_train

def _download_from_server(server, client):
    # copy the shared layer
    _copy_shared_layer(server.Ws, client.Ws, 'feature1')
    _copy_shared_layer(server.Ws, client.Ws, 'classifier')
    # copy the split GCN according to the alignment
    for (client_id, idx_cluster), idx_group in server.groups.items():
        # download weights
        _copy_branch(server.Ws, idx_group, client.Ws, idx_cluster)

def run_FedGCN(globaldata, clientData, foldk, num_round, local_epoch, outpath):
    """ run FedAvg with basic models """
    print("> Run Fed-GCN baseline ...")
    clients = []
    for cid, data in clientData.items():
        model = GCN(args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(FL_client(cid, model, data, optimizer, args))

    model = GCN(args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
    server = FL_server(model, args.task, args.device)

    for client in clients:
        client.download_from_server(server)

    trainhistory_local = {}
    df_trainhistory_global = pd.DataFrame()
    for r in range(num_round):
        if r % 20 == 0:
            print(f'  round {r}')
        for client in clients:
            (train_loss, train_metrics, val_loss, val_metrics) = client.local_train(local_epoch)
            if client.id not in trainhistory_local:
                trainhistory_local[client.id] = pd.DataFrame()
            _output_results(trainhistory_local[client.id], r, train_loss, train_metrics, 'train')
            _to_wandb(f'client{client.id}_train', r, train_loss, train_metrics)
            _output_results(trainhistory_local[client.id], r, val_loss, val_metrics, 'val')
            _to_wandb(f'client{client.id}_val', r, val_loss, val_metrics)

        server.aggregate(clients)
        for client in clients:
            client.download_from_server(server)

        # evaluate on the global graph
        val_loss, val_metrics = server.evaluate(globaldata, 'val_mask')
        _output_results(df_trainhistory_global, r, val_loss, val_metrics, 'val')
        _to_wandb('global_val', r, val_loss, val_metrics)

    # write local training history to files
    for client_id, df in trainhistory_local.items():
        outfile = os.path.join(outpath, f'{foldk}_trainhistory_trainval_FedGCN_local_client{client_id}.csv')
        df.to_csv(outfile, header=True, index=True)
        print(f"Wrote to: {outfile}")

    outfile = os.path.join(outpath, f'{foldk}_trainhistory_val_FedGCN_global.csv')
    df_trainhistory_global.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    # evaluate on the test data (in local level)
    df_test_local = pd.DataFrame()
    for client in clients:
        test_loss, test_metrics = client.evaluate('test_mask')
        # df_test_local.loc[client.id, ['test_loss', 'test_acc']] = [test_loss, test_acc]
        _output_results(df_test_local, client.id, test_loss, test_metrics, 'test')
        _to_wandb(f'client{client.id}_test', None, test_loss, test_metrics)
    outfile = os.path.join(outpath, f'{foldk}_result_FedGCN_local.csv')
    df_test_local.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    # evaluate on the test data (in global level)
    df_test_global = pd.DataFrame()
    test_loss, test_metrics = server.evaluate(globaldata, 'test_mask')
    _output_results(df_test_global, 'FedGCN', test_loss, test_metrics, 'test')
    _to_wandb('global_test', None, test_loss, test_metrics)
    outfile = os.path.join(outpath, f'{foldk}_result_FedGCN_global.csv')
    df_test_global.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--dataset', type=str, default='dblp-dm',
                        help='The name of the dataset.')
    parser.add_argument('--datapath', type=str, default='./data/dm',
                        help='The path to the dataset.')
    parser.add_argument('--outpath', type=str, default='./output/dm',
                        help='The output path.')
    parser.add_argument('--foldk', type=int, default=0,
                        help='The kth fold.')
    parser.add_argument('--baseline', type=str, default='oracle',
                        help='The name of baselines.')
    parser.add_argument('--partition', type=str, default='oneDominant',
                        help='The way of data partitioning.')
    parser.add_argument('--nClients', type=int, default=4,
                        help='The number of clients to split (for balanced data partition).')
    parser.add_argument('--nlinktype', type=int, default=4,
                        help='The num of linktypes.')
    parser.add_argument('--test_linktypes', type=str, default='0-1-2-3',
                        help='The linktypes of the test graph.')
    parser.add_argument('--task', type=str, default='classification',
                        help='The downstream task (classification/regression)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for randomness.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='The size of node batches.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='The number of workers for node batches.')
    parser.add_argument('--num_round', type=int, default=50,
                        help='The number of communication round')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='The number of local training epoch')
    parser.add_argument('--edge_batchsize', type=int, default=5000000,
                        help='The batch size of edges for clustering')
    parser.add_argument('--num_iterEM', type=int, default=1,
                        help='The number of E-M iteration for clustering')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of GNN conv layers')
    parser.add_argument('--nhidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nfeature', type=int, default=200,
                        help='Number of node features')
    parser.add_argument('--nclass', type=int, default=12,
                        help='Number of node labels')

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dgl.seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    linktypes = [int(x) for x in args.test_linktypes.split("-")]
    print(f'> Linktypes: {linktypes}')

    outpath = os.path.join(args.outpath, args.test_linktypes)

    if not args.baseline.startswith('global'):
        print("> Loading local data ...")
        assert args.partition in ['distinct', 'oneDominant', 'balanced']
        print(f'    {args.partition}')
        if args.partition == 'distinct':
            clientData = prepare_local_data_distinct(args.datapath, args.foldk)
            outpath = os.path.join(outpath, 'distinct')
        if args.partition == 'oneDominant':
            clientData = prepare_local_data_oneDominant(args.datapath, args.foldk)
            outpath = os.path.join(outpath, 'oneDominant')
        if args.partition == 'balanced':
            clientData = prepare_local_data_balanced(args.datapath, args.foldk)
            outpath = os.path.join(outpath, 'balanced')
        print("> Data loaded.")

    wandb.init(
        project="FedLIT",
        name=f'baselines_{args.dataset}_{args.partition}_fold{args.foldk}',
        config=args
    )

    Path(outpath).mkdir(parents=True, exist_ok=True)

    print(f'> Baseline: {args.baseline}')
    assert args.baseline in ['GCN', 'mGCN', 'cGCN', 'FedGCN', 'FedmGCN', 'local_GCN']

    # 6. local+1
    if args.baseline == 'local_GCN':
        run_local_GCN(clientData, args.foldk, args.num_round, outpath)

    print("> Loading global data ...")
    globalData = prepare_global_data(args.datapath, args.test_linktypes, args.foldk, args.device)
    print("> Data loaded.")

    # 1. Fed-GCN
    if args.baseline == 'FedGCN':
        run_FedGCN(globalData, clientData, args.foldk, args.num_round, args.local_epoch, outpath)
    # 2. Fed-mGCN
    if args.baseline == 'FedmGCN':
        run_FedmGCN(globalData, clientData, args.foldk, linktypes, args.num_round, args.local_epoch, args.task, outpath)
    # 3. GCN
    if args.baseline == 'GCN':
        run_GCN(globalData, args.foldk, args.num_round, outpath)
    # 5. mGCN
    if args.baseline == 'mGCN':
        run_mGCN(globalData, args.foldk, linktypes, args.num_round, args.task, outpath)
    # 4. cGCN
    if args.baseline == 'cGCN':
        run_cGCN(globalData, args.foldk, args.num_round, outpath)
