import os
import re
import pandas as pd
import torch
from dgl import load_graphs

from ..models.models import multichannel_GCN
from ..models.client import Client_NC
from ..models.server import Server

import collections

def _get_fold(datapath, dglG, cvfolder, foldk, suffix):
    # get tran/val masks for one fold of data
    train_indices = \
        pd.read_csv(os.path.join(datapath, cvfolder, f'{foldk}_indices_train_{suffix}.txt'), header=None,
                    index_col=None)[0].tolist()
    val_indices = \
        pd.read_csv(os.path.join(datapath, cvfolder, f'{foldk}_indices_val_{suffix}.txt'), header=None,
                    index_col=None)[0].tolist()
    fold_train_mask = torch.tensor([True if i in train_indices else False for i in range(dglG.number_of_nodes())])
    fold_val_mask = torch.tensor([True if i in val_indices else False for i in range(dglG.number_of_nodes())])

    dglG.ndata['train_mask'] = fold_train_mask
    dglG.ndata['val_mask'] = fold_val_mask

    return dglG

def prepare_data(datapath, str_linktypes, foldk):
    """ Setting: each client has one distinct link type. """
    linktypes = str_linktypes.split('-')
    clientData = {}   # key: link-type, value: (graph, train_size)
    for ltype in linktypes:
        print("  link type:", ltype)
        cvfolder = 'dgl_cv_folds_oracle_10'
        dglG = load_graphs(os.path.join(datapath, f'graph_oracle_linkType{ltype}.bin'))[0][0]
        dglG = _get_fold(datapath, dglG, cvfolder, foldk, f'linkType{ltype}')
        dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
        dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)

        clientData[ltype] = (dglG, len(dglG.ndata['train_mask'].nonzero()))

        counter = collections.Counter(dglG.edata['edge_type'].cpu().numpy())
        print(f"        link-type distribution: {counter}")

    return clientData


def prepare_local_data_distinct(datapath, foldk):
    datapath = os.path.join(datapath, 'graphs_oracle_distinct_numClient10')
    cvfolder = 'cv_folds_10'
    clientData = {}
    for filename in os.listdir(datapath):
        if filename.startswith('graph_'):
            groups = re.search(r'graph_client(\d+)_linktype(\d+).bin', filename)
            idxc = groups.group(1)
            ltype = groups.group(2)
            print(f"    client {idxc}: linktype - {ltype}")

            dglG = load_graphs(os.path.join(datapath, filename))[0][0]
            dglG = _get_fold(datapath, dglG, cvfolder, foldk, f'client{idxc}_linktype{ltype}')
            dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
            dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)

            clientData[f'{idxc}_{ltype}'] = (dglG, len(dglG.ndata['train_mask'].nonzero()))

            counter = collections.Counter(dglG.edata['edge_type'].cpu().numpy())
            print(f"        link-type distribution: {counter}")
    return clientData


def prepare_local_data_common_nodeset(datapath, str_linktypes, device):
    """ for data insight """
    allData = {}
    linktypes = str_linktypes.split('-') + [str_linktypes]
    for ltype in linktypes:
        print("  link type:", ltype)
        dglG = load_graphs(os.path.join(datapath, 'graphs_common_nodeset', f'graph_oracle_linkType{ltype}.bin'))[0][0]
        dglG.ndata['train_mask'] = dglG.ndata['train_mask'].to(torch.bool)
        dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
        dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)

        allData[ltype] = (dglG.to(device), len(dglG.ndata['train_mask'].nonzero()))

    return allData


def prepare_local_data_oneDominant(datapath, foldk):
    datapath = os.path.join(datapath, 'graphs_oracle_oneDominant_numClient10')
    cvfolder = 'cv_folds_10'
    clientData = {}
    for filename in os.listdir(datapath):
        if filename.startswith('graph_'):
            groups = re.search(r'graph_client(\d+)_linktype(\d+).bin', filename)
            idxc = groups.group(1)
            ltype = groups.group(2)
            print(f"    client {idxc}: linktype - {ltype}")

            dglG = load_graphs(os.path.join(datapath, filename))[0][0]
            dglG = _get_fold(datapath, dglG, cvfolder, foldk, f'client{idxc}_linktype{ltype}')
            dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
            dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)

            clientData[f'{idxc}_{ltype}'] = (dglG, len(dglG.ndata['train_mask'].nonzero()))

            counter = collections.Counter(dglG.edata['edge_type'].cpu().numpy())
            print(f"        link-type distribution: {counter}")
    return clientData


def prepare_local_data_balanced(datapath, foldk):
    datapath = os.path.join(datapath, 'graphs_oracle_balanced_numClient10')
    cvfolder = 'cv_folds_10'
    clientData = {}
    for filename in os.listdir(datapath):
        if filename.startswith('graph_'):
            groups = re.search(r'graph_client(\d+).bin', filename)
            idxc = groups.group(1)
            print(f"    client {idxc}")

            dglG = load_graphs(os.path.join(datapath, filename))[0][0]
            dglG = _get_fold(datapath, dglG, cvfolder, foldk, f'client{idxc}')
            dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
            dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)

            clientData[idxc] = (dglG, len(dglG.ndata['train_mask'].nonzero()))

            counter = collections.Counter(dglG.edata['edge_type'].cpu().numpy())
            print(f"        link-type distribution: {counter}")
    return clientData

def prepare_global_data(datapath, str_linktypes, foldk, device):
    cvfolder = 'dgl_cv_folds_oracle_10'
    dglG = load_graphs(os.path.join(datapath, f'graph_oracle_linkType{str_linktypes}.bin'))[0][0]
    dglG = _get_fold(datapath, dglG, cvfolder, foldk, f'linkType{str_linktypes}')
    dglG.ndata['test_mask'] = dglG.ndata['test_mask'].to(torch.bool)
    dglG.ndata['label_mask'] = dglG.ndata['label_mask'].to(torch.bool)


    return dglG.to(device)


def setup(clientData, args):
    clients = []
    for id, data in clientData.items():
        cmodel = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_NC(id, cmodel, data, optimizer, args))

    smodel = multichannel_GCN(args.nlinktype, args.nfeature, args.nhidden, args.nclass, args.nlayer, args.dropout, args.seed).to(args.device)
    server = Server(smodel, args.nlinktype, args.edge_batchsize, args.task, args.device)

    return clients, server

