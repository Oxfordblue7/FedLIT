import os

import argparse
import random
import time
from pathlib import Path
import pickle

import numpy as np

import dgl
# import wandb
# os.environ["WANDB_SILENT"] = "true"

from src.utils.setup_devices import *
from src.utils.util import _output_results, _to_wandb, _track_centers, _track_kGradNorms


def run_fedLIT(server, clients, globaldata, foldk, num_rounds, local_epoch, outpath, samp=None, frac=1.0):
    print("> Running ...")
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    # clients download the initialized model from server
    for client in clients:
        client.download_from_server(server)

    for round in range(num_rounds):
        if round % 20 == 0:
            print(f" round {round}")
        # samples clients
        selected_clients = sampling_fn(clients, frac)

        # train model for different clusters of linktype
        for client in selected_clients:
            client.train(local_epoch)

        # uploading centroids, local updates
        ids_clients = {client.id: client for client in selected_clients}
        server.group_centroids(ids_clients)
        server.aggregate(ids_clients)

        for client in selected_clients:
            client.download_from_server(server)
            # evaluate local training & validation data
            # (train_loss, train_metrics, val_loss, val_metrics) = client.evaluate('train_val_mask')
            train_loss, train_metrics = client.evaluate('train_mask')
            val_loss, val_metrics = client.evaluate('train_mask')
            
            # _to_wandb(f'client{client.id}_train', round, train_loss, train_metrics)
            # _to_wandb(f'client{client.id}_val', round, val_loss, val_metrics)

        # evaluate on the global graph
        val_loss, val_metrics = server.evaluate(globaldata, 'val_mask')
        
        # _to_wandb(f'global_val', round, val_loss, val_metrics)

    # evaluate on the test data (in local level)
    df_test_local = pd.DataFrame()
    for client in clients:
        (test_loss, test_metrics) = client.evaluate('test_mask')
        _output_results(df_test_local, client.id, test_loss, test_metrics, 'test')
        # _to_wandb(f'client{client.id}_test', None, test_loss, test_metrics)
    outfile = os.path.join(outpath, f'{foldk}_result_local.csv')
    df_test_local.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    # evaluate on the test data (in global level)
    df_test_global = pd.DataFrame()
    test_loss, test_metrics = server.evaluate(globaldata, 'test_mask')
    _output_results(df_test_global, 'global', test_loss, test_metrics, 'test')
    # _to_wandb(f'global_test', None, test_loss, test_metrics)
    outfile = os.path.join(outpath, f'{foldk}_result_global.csv')
    df_test_global.to_csv(outfile, header=True, index=True)
    print(f"Wrote to {outfile}")

    print("=======================================================")
    print(f"Test loss = {test_loss}; {test_metrics}")
    print("-------------------------------------------------------")


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
    parser.add_argument('--partition', type=str, default='dominant',
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
    parser.add_argument('--num_round', type=int, default=100,
                        help='The number of communication round')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='The number of local training epoch')
    parser.add_argument('--edge_batchsize', type=int, default=2000000,
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

    outpath = os.path.join(args.outpath, args.test_linktypes)

    # wandb.init(
    #     project="FedLIT",
    #     name=f'fedlit_{args.dataset}_{args.partition}_fold{args.foldk}',
    #     config=args
    # )

    print("> Loading local data ...")
    assert args.partition in ['distinct', 'dominant', 'balanced']
    print(f'    {args.partition}')
    if args.partition == 'distinct':
        clientData = prepare_local_data_distinct(args.datapath, args.foldk)
        outpath = os.path.join(outpath, 'distinct')
    if args.partition == 'dominant':
        clientData = prepare_local_data_oneDominant(args.datapath, args.foldk)
        outpath = os.path.join(outpath, 'dominant')
    if args.partition == 'balanced':
        clientData = prepare_local_data_balanced(args.datapath, args.foldk)
        outpath = os.path.join(outpath, 'balanced')

    Path(outpath).mkdir(parents=True, exist_ok=True)

    print("> Loading global data ...")
    globalData = prepare_global_data(args.datapath, args.test_linktypes, args.foldk, args.device)
    print("> Data loaded.")

    clients, server = setup(clientData, args)
    print(f"> Devices set up.")

    print(f"> Path to outputs: {outpath}")
    run_fedLIT(server, clients, globalData, args.foldk, args.num_round, args.local_epoch, outpath)




