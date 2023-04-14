import random

from numpy import unravel_index
from sklearn.metrics.pairwise import cosine_similarity
import torch

from ..utils.util import _subgraph_byLinktype, loss_ce, accuracy_dgl, loss_rmse, metric_rmsle, metric_mae, _track_kGradDistances


class Server():
    def __init__(self, model, nlinktype, edge_batchsize, task, device):
        self.model = model
        self.nlinktype = nlinktype  # number of different link types
        # self.linktypes = linktypes  # a list of link types
        self.edge_batchsize = edge_batchsize
        self.task = task
        self.device = device
        if self.task == 'classification':
            self.loss_func = loss_ce
            self.metric_func = {'acc': accuracy_dgl}
        if self.task == 'regression':
            self.loss_func = loss_rmse
            self.metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

        self.Ws = {k: v for k, v in self.model.named_parameters()}
        self.groups = None  # maintaining the mapping between global models and local models, {(client_id, idx_cluster): idx_group, ...}
        self.centers = None # averaging centroids by groups

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def group_centroids(self, ids_clients):
        if self.groups is None: # first time to group centroids
            init_client_id = next(iter(ids_clients))    # get the id of the first item (client)
            self.centers = ids_clients[init_client_id].centroids.clone()    # initialize the centers with the first item (client)
            self.groups = {(init_client_id, i): i for i in range(self.nlinktype)}   # initialize each group with one centroid of the first item (client)
            for client_id, client in ids_clients.items():
                if client_id == init_client_id:
                    continue
                # calculate cosine similarity matrix for each client to centers
                cos_sim = torch.stack([torch.nn.functional.cosine_similarity(client.centroids, cen) for cen in self.centers])
                # cos_sim = cosine_similarity(client.centroids, self.centers)
                # separate centroids to different groups
                _update_groups(self.nlinktype, self.groups, client_id, cos_sim)

        else:
            _clear_groups(self.groups)
            for client_id, client in ids_clients.items():
                cos_sim = torch.stack([torch.nn.functional.cosine_similarity(client.centroids, cen) for cen in self.centers])
                # cos_sim = cosine_similarity(client.centroids, self.centers)
                _update_groups(self.nlinktype, self.groups, client_id, cos_sim)

        # re-compute the group centers
        _update_centers(self.groups, self.centers, ids_clients)
        # print('server center', self.centers)

    def aggregate(self, ids_clients):
        # averagely aggregates the shared feature projection layer (feature1)
        total_size = 0
        for _, client in ids_clients.items():
            total_size += client.total_train_size
        _aggregate_shared_layer(self.Ws, ids_clients.values(), 'feature1', total_size)
        _aggregate_shared_layer(self.Ws, ids_clients.values(), 'classifier', total_size)

        # averagely aggregates the branches of split_gcns (note the train size is for subgraphs)
        groups = {}
        for client_cluster, idx_group in self.groups.items():
            if idx_group not in groups:
                groups[idx_group] = []
            groups[idx_group].append(client_cluster)
        for idx_group, group in groups.items():
            _aggregate_branch(self.Ws, ids_clients, idx_group, group)

        """ For analysis of FL client's contribution """
        # calculate the cosine distance between local gradients and server gradients
        self.kGradDistances = [_track_kGradDistances(ids_clients, group) for _, group in groups.items()]

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            node_embs_all = self.model.feature_projection(data.ndata['x']).detach()
            edges = torch.stack(
                (torch.where(data.edges()[0] < data.edges()[1], data.edges()[0], data.edges()[1]),
                 torch.where(data.edges()[0] > data.edges()[1], data.edges()[0], data.edges()[1])))

            # use batches for edge_embds in case of out of memory
            clusters = []
            for b in range(len(edges[0]) // self.edge_batchsize + 1):
                edge_embs_part = torch.cat((node_embs_all[edges[0][b*self.edge_batchsize:(b+1)*self.edge_batchsize]],
                                            node_embs_all[edges[1][b*self.edge_batchsize:(b+1)*self.edge_batchsize]]), 1)
                clusters_part = torch.argmax(
                    torch.stack([torch.nn.functional.cosine_similarity(edge_embs_part, cen) for cen in self.centers]), dim=0)
                clusters.append(clusters_part)
            del edge_embs_part
            clusters = torch.cat(clusters)

            subgraphs = _subgraph_byLinktype(data.cpu(), clusters, self.nlinktype, self.device)
            for subg in subgraphs:
                subg.ndata['x'] = node_embs_all[subg.ndata['_ID']].clone()
            del node_embs_all
            out = self.model.split_forward(subgraphs, data.num_nodes(), self.device)
            del subgraphs
            pred = self.model.classify(out)
            loss = self.loss_func(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]]).item()
            metrics = {}
            for mname, mfunc in self.metric_func.items():
                metrics[mname] = mfunc(pred[data.ndata[mask]], data.ndata['y'][data.ndata[mask]])
        return loss, metrics


def _clear_groups(groups):
    for k in groups:
        groups[k] = []

def _update_groups(nlinktype, groups, client_id, cos_sim):
    for _ in range(nlinktype):
        (idx_cluster, idx_center) = unravel_index(cos_sim.cpu().argmax(), cos_sim.shape)
        groups[(client_id, idx_cluster)] = idx_center
        cos_sim[idx_cluster, :] = -float('Inf')
        cos_sim[:, idx_center] = -float('Inf')

def _update_centers(groups, centers, ids_clients):
    tmps = {}
    for (client_id, idx_cluster), idx_group in groups.items():
        if idx_group not in tmps:
            tmps[idx_group] = []
        tmps[idx_group].append(ids_clients[client_id].centroids[idx_cluster])
    for idx_group, v in tmps.items():
        centers[idx_group] = torch.mean(torch.stack(v, dim=0), dim=0)

def _aggregate_shared_layer(serverWs, clients, name, total_size):
    serverWs[f'{name}.weight'].data = torch.div(
            torch.sum(torch.stack([torch.mul(client.Ws[f'{name}.weight'].data, client.total_train_size) for client in clients]),
                      dim=0), total_size).clone()
    serverWs[f'{name}.bias'].data = torch.div(
        torch.sum(torch.stack([torch.mul(client.Ws[f'{name}.bias'].data, client.total_train_size) for client in clients]),
                  dim=0), total_size).clone()

def _aggregate_branch(serverWs, ids_clients, idx_group, group):
    for name in serverWs:
        if name.startswith(f'split_gnns.{idx_group}'):
            suffix = name.split(f'split_gnns.{idx_group}')[1]
            # only aggregate when parameter is not all-zeros
            tmp = [ids_clients[client_id].Ws[f'split_gnns.{idx_cluster}{suffix}'].data for (client_id, idx_cluster) in
                 group if ids_clients[client_id].cluster_train_size[idx_cluster] != 0]
            if len(tmp) != 0:
                serverWs[name].data = torch.mean(torch.stack(tmp, dim=0), dim=0)
