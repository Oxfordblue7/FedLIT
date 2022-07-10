import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import dgl
import time


from ..utils.utils import _subgraph_byLinktype, loss_ce, accuracy_dgl, loss_rmse, metric_mae, metric_rmsle

class Client_NC():
    def __init__(self, id, model, data, optimizer, args):
        self.id = id
        self.model = model
        self.data = data[0].to(args.device)
        self.total_train_size = data[1]
        self.optimizer = optimizer
        self.args = args
        if self.args.task == 'classification':
            self.loss_func = loss_ce
            self.metric_func = {'acc': accuracy_dgl}
        if self.args.task == 'regression':
            self.loss_func = loss_rmse
            self.metric_func = {'rmsle': metric_rmsle, 'mae': metric_mae}

        self.Ws = {k: v for k, v in self.model.named_parameters()}
        self.initial_download = True    # first downloading or not
        self.initial_centroids = True   # initializing centroids or not
        self.centroids = None   # maintaining centroids, dim = (nlinktype, dim_edgeembeddings)
        self.clusters = None    # maintaining the mapping between edges and centroids
        self.cluster_train_size = []  # maintaining the training size for clusters, dim = (nlinktype, )

        # get the one-directional edges, to make the embedding of (v, u) same as it of (u, v)
        self.edges = torch.stack(
            (torch.where(self.data.edges()[0] < self.data.edges()[1], self.data.edges()[0], self.data.edges()[1]),
             torch.where(self.data.edges()[0] > self.data.edges()[1], self.data.edges()[0], self.data.edges()[1])))
        self.node_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.nlayer)

    def download_from_server(self, server):
        if self.initial_download:
            _copy_model(server.Ws, self.Ws)
            self.initial_download = False
        else:
            # copy the shared layer
            _copy_shared_layer(server.Ws, self.Ws, 'feature1')
            _copy_shared_layer(server.Ws, self.Ws, 'classifier')
            # copy the split GCN according to the alignment
            for (client_id, idx_cluster), idx_group in server.groups.items():
                # download weights
                _copy_branch(server.Ws, idx_group, self.Ws, idx_cluster)
                # download centroids
                self.centroids[idx_cluster] = server.centers[idx_group].clone()

    def train(self, local_epoch):
        for e in range(local_epoch):
            self.model.train()
            node_embs_all = self.model.feature_projection(self.data.ndata['x'])
            self.assign_centroids(node_embs_all)
            # subgraph based on different linktypes
            subgraphs = _subgraph_byLinktype(self.data.cpu(), self.clusters, self.args.nlinktype, self.args.device)
            # get the training size for subgraphs
            self.cluster_train_size = []
            for subg in subgraphs:
                subg.ndata['x'] = node_embs_all[subg.ndata['_ID']].clone()
                self.cluster_train_size.append(len(subg.ndata['train_mask'].nonzero()))
            del node_embs_all

            # train split-GCN
            self.optimizer.zero_grad()
            out = self.model.split_forward(subgraphs, self.data.num_nodes(), self.args.device)
            del subgraphs
            pred = self.model.classify(out)
            loss = self.loss_func(pred[self.data.ndata['train_mask']], self.data.ndata['y'][self.data.ndata['train_mask']])
            loss.backward()
            self.optimizer.step()

    def assign_centroids(self, node_embs_all):
        if self.initial_centroids:
            edge_embs = torch.cat(
                (node_embs_all[self.edges[0]].cpu().detach(), node_embs_all[self.edges[1]].cpu().detach()), 1)
            self.centroids, self.clusters = _initialize_centroids(edge_embs, self.args.nlinktype, self.args.seed)
            self.centroids = self.centroids.to(self.args.device)
            self.clusters = self.clusters.to(self.args.device)
            self.initial_centroids = False
            del edge_embs
        else:
            if self.args.device == 'cpu':
                edge_embs = torch.cat(
                    (node_embs_all[self.edges[0]].cpu().detach(), node_embs_all[self.edges[1]].cpu().detach()), 1)
                self.centroids, self.clusters = _update_clusters_cpu(edge_embs, self.centroids, self.args.num_iterEM)
                del edge_embs
            else:
                if self.edges.size(1) > 10000000:
                    print('  batching edges')
                    self.centroids, self.clusters = _update_clusters_gpu_batching(node_embs_all.detach(),
                                                                                  self.edges,
                                                                                  self.args.edge_batchsize,
                                                                                  self.centroids, self.args.num_iterEM,
                                                                                  self.args.device)
                else:
                    edge_embs = torch.cat(
                        (node_embs_all[self.edges[0]].detach(), node_embs_all[self.edges[1]].detach()), 1)
                    self.centroids, self.clusters = _update_clusters_gpu(edge_embs, self.centroids, self.args.num_iterEM)
                    del edge_embs


    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            node_embs_all = self.model.feature_projection(self.data.ndata['x']).detach()
            if self.args.device == 'cpu':
                clusters = torch.tensor(np.argmax(cosine_similarity(
                    torch.cat((node_embs_all[self.data.edges()[0]], node_embs_all[self.data.edges()[1]]), 1).cpu(),
                    self.centroids), axis=1))
            else:
                # use batches for edge_embds in case of out of memory
                clusters = []
                for b in range(len(self.data.edges()[0]) // self.args.edge_batchsize + 1):
                    edge_embs_part = torch.cat(
                        (node_embs_all[self.edges[0][b * self.args.edge_batchsize:(b + 1) * self.args.edge_batchsize]],
                         node_embs_all[self.edges[1][b * self.args.edge_batchsize:(b + 1) * self.args.edge_batchsize]]), 1)
                    # assign edges to centroids (update clusters)
                    clusters_part = torch.argmax(
                        torch.stack(
                            [torch.nn.functional.cosine_similarity(edge_embs_part, cen) for cen in self.centroids]),
                        dim=0)
                    clusters.append(clusters_part)
                del edge_embs_part
                clusters = torch.cat(clusters)
            # del edge_embs

            subgraphs = _subgraph_byLinktype(self.data.cpu(), clusters, self.args.nlinktype, self.args.device)
            for subg in subgraphs:
                subg.ndata['x'] = node_embs_all[subg.ndata['_ID']].clone()
            del node_embs_all
            out = self.model.split_forward(subgraphs, self.data.num_nodes(), self.args.device)
            del subgraphs
            pred = self.model.classify(out)

            if mask == 'train_val_mask':
                loss_train = self.loss_func(pred[self.data.ndata['train_mask']], self.data.ndata['y'][self.data.ndata['train_mask']]).item()
                metrics_train = {}
                for mname, mfunc in self.metric_func.items():
                    metrics_train[mname] = mfunc(pred[self.data.ndata['train_mask']], self.data.ndata['y'][self.data.ndata['train_mask']])
                loss_val = self.loss_func(pred[self.data.ndata['val_mask']], self.data.ndata['y'][self.data.ndata['val_mask']]).item()
                metrics_val = {}
                for mname, mfunc in self.metric_func.items():
                    metrics_val[mname] = mfunc(pred[self.data.ndata['val_mask']], self.data.ndata['y'][self.data.ndata['val_mask']])
                return (loss_train, metrics_train, loss_val, metrics_val)
            else:
                loss = self.loss_func(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]]).item()
                metrics = {}
                for mname, mfunc in self.metric_func.items():
                    metrics[mname] = mfunc(pred[self.data.ndata[mask]], self.data.ndata['y'][self.data.ndata[mask]])
                return (loss, metrics)


def _copy_model(source, target):
    for k in source:
        target[k].data = source[k].data.clone()

def _copy_shared_layer(source, target, name):
    target[f'{name}.weight'].data = source[f'{name}.weight'].data.clone()
    target[f'{name}.bias'].data = source[f'{name}.bias'].data.clone()

def _copy_branch(source, sbranch, target, tbranch):
    for name in source:
        if name.startswith(f'split_gnns.{sbranch}'):
            suffix = name.split(f'split_gnns.{sbranch}')[1]
            target[f'split_gnns.{tbranch}{suffix}'].data = source[name].data.clone()

def _initialize_centroids(edge_embs, k, seed):
    # use k-mean to initialize the centroids
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, max_iter=1).fit(edge_embs)
    return torch.tensor(kmeans.cluster_centers_), torch.tensor(kmeans.labels_)

def _update_clusters_gpu(edge_embs, centroids, niter):
    """ E-M: remapping edges to new centroids (M), updating centroids (E) """
    for i in range(niter):
        clusters = torch.argmax(
            torch.stack([torch.nn.functional.cosine_similarity(edge_embs, cen) for cen in centroids]), dim=0)
        centroids = torch.stack([torch.mean(edge_embs[clusters == c], dim=0) for c in range(centroids.size(0))], dim=0)
    return centroids, clusters

def _update_clusters_gpu_batching(node_embs_all, edges, edge_batchsize, centroids, niter, device):
    """ E-M: remapping edges to new centroids (M), updating centroids (E) """
    nlinktype = centroids.size(0)
    for i in range(niter):
        # print('iteration', i)
        clusters = []
        for b in range(edges.size(1) // edge_batchsize + 1):
            # assign edges to centroids (update clusters)
            edge_embs_part = torch.cat((node_embs_all[edges[0][b * edge_batchsize:(b + 1) * edge_batchsize]],
                       node_embs_all[edges[1][b * edge_batchsize:(b + 1) * edge_batchsize]]), 1)
            clusters_part = torch.argmax(
                torch.stack([torch.nn.functional.cosine_similarity(edge_embs_part, cen) for cen in centroids]), dim=0)
            clusters.append(clusters_part)
        clusters = torch.cat(clusters)

        centroids = torch.zeros((nlinktype, node_embs_all.size(1) * 2)).to(device)
        counts = torch.zeros(nlinktype).to(device)
        for b in range(edges.size(1) // edge_batchsize + 1):
            edge_embs_part = torch.cat((node_embs_all[edges[0][b * edge_batchsize:(b + 1) * edge_batchsize]],
                                        node_embs_all[edges[1][b * edge_batchsize:(b + 1) * edge_batchsize]]), 1)
            centroids += torch.stack(
                [torch.sum(edge_embs_part[clusters[b * edge_batchsize:(b + 1) * edge_batchsize] == c].detach(), dim=0)
                 for c in range(nlinktype)], dim=0)
            counts += torch.stack([(clusters[b * edge_batchsize:(b + 1) * edge_batchsize] == c).count_nonzero() for c in
                                   range(nlinktype)])
        centroids = torch.divide(centroids, counts.reshape(-1, 1))

    return centroids, clusters


def _update_clusters_cpu(edge_embs, centroids, niter):
    """ E-M: remapping edges to new centroids (M), updating centroids (E) """
    for i in range(niter):
        # assign edges to centroids (update clusters)
        clusters = torch.tensor(np.argmax(cosine_similarity(edge_embs, centroids), axis=1))
        # calculate new centroids
        centroids = torch.stack([torch.mean(edge_embs[clusters == c], dim=0) for c
                                 in range(centroids.size(0))], dim=0)

    return centroids, clusters

