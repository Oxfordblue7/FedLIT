import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import r2_score, f1_score
import wandb


def normalize_partial(x, normalizer, mask):
    x_norm_partial = normalizer.transform(x[mask])
    # assign values
    idx2 = 0
    for idx1, v in enumerate(mask):
        if v:
            x[idx1] = torch.tensor(x_norm_partial[idx2])
            idx2 += 1

def subgraph_byLinktype(node_embs_all, data, clusters, nlinktype, device):
    subgraphs = []
    for c in range(nlinktype):
        u = data.edge_index[0][clusters == c]
        v = data.edge_index[1][clusters == c]
        selected_nodes = torch.unique(torch.cat((u, v)))
        edge_index = torch.stack([u, v], dim=0).cpu()
        map_e = dict(zip(selected_nodes.cpu().numpy(), np.arange(len(selected_nodes))))
        edge_index.apply_(lambda x: map_e[x])
        subgraphs.append({'x': node_embs_all[selected_nodes].to(device), 'edge_index': edge_index.to(device),
                          'y': data.y[selected_nodes].to(device), 'train_mask': data.train_mask[selected_nodes].to(device),
                          'val_mask': data.val_mask[selected_nodes].to(device), 'test_mask': data.test_mask[selected_nodes].to(device)})
    return subgraphs


def _subgraph_byLinktype(data, clusters, nlinktype, device):
    subgraphs = [dgl.edge_subgraph(data, clusters==c).to(device) for c in range(nlinktype)]
    return subgraphs

## ------------------------- tracking -------------------------- ##
def _track_centers(groups, ids_clients):
    """ tracking local cluster centroids """
    g = {}
    g_clients = {}
    for (client_id, idx_cluster), idx_group in groups.items():
        # print((client_id, idx_cluster), idx_group)
        if idx_group not in g:
            g[idx_group] = []
            g_clients[idx_group] = []
        g[idx_group].append(ids_clients[client_id].centroids[idx_cluster].clone().cpu().numpy())
        g_clients[idx_group].append(client_id)
    # for k, v in g.items():
    #     print([vv[:5] for vv in v])
    return (g, g_clients)

def _track_kGradNorms(client, nlinktype):
    """ tracking clients' gradient norms of each branch (cluster) """
    gradNorms = []
    for k in range(nlinktype):
        grads = [value.grad.flatten() for key, value in client.Ws.items() if key.startswith(f'split_gnns.{k}')]
        gradNorms.append(torch.norm(torch.cat(grads)).item())
    return gradNorms

def _track_kGradDistances(ids_clients, group):
    """ tracking the cosine distance between a client's gradients and the server's aggregagted gradients """
    cos_dists = {}
    local_grads = []
    cids = []
    for (client_id, idx_cluster) in group:
        if ids_clients[client_id].cluster_train_size[idx_cluster] != 0:
            local_grads.append(torch.cat([value.grad.flatten() for key, value in ids_clients[client_id].Ws.items() if key.startswith(f'split_gnns.{idx_cluster}')]))
            cids.append(client_id)
        else:
            cos_dists[client_id] = 0.
    if len(local_grads) != 0:
        avg_grad = torch.mean(torch.stack(local_grads, dim=0), dim=0).reshape(1, -1)
        for i, cid in enumerate(cids):
            cos_dists[cid] = 1. - F.cosine_similarity(avg_grad, local_grads[i].reshape(1, -1)).item()
    return cos_dists
## ------------------------------------------------------------- ##

## ------------------------ outputing -------------------------- ##
def _output_results(df_out, idx, loss, metrics, mask):
    df_out.loc[idx, f'{mask}_loss'] = loss
    for mname, metric in metrics.items():
        df_out.loc[idx, f'{mask}_{mname}'] = metric

def _to_wandb(kname, r, loss, metrics):
    if r is not None:
        wandb.log({f'{kname}_loss': loss, 'round': r})
        for mname, metric in metrics.items():
            wandb.log({f'{kname}_{mname}': metric, 'round': r})
    else:
        wandb.log({f'{kname}_loss': loss})
        for mname, metric in metrics.items():
            wandb.log({f'{kname}_{mname}': metric})
## ------------------------------------------------------------- ##

## ------------------ for node classification ------------------ ##
def loss_nll(pred, label):
    return F.nll_loss(F.log_softmax(pred, dim=1), label)

def loss_ce(pred, label):
    return F.cross_entropy(pred, label)

def accuracy(data, logits, mask):
    acc = logits[data[mask]].max(dim=1)[1].eq(
        data.y[data[mask]]).sum().item() * 1. / len(data.y[data[mask]])
    return acc

def accuracy_dgl(pred, label):
    acc = pred.max(dim=1)[1].eq(label).sum().item() * 1. / len(label)
    return acc

def f1_micro(pred, label):
    return f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='micro')
## ------------------------------------------------------------- ##

## -------------------- for node regression -------------------- ##
def loss_rmse(pred, label):
    return torch.sqrt(F.mse_loss(pred.squeeze(), label) + 1e-8)

def metric_rmsle(pred, label):
    return torch.sqrt(F.mse_loss(torch.log(pred.squeeze() + 1), torch.log(label + 1)) + 1e-8).item()

def metric_mae(pred, label):
    return F.l1_loss(pred.squeeze(), label).item()

def metric_r2(pred, label):
    return r2_score(label.cpu().numpy(), pred.cpu().numpy())
## ------------------------------------------------------------- ##

## ------------ for node multi-class classification ------------ ##
def loss_bce(pred, target, pos_weight=None):
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight).item()

def exact_match_ratio(pred, target):
    return (pred > 0.5).eq(target).sum(axis=1).true_divide(target.size()[1]).sum().true_divide(target.size()[0])
## ------------------------------------------------------------- ##

# metrics: evaluate for each label (vertical); micro/macro to get rid of unbalance
