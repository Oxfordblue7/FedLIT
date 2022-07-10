import torch
import torch.nn.functional as F
from dgl.nn import GraphConv

import collections
import operator


class GCN(torch.nn.Module):
    def __init__(self, nfeature, nhidden, nclass, nlayer, dropout, seed):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.dropout = dropout

        self.feature1 = torch.nn.Linear(nfeature, nhidden)
        self.feature2 = torch.nn.Linear(nhidden, nhidden)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(nhidden, nhidden))
        for l in range(nlayer - 1):
            self.convs.append(GraphConv(nhidden, nhidden))
        self.classifier = torch.nn.Linear(nhidden, nclass)

    def forward(self, g):
        x = self.feature1(g.ndata['x'])
        x = self.feature2(x)
        x = self.convs[0](g, x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for l in range(1, len(self.convs)):
            x = self.convs[l](g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        return x


class multichannel_GCN(torch.nn.Module):
    def __init__(self, nlinktype, nfeature, nhidden, nclass, nlayer, dropout, seed):
        super(multichannel_GCN, self).__init__()
        torch.manual_seed(seed)
        self.dropout = dropout
        self.nhidden = nhidden

        self.feature1 = torch.nn.Linear(nfeature, nhidden) # shared 1st feature projection layer
        self.split_gnns = torch.nn.ModuleList()    # split GCNs
        for _ in range(nlinktype):
            feature2 = torch.nn.Linear(nhidden, nhidden)
            gcn = torch.nn.ModuleList()
            for l in range(nlayer):
                gcn.append(GraphConv(nhidden, nhidden))
            self.split_gnns.append(torch.nn.ModuleList([feature2, gcn]))
        self.classifier = torch.nn.Linear(nhidden, nclass)

    def feature_projection(self, x):
        x = self.feature1(x)
        return x    # for subgraphing

    def split_forward(self, subgraphs, num_nodes, device):
        # outs = []
        out = torch.zeros((num_nodes, self.nhidden)).to(device)
        # count = torch.zeros((num_nodes, 1)).to(device)

        for i, (feature2, gcn) in enumerate(self.split_gnns):
            x = feature2(subgraphs[i].ndata['x'])
            for l in range(len(gcn)):
                x = gcn[l](subgraphs[i], x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # outs.append(dict(zip(subgraphs[i].ndata['_ID'].cpu().numpy(), x)))
            # sum up the embeddings correspondingly
            out[subgraphs[i].ndata['_ID']] += x
            # count[subgraphs[i].ndata['_ID']] += 1.

        # # sum up the embeddings correspondingly
        # counter = collections.Counter()
        # for d in outs:
        #     counter.update(d)
        # out = torch.stack([v for k, v in sorted(dict(counter).items(), key=operator.itemgetter(0), reverse=False)], dim=0)

        # # average the embeddings correspondingly
        # out = out.true_divide(count)

        return out

    def split_forward_subgraph(self, idx_subg, subgraph):
        x = self.split_gnns[idx_subg][0](subgraph.ndata['x'])
        for l in range(len(self.split_gnns[idx_subg][1])):
            x = self.split_gnns[idx_subg][1][l](subgraph, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def split_forward_subgraph_batching(self, idx_subg, blocks):
        x = self.split_gnns[idx_subg][0](blocks[0].srcdata['x'])

        for l in range(len(self.split_gnns[idx_subg][1])):
            x = self.split_gnns[idx_subg][1][l](blocks[l], x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = self.classifier(x)
        return x

    def classify(self, x):
        x = self.classifier(x)
        return x
