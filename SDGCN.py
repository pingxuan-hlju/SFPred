
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class Graphsn_GCN(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Graphsn_GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.eps = nn.Parameter(torch.FloatTensor(1))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.95 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv_eps = 0.21 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj,x):

        v = (self.eps ) *torch.diag(adj)
        mask = torch.diag(torch.ones_like(v))
        adj = mask*torch.diag(v) + (1. - mask)*adj

        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, dropout):
        super(GNN, self).__init__()

        self.gc1 = Graphsn_GCN(in_feat, hidden_feat)
        self.gc2 = Graphsn_GCN(hidden_feat, out_feat)
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.act = nn.Sigmoid()
        self.fc1 = nn.Sequential(
            nn.Linear(1134, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 16),
            nn.Dropout(0.5),
            nn.Linear(16, 1)

        )
        self.fc2 = nn.Sequential(
            nn.Linear(1134, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 16),
            nn.Dropout(0.5),
            nn.Linear(16, 1)

        )
        self.fea_fc=nn.Sequential(
            nn.Linear(2268,256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.gnn_fc = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc[0].weight, nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.fc[3].weight, nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.fc[3].weight)
        nn.init.xavier_normal_(self.fc1[0].weight)
        nn.init.xavier_normal_(self.fc1[2].weight)
        nn.init.xavier_normal_(self.fc1[4].weight)
        nn.init.xavier_normal_(self.fc2[0].weight)
        nn.init.xavier_normal_(self.fc2[2].weight)
        nn.init.xavier_normal_(self.fc2[4].weight)
        nn.init.xavier_normal_(self.fea_fc[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.gnn_fc[0].weight, nn.init.calculate_gain('relu'))

    def forward(self, left,right, adj,x):
        A = self.fc1(x)
        B = self.fc2(x)
        attr_matrix = A + torch.t(B)
        attr_matrix = self.act(attr_matrix)
        adj = self.a * adj + self.b * attr_matrix

        x1 = F.relu(self.gc1(adj,x))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = self.gc2(adj,x1)
        emb1 = torch.cat([x[left],x[right]],dim=1)
        emb_fea = self.fea_fc(emb1)

        emb_gnn =torch.cat([x1,x2],dim=1)
        emb_gnn=torch.cat([emb_gnn[left],emb_gnn[right]],dim=1)

        emb_gnn = self.gnn_fc(emb_gnn)


        emb =torch.cat([emb_fea,emb_gnn],dim=1)

        x = self.fc(emb)
        return x

        # return F.log_softmax(x, dim=-1)