#导包
import pandas as pd
import torch
import math
import numpy as np
from numpy import int64
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import networkx as nx
import scipy.sparse as sp


from torch.autograd import Variable
def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc




def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes,in_feat,out_feat, kernel_att=1, head=4, kernel_conv=1, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)
        self.layer_norm, self.act = nn.LayerNorm(in_feat), nn.GELU()

        # self.padding_att = (self.dilation * (self.kernel_att-1) + 1) // 2
        # self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.pad_att = nn.ReflectionPad2d((0, 0, 0, 0))

        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)
        self.mlp1 = nn.Linear(in_planes,out_planes)
        self.mlp2 = nn.Linear(in_feat,out_feat)
        self.layer = nn.Linear(in_feat,out_feat)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.layer.weight)
        nn.init.xavier_normal_(self.mlp1.weight)
        nn.init.xavier_normal_(self.mlp2.weight)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)
        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
        out_res = self.act(self.mlp1(self.layer_norm(x).transpose(1,3)).transpose(1,3))
        out_res1 = F.relu(self.mlp2(out_res))
        out = out_att+out_res
        h = F.relu(self.layer(self.dropout(out)))
        return h+out_res1
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(16*64, 256),# 全连接层
                        nn.Dropout(0.5),
                        nn.ReLU(),
                        # nn.Linear(256, 2) # 全连接层
                      )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc[0].weight,nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.fc[0].weight)


    def forward(self,x):
        x= self.fc(x.view(x.shape[0], -1))
        return x
class CNN_ATT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ACmix(in_planes=2,out_planes=64,in_feat=1134,out_feat=256)  #卷积+att一次结果
        self.layer2 = ACmix(in_planes=64,out_planes=16,in_feat=256,out_feat=64)
        self.mlp =MLP()

    def forward(self, left,right,feature):
        left_emb = feature[left]
        left_emb =left_emb.unsqueeze(1).unsqueeze(1)
        right_emb = feature[right]
        right_emb =right_emb.unsqueeze(1).unsqueeze(1)

        h = torch.cat([left_emb,right_emb],dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.mlp(h)
        return h
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
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
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
class model(nn.Module):
    def __init__(self,in_feat,hidden_feat,out_feat,dropout):
        super().__init__()
        self.layer1 =GNN(in_feat,hidden_feat,out_feat,dropout)
        self.layer2 = CNN_ATT()
        self.fc = nn.Sequential(
            nn.Linear(384,64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64,2)

        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc[0].weight, nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.fc[3].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[3].weight)
    def forward(self,left,right,adj,feature):
        h1 = self.layer1(left,right,adj,feature)
        h2 = self.layer2(left,right,feature)
        #h2是合并后的
        h2 = torch.cat([h1,h2],dim=1)
        h2 = self.fc(h2)
        return h2


def load_data():
    mm = torch.tensor(np.loadtxt("data/miRNA_sim.txt"))
    dd = torch.tensor(np.loadtxt("data/Dis_Sim_T.txt"))
    md = torch.tensor(np.loadtxt("data/mi_dis.txt"))
    return mm,dd,md
def construct_fea_adj(mm,dd,md):
    fea =torch.cat([torch.cat([mm,md],dim=1),torch.cat([md.T,dd],dim=1)],dim=0)
    mm[mm > 0] = 1
    dd[dd > 0] = 1
    adj =torch.cat([torch.cat([mm,md],dim=1),torch.cat([md.T,dd],dim=1)],dim=0)
    return fea,adj
class MyDataset(Dataset):
    def __init__(self,tri,md):
        self.tri=tri
        self.md=md
    def __getitem__(self,idx):
        x,y=self.tri[idx,:]
        label=self.md[x][y]
        return x,y,label
    def __len__(self):
        return self.tri.shape[0]

# adj = sp.coo_matrix((torch.ones(len(A_array.nonzero()[:, 0])), (A_array.nonzero()[:, 0], A_array.nonzero()[:, 1])),shape=(1134,1134))
# G = nx.from_numpy_matrix(adj.toarray())
# nx.set_node_attributes(G, fea, "attr_name")
#
# sub_graphs = []
# for i in np.arange(A_array.shape[0]):
#     s_indexes = []
#     for j in np.arange(A_array.shape[1]):
#         s_indexes.append(i)
#         if (A_array[i][j] == 1):
#             s_indexes.append(j)
#     sub_graphs.append(G.subgraph(s_indexes))
# subgraph_nodes_list = []
#
# for i in np.arange(len(sub_graphs)):
#     print(i)
#     subgraph_nodes_list.append(list(sub_graphs[i].nodes))
# sub_graphs_adj = []
# for index in np.arange(len(sub_graphs)):
#     print(index)
#     sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
# new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
# for node in np.arange(len(subgraph_nodes_list)):
#         # print(node)
#     sub_adj = sub_graphs_adj[node]
#     nodes_list = subgraph_nodes_list[node]
#     c_neighbors = np.intersect1d(nodes_list, np.concatenate(subgraph_nodes_list))
#         # print(node,c_neighbors)
#
#     c_neighbors_indices = [nodes_list.index(index) for index in c_neighbors]
#
#     sub_adj_selected = sub_adj[:, c_neighbors_indices]
#     sub_adj_selected = sub_adj_selected[c_neighbors_indices, :]
#
#     count = torch.sum(torch.tensor(sub_adj_selected))
#
#         # new_adj[node, c_neighbors] = count / 2
#         # print("c_neighbors",c_neighbors)
#     new_adj[node, c_neighbors] = count.float() / 2
#
#     new_adj[node, c_neighbors] /= len(c_neighbors) * (len(c_neighbors) - 1)
#     new_adj[node, c_neighbors] *= len(c_neighbors) ** 1
#     print("生成图成功")
#     print("循环结束")
#     weight = torch.FloatTensor(new_adj)
#     weight = weight / weight.sum(1, keepdim=True)
#     weight = weight + torch.FloatTensor(A_array.float())
#     coeff = weight.sum(1, keepdim=True)
#     coeff = torch.diag(coeff.T[0])
#     weight = weight + coeff
#     weight = weight.detach().cpu().numpy()
#     weight = np.nan_to_num(weight, nan=0)
#
#     row_sum = np.array(np.sum(weight, axis=1))
#     degree_matrix = np.diag(row_sum + 1)
#
#     D = fractional_matrix_power(degree_matrix, -0.5)
#     A_tilde_hat = D.dot(weight).dot(D)
#     adj = torch.FloatTensor(A_tilde_hat)
#     # A_tilde_hat = D.dot(torch.FloatTensor(weight)).to(device).dot(D)
#     # adj = torch.FloatTensor(A_tilde_hat).to(device)
#
#
#
# # np.savetxt('result/adj_matrix.txt', adj.cpu().numpy(), fmt='%f')
# adj = torch.tensor(adj)
# ti = torch.argwhere(md>-1)
# trset=DataLoader(MyDataset(ti,md),batch,shuffle=True)
# teset=DataLoader(MyDataset(ti,md),batch,shuffle=False)
# train_set,test_set=[],[]
# for x1,x2,y in trset:
#     train_set.append((x1,x2,y))
# for x1,x2,y in teset:
#     test_set.append((x1,x2,y))
# torch.save(train_set,"data/train_set.pth")
# torch.save(test_set,"data/test_set.pth")
# torch.save([ti,fea,adj],"data/par.pth")
def train(model,train_set,test_set,fea,adj,tei,epoch,learn_rate):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate,weight_decay=0.001)
    cost=nn.CrossEntropyLoss()
    model.train()
    fea,adj = fea.float(),adj.float()
    Amax = [0, 0]
    for i in range(epoch):
        print("第{}轮".format(i))
        for x1,x2,y in train_set:

            x1,x2,y=Variable(x1.long()),Variable(x2.long()),Variable(y.long())
            out=model(x1,x2,adj,fea)
            loss=cost(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (i+1)%1 == 0: #and i+1>=50:
            print(i)
            tacc(model,test_set,adj,fea,tei,Amax)
        #if i+1==epoch:
            #tacc(model,test_set,fea,G1,0,cros)
        torch.cuda.empty_cache()
def calculate_TPR_FPR(RD, f, B):
    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))
    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)
    for i in range(RD.shape[0]):
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]
    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]
    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    Pr = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, Pr
def tacc(model,tset,fea,G1,tei,Amax):
    predall,yall=torch.tensor([]),torch.tensor([])
    model.eval()
    for x1,x2,y in tset:
        x1,x2,y=Variable(x1.long()),Variable(x2.long()),Variable(y.long())
        pred=model(x1,x2,fea,G1).data
        predall=torch.cat([predall,torch.as_tensor(pred,device='cpu')],dim=0)
        yall=torch.cat([yall,torch.as_tensor(y,device='cpu')])
    pred=torch.softmax(predall,dim=1)[:,1]
    trh=torch.zeros(793,341)-1
    tlh=torch.zeros(793,341)-1
    trh[tei[:,0],tei[:,1]]=pred
    tlh[tei[:,0],tei[:,1]]=yall
    R=trh.numpy()
    label=tlh.numpy()
    f = np.zeros(shape=(R.shape[0], 1))
    for i in range(R.shape[0]):
        f[i] = np.sum(R[i] > -1)
    if min(f)>0:
        TPR,FPR,P=calculate_TPR_FPR(R,f,label)
        AUC=metrics.auc(FPR, TPR)
        AUPR=metrics.auc(TPR, P) + (TPR[0] * P[0])
        print("AUC:%.4f_AUPR:%.4f"%(AUC,AUPR))
        if AUPR>Amax[1]:
            Amax[0]=AUC
            Amax[1]=AUPR
            print("save")
            torch.save((predall,yall),"PandY")


batch = 500
mm,dd,md = load_data()
fea,A_array = construct_fea_adj(mm,dd,md)
ti,fea,adj=torch.load('data/par.pth')
train_set=torch.load('data/train_set.pth')
test_set=torch.load('data/test_set.pth')
net=model(1134,512,256,0.9)
train(net,train_set,test_set,fea,adj,ti,epoch=15,learn_rate=0.0001)

ti,_,_=torch.load('data/par.pth')
pred,_=torch.load('./PandY')
pred=torch.softmax(pred,dim=1)[:,1]
trh=torch.zeros(793,341)
trh[ti[:,0],ti[:,1]]=pred
miRNA_dis_score=trh
index = np.argsort(-miRNA_dis_score, axis=0)
miRNA_name=np.loadtxt("data/mi_name.txt",dtype=str)
dis_name=pd.read_csv("data/dis_name.txt",header=None,sep='\t')
miRNA_all_50 = index[:50,:]
miRNA_name_50 = miRNA_all_50.T
candidate_list = []
for i in range(miRNA_name_50.shape[0]): # 405
    for j in range(miRNA_name_50.shape[1]): #50
        candidate_list.append([dis_name[0][i],miRNA_name[miRNA_name_50[i][j]],j+1,miRNA_dis_score[miRNA_name_50[i][j]][i]])
result = open('data/ST4.csv', 'w', encoding='gbk')
result.write('Disease Name,Candidate miRNA name,Rank,Association score\n')
for m in range(len(candidate_list)): # 遍历的是405 * 50 的长度
    for n in range(len(candidate_list[m])): # 将每一行的元素求一下长度
        result.write(str(candidate_list[m][n])) # 一个一个的写入
        result.write(',') # 以 \t结束一行的写入
    result.write('\n') # 换行重新写
result.close() # 写完关闭文件