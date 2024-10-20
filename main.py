#导包
import argparse
import torch
import numpy as np
import torch.nn as nn
from scipy.linalg import fractional_matrix_power
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import networkx as nx
import scipy.sparse as sp
from torch.autograd import Variable
from model.Model import model




def load_data():
    mm = torch.tensor(np.loadtxt("data/mi_sim.txt"))
    dd = torch.tensor(np.loadtxt("data/dis_sim_T.txt"))
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


def train(model,train_set,test_set,fea,adj,tei,epoch,learn_rate):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate,weight_decay=0.0005)
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

def run_model(args):

    mm,dd,md = load_data()
    fea,A_array = construct_fea_adj(mm,dd,md)

    '''生成子图拓扑，计算拓扑系数矩阵'''
    adj = sp.coo_matrix((torch.ones(len(A_array.nonzero()[:, 0])), (A_array.nonzero()[:, 0], A_array.nonzero()[:, 1])),shape=(1134,1134))
    G = nx.from_numpy_matrix(adj.toarray())
    nx.set_node_attributes(G, fea, "attr_name")

    sub_graphs = []
    for i in np.arange(A_array.shape[0]):
        s_indexes = []
        for j in np.arange(A_array.shape[1]):
            s_indexes.append(i)
            if (A_array[i][j] == 1):
                s_indexes.append(j)
        sub_graphs.append(G.subgraph(s_indexes))
    subgraph_nodes_list = []

    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))
    sub_graphs_adj = []
    for index in np.arange(len(sub_graphs)):
        print(index)
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
    new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
    for node in np.arange(len(subgraph_nodes_list)):
            # print(node)
        sub_adj = sub_graphs_adj[node]
        nodes_list = subgraph_nodes_list[node]
        c_neighbors = np.intersect1d(nodes_list, np.concatenate(subgraph_nodes_list))
            # print(node,c_neighbors)

        c_neighbors_indices = [nodes_list.index(index) for index in c_neighbors]

        sub_adj_selected = sub_adj[:, c_neighbors_indices]
        sub_adj_selected = sub_adj_selected[c_neighbors_indices, :]

        count = torch.sum(torch.tensor(sub_adj_selected))

            # new_adj[node, c_neighbors] = count / 2
            # print("c_neighbors",c_neighbors)
        new_adj[node, c_neighbors] = count.float() / 2

        new_adj[node, c_neighbors] /= len(c_neighbors) * (len(c_neighbors) - 1)
        new_adj[node, c_neighbors] *= len(c_neighbors) ** 1
        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)
        weight = weight + torch.FloatTensor(A_array.float())
        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag(coeff.T[0])
        weight = weight + coeff
        weight = weight.detach().cpu().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.diag(row_sum + 1)

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)
        adj = torch.FloatTensor(A_tilde_hat)
    adj = torch.tensor(adj)
    ti = torch.argwhere(md>-1)
    trset=DataLoader(MyDataset(ti,md),args.train_bt,shuffle=True)
    teset=DataLoader(MyDataset(ti,md),args.test_bt,shuffle=False)
    train_set,test_set=[],[]
    for x1,x2,y in trset:
        train_set.append((x1,x2,y))
    for x1,x2,y in teset:
        test_set.append((x1,x2,y))
    torch.save(train_set,"data/train_set.pth")
    torch.save(test_set,"data/test_set.pth")
    torch.save([ti,fea,adj],"data/par.pth")
    ti,fea,adj=torch.load('data/par.pth')
    train_set=torch.load('data/train_set.pth')
    test_set=torch.load('data/test_set.pth')
    net=model(args.input_dim, args.hidden_dim,args.output_dim, args.dropout)
    train(net,train_set,test_set,fea,adj,ti,epoch=args.epoch,learn_rate=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFPred")
    parser.add_argument('--epoch', type=int, default=100, help='batch size for training')
    parser.add_argument('--input_dim', type=int, default=1134, help='batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=512, help='batch size for training')
    parser.add_argument('--output_dim', type=int, default=256, help='batch size for training')
    parser.add_argument('--dropout', type=float, default=0.9, help='batch size for training')
    parser.add_argument('--train_bt', type=int, default=128, help='batch size for training')
    parser.add_argument('--test_bt', type=int, default=500, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='batch size for training')
    args = parser.parse_args()
    run_model(args)