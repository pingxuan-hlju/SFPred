import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from SDGCN import SDGCN
from FCTransformer import FCTransformer

class model(nn.Module):
    def __init__(self,in_feat,hidden_feat,out_feat,dropout):
        super().__init__()
        self.layer1 =SDGCN(in_feat,hidden_feat,out_feat,dropout)
        self.layer2 = FCTransformer()
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