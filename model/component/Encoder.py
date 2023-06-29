import torch
import numpy as np
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25,linear=0):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        if linear==1:
            self.w1 = nn.Linear(self.l_size, self.l_size)
        else:
            self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        
        if linear==1:
            self.w2 = nn.Linear(self.l_size, self.l_size)
        else:
            self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCBlock(nn.Module):

    def __init__(self, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.stage_num = 3
        self.p_dropout = 0.1
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        self.fc_2 = nn.Conv1d(self.linear_size, channel_out, kernel_size=1)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)
        return x
    
class Simplemlp(nn.Module):

    def __init__(self, channel_in, linear_size):
        super(Simplemlp, self).__init__()

        self.channel_in = channel_in
        self.linear_size = linear_size
        
        self.p_dropout = 0.1
        self.fc_1 = nn.Conv1d(self.channel_in, self.linear_size, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Encoder2D(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.mlp = FCBlock(channel_in=2, 
                               channel_out=args.channel, linear_size=args.channel,
                               block_num=3,linear = 0)
    def forward(self, x):
        # input 3D (batchsize,2* joints_num, frame)
        out = self.mlp(x)
        # output 3D (batchsize, 128, frame)
        return out
    
    
    
class Encoder3D(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.mlp = FCBlock(channel_in=3, 
                            channel_out=args.channel, linear_size=args.channel,
                            block_num=3,linear = 0)
    def forward(self, x):
        # input 3D (batchsize,3* joints_num, frame)
        out = self.mlp(x)
        # output 3D (batchsize,128, frame)
        return out