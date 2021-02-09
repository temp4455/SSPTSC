import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import numpy as np

def clones(moudel, N):
    return nn.ModuleList([copy.deepcopy(moudel) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.view(batch_size, channels * seq_len)
        # x1: [batch, channel, seq_len]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2).view(batch_size, channels, seq_len)


def attention(query: torch.Tensor, key: torch.Tensor, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    # p_attn attention
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class BasicLayer(nn.Module):
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]

    def __init__(self, in_channels, planes, stride,global_index_num=100,attention_window=20, downsample=True, dropout=0.5,
                 kernel_sizes=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]):
        super(BasicLayer, self).__init__()
        self.d_k = planes

        self.kernel_sizes = kernel_sizes
        self.h = len(self.kernel_sizes)
        self.d_model = self.d_k * self.h

        self.attn = None
        self.in_channels = in_channels
        self.models = nn.ModuleList([
            clones(nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=planes // 4, kernel_size=1, bias=False),

                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes // 4),

                nn.Conv1d(planes // 4, planes, kernel_size, stride, (kernel_size - 1) // 2, bias=False),

                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes),

                nn.Conv1d(planes, planes, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes),

                # nn.Conv2d(in_channels=in_channels,out_channels=planes,)
            ), 3) for kernel_size in self.kernel_sizes
        ])

        self.linears = clones(nn.Linear(self.d_model, self.d_model), 3)
        self.dropout_p = dropout

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # self.sublayer= SublayerConnection(self.d_model,dropout)
        self.sublayer_bn = nn.BatchNorm1d(in_channels)

        # self.sublayer_bn =LayerNorm(batch_size)
        self.sublayer_dropout = nn.Dropout(p=dropout)

        # self.attention_mask= torch.zeros(())
        self.global_index_num=global_index_num
        self.attention_window=attention_window

    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        # if self.in_channels>1:
        #     x1= self.sublayer_bn(x)
        #     # x1=self.sublayer_bn(x.permute(0,2,1)).permute(0,2,1)
        # else:
        #     x1=x
        nbatches,channels,seq_len= x.shape
        attention_mask= x.new_zeros((nbatches,seq_len))
        # global_index_num= 100
        if self.global_index_num>0:
            interval = seq_len// self.global_index_num
            attention_mask[:,::interval]=1
        x1 = self.sublayer_bn(x)


        # x1:[batch, channel, seq_len ]->  [batch,seq_len,channel] -> [batch,seq_len,h,d_model//h]
        query, key, value = [
            torch.cat([self.models[i][j](x1) for i in range(len(self.kernel_sizes))], dim=1).permute(0, 2, 1).view(
                nbatches, -1, self.h, self.d_k).permute(0, 2, 1, 3) for j in range(3)]

        # query,key,value=[torch.cat([self.bottleneck[i][j](x) for i in range(len(self.kernel_sizes))],dim=1).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for j in range(3)]
        # calculate q,k,v : [batch,h,seq_len,d_model//h]-> out:[batch,h,seq_len,d_model//h],atten:[batch,h,seq_len,seq_len]
        # out, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        out,self.attn =attention(query,key,value,mask=mask,dropout=self.dropout)
        # batch_size * h * len * hidden_dim
        # out=out.transpose(1,2).contiguous().view(nbatches,-1,self.d_model)
        # out : [batch, channel,seq_len ]
        out = out.permute(0, 1, 3, 2).contiguous().view(nbatches, self.d_model, -1)
        if x1.shape[1] == out.shape[1]:
            return x + self.sublayer_dropout(out)
        else:
            return out


class TSEncoder(nn.Module):
    def __init__(self,global_index_num=100,attention_window=20):
        super(TSEncoder, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1,global_index_num,attention_window ),
                                    BasicLayer(self.hidden*self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window))
        self.bn = nn.BatchNorm1d(self.hidden*self.kernel_num)
        self.conv1d = nn.Conv1d(self.hidden*self.kernel_num, self.hidden, 1, 1)
        self.out_bn= nn.BatchNorm1d(self.hidden)
        self.out = nn.Conv1d(self.hidden , 1, 1, 1)


    def forward(self, x):
        # x:[batch,channel,seq_len,]-> x:[batch,channel,seq_len]
        x = self.layers(x)
        # return :[batch,seq_len]
        return self.out(self.out_bn(torch.relu(self.conv1d(self.bn(x))))).squeeze()
        # return self.conv1d(x).squeeze()
        




class TSEncoderDTWED(nn.Module):
    def __init__(self,global_index_num=100,attention_window=20,similar_matrix=None):
        super(TSEncoderDTWED, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1,global_index_num,attention_window ),
                                    BasicLayer(self.hidden*self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window))
        self.bn = nn.BatchNorm1d(self.hidden*self.kernel_num)
        self.conv1d = nn.Conv1d(self.hidden*self.kernel_num, self.hidden, 1, 1)
        self.similar_matrix= similar_matrix
        # self.out_bn= nn.BatchNorm1d(self.hidden)
        # self.out = nn.Conv1d(self.hidden , 1, 1, 1)


    def forward(self, x):
        batch_size=x.shape[0]
        # x:[batch,channel,seq_len,]-> x:[batch,channel,seq_len]
        x = self.layers(x)
        # return :[batch,hidden*seq_len]
        x=self.conv1d(self.bn(x)).reshape(batch_size,-1)
        # idx=np.random.randint(0, batch_size)
        # res=torch.cat([torch.matmul(x[i],x[]) for i in range(batch_size)],dim=0)
        res=[]
        for i in range(batch_size):
            index=torch.ones(batch_size,dtype=torch.long)
            index[i]=0
            b= x[index==1,:].transpose(-2,-1)
            a=x[i,:].reshape(1,-1)
            res.append(torch.matmul(a,b))
            # res.append(torch.softmax(torch.matmul(a,b),dim=1))
        res = torch.cat(res,dim=0)
        return res,x
        # return res/torch.sum(res,dim=1).unsqueeze(-1)
        # return self.out(self.out_bn(torch.relu(self.conv1d(self.bn(x))))).squeeze()
        # return self.conv1d(x).squeeze()


class TSClassifcation(nn.Module):
    def __init__(self, nclass,global_index_num=100,attention_window=20):
        super(TSClassifcation, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1, global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window))
        self.dropout1 = nn.Dropout(0.5)
        self.conv1d_new = nn.Conv1d(self.hidden * self.kernel_num, nclass, 3, 1, 1)
        self.dropout2 = nn.Dropout(0.5)
        self.adapt_pool = nn.AdaptiveAvgPool1d(1)
        # self.adapt_pool = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(nclass, nclass, 1, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv1d_new(self.dropout1(x))
        # out1:[batch,nclass,seq_len]
        # out1= torch.softmax(x,dim=1)
        # out2 :[batch,nclass,1]
        out2 = self.out(self.adapt_pool(torch.relu(self.dropout2(x))))
        return x, out2.squeeze(-1)
        # out2= torch.softmax(self.out(self.adapt_pool(torch.relu(x))),dim=1)
        # return out1,out2.squeeze()


if __name__ == '__main__':
    # batch_size=2
    # channels=1
    # seq_len =1024
    # x= torch.randn(batch_size,channels,seq_len)
    # encoder= TSEncoder()
    # device= torch.device('cuda:0')
    # x=x.to(device)
    # encoder.to(device)
    # b=encoder(x)
    # print(b.shape)
    # print(b)

    # for TSClassifcation
    batch_size = 64
    channels = 1
    seq_len = 1000
    x = torch.randn(batch_size, channels, seq_len)
    # encoder = TSEncoder()
    model = TSClassifcation(3)
    device = torch.device('cuda:0')
    x = x.to(device)
    model.to(device)

    out1, out2 = model(x)
    print(out1.shape)
    print(out2.shape)
    # print(out2)

# class Decoder(nn.Module):
#     def __init__(self,layer,N):
#         super(DecoderLayer, self).__init__()
#         self.layers= clones(layer,N)
#         self.norm= LayerNorm(layer.size)
#     def forward(self, x,memory,src_mask,tgt_mask):
#         for layer in self.layers:
#             x= layer(x,memory,src_mask,tgt_mask)
#         return self.norm(x)
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self,):
#         super(DecoderLayer, self).__init__()
#
