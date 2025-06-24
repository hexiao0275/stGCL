import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


# class Spatial_MGCN(nn.Module):
#     def __init__(self, nfeat, nhid1, nhid2, dropout):
#         super(Spatial_MGCN, self).__init__()
#         self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
#         self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
#         self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
#         self.ZINB = decoder(nfeat, nhid1, nhid2)
#         self.dropout = dropout
#         self.att = Attention(nhid2)
#         self.MLP = nn.Sequential(
#             nn.Linear(nhid2, nhid2)
#         )

#     def forward(self, x, sadj, fadj):
#         emb1 = self.SGCN(x, sadj)  # Spatial_GCN
#         com1 = self.CGCN(x, sadj)  # Co_GCN
#         com2 = self.CGCN(x, fadj)  # Co_GCN
#         emb2 = self.FGCN(x, fadj)  # Feature_GCN

#         emb = torch.stack([emb1, (com1 + com2) / 2, emb2], dim=1)
#         emb, att = self.att(emb)
#         emb = self.MLP(emb)

#         [pi, disp, mean] = self.ZINB(emb)
#         return com1, com2, emb, pi, disp, mean


# nhid1 = 128
# nhid2 = 64
# nfeat = fdim = 3000
    
# nhid1 = 3000
# nhid2 = 3000
# nfeat = fdim = 3000


class Spatial_MGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(Spatial_MGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        # self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2*2, nhid2)
        )

        d_model = 64
        n_heads = 4
        d_ff = 256
        n_layers = 2
        # Create TransformerEncoder model
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        self.transformer_2 = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        self.transformer_3 = TransformerEncoder(d_model*2, n_heads, d_ff, n_layers)
        # Forward pass

    def forward(self, x, sadj, fadj):

        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        # com1 = self.CGCN(x, sadj)  # Co_GCN
        # com2 = self.CGCN(x, fadj)  # Co_GCNS
        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        emb1_t = self.transformer (emb1)
        
        emb2_t = self.transformer_2 (emb2)
        
        all = torch.cat([emb1_t, emb2_t], dim=1)

        all_t  = self.transformer_3 (all) + all
        
        all_emb = torch.cat([all_t, emb1_t, emb2_t], dim=1)

    
        all_emb = torch.cat([emb1, emb2], dim=1)
        emb = self.MLP(all_emb)

        [pi, disp, mean] = self.ZINB(emb)
         
        # kk = all

        return emb1, emb2, emb, pi, disp, mean

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = self.norm1(attn_output + x)
        ff_output = self.feedforward(attn_output)
        return self.norm2(ff_output + attn_output)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
