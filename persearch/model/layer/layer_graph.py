import torch
import torch.nn.functional as F

from .gcn_modified import GCNMConv
from .gat_modified import GATConv


class GcnNet(torch.nn.Module):
    def __init__(self, d_inp, d_outp, d_hidden=64, a=0.5, drop=0.5, k=2):
        super(GcnNet, self).__init__()
        self.conv1 = GCNMConv(d_inp, d_hidden, improved=True)
        self.conv2 = GCNMConv(d_hidden, d_hidden, improved=True)
        self.conv3 = GCNMConv(d_hidden, d_hidden, improved=True)
        self.conv4 = GCNMConv(d_hidden, d_hidden, improved=True)
        self.k = k
        self.d_outp = d_outp
        self.a_self = a
        self.beta_self = 0.1
        self.drop = drop

    def forward(self, x, edge_index):
        x_conv = self.conv1(x, edge_index)
        x_conv = F.dropout(x_conv, p=self.drop, training=self.training)
        x_conv = self.conv2(self.beta_self * x + (1 - self.beta_self) * x_conv, edge_index)
        x_conv = self.conv3(self.beta_self * x + (1 - self.beta_self) * x_conv, edge_index)
        x_conv = self.conv4(self.beta_self * x + (1 - self.beta_self) * x_conv, edge_index)
        return self.a_self * x + (1 - self.a_self) * x_conv


class GatNet(torch.nn.Module):
    def __init__(self, d_inp, d_outp, dropout=0.2):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(d_inp, d_inp, dropout=dropout, bias=False)
        self.conv2 = GATConv(d_inp, d_outp, dropout=dropout, bias=False)

    def forward(self, x, edges):
        x_conv = self.conv1(x, edges)
        x_conv = F.dropout(x_conv, p=0.2, training=self.training)
        h = self.conv2(x_conv, edges)
        return h
