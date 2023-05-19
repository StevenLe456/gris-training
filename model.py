from layers import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_list(l):
    li = []
    for lis in l:
        li.append(torch.sigmoid(lis))
    return li

def relu_list(l):
    li = []
    for lis in l:
        li.append(F.relu(lis))
    return li

class GCN(nn.Module):
    def __init__(self, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(512, 128, 512, 512, node_layer=True)
        self.gc2 = GraphConvolution(128, 128, 512, 128, node_layer=False)
        self.gc3 = GraphConvolution(128, 32, 128, 128, node_layer=True)
        self.gc4 = GraphConvolution(32, 32, 128, 32, node_layer=False)
        self.gc5 = GraphConvolution(32, 4, 32, 32, node_layer=True)
        self.gc6 = GraphConvolution(4, 4, 32, 4, node_layer=False)
        self.gc7 = GraphConvolution(4, 1, 4, 4, node_layer=True)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T):
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc1[0]), relu_list(gc1[1])
        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc2[0]), relu_list(gc2[1])
        gc3 = self.gc3(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc3[0]), relu_list(gc3[1])
        gc4 = self.gc4(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc4[0]), relu_list(gc4[1])
        gc5 = self.gc5(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc5[0]), relu_list(gc5[1])
        gc6 = self.gc6(X, Z, adj_e, adj_v, T)
        X, Z = relu_list(gc6[0]), relu_list(gc6[1])
        X, Z = self.gc7(X, Z, adj_e, adj_v, T)
        return sigmoid_list(X)