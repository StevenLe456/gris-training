import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.DoubleTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).double())
            if bias:
                self.bias = Parameter(torch.DoubleTensor(out_features_v))
            else:
                self.register_parameter("bias", None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.DoubleTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).double())
            if bias:
                self.bias = Parameter(torch.DoubleTensor(out_features_e))
            else:
                self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            ret = []
            for he, hv, av, t in zip(H_e, H_v, adj_v, T):
                multiplier1 = torch.mm(torch.from_numpy(t.todense()), torch.diag((he @ self.p.t()).t()[0])) @ torch.from_numpy(t.todense()).t()
                mask1 = torch.eye(multiplier1.shape[0])
                M1 = mask1 * torch.ones(multiplier1.shape[0]) + (1. - mask1)*multiplier1
                adjusted_A = torch.mul(M1, torch.from_numpy(av.todense()))
                output = torch.mm(adjusted_A, torch.mm(hv, self.weight))
                if self.bias is not None:
                    retu = output + self.bias
                ret.append(retu)
            return ret, H_e
        else:
            ret = []
            for he, hv, ae, t in zip(H_e, H_v, adj_e, T):
                multiplier2 = torch.mm(torch.from_numpy(t.todense()).t(), torch.diag((hv @ self.p.t()).t()[0])) @ torch.from_numpy(t.todense())
                mask2 = torch.eye(multiplier2.shape[0])
                M3 = mask2 * torch.ones(multiplier2.shape[0]) + (1. - mask2)*multiplier2
                adjusted_A = torch.mul(M3, torch.from_numpy(ae.todense()))
                output = torch.mm(adjusted_A, torch.mm(he, self.weight))
                if self.bias is not None:
                    retu = output + self.bias
                ret.append(retu)
            return H_v, ret