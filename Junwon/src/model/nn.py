import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_size, output_size, fc_layer_sizes, dropout):
        super(DNN, self).__init__()
        assert isinstance(fc_layer_sizes, list) and len(fc_layer_sizes) >= 4
        assert dropout > 0 and dropout < 1
        self.input_size = input_size
        self.output_size = output_size
        feature_list = [input_size]+fc_layer_sizes+[output_size]
        self.feature_list = feature_list
        self.FCs = []
        for idx, features in enumerate(zip(feature_list[:-1], feature_list[1:])):
            in_features, out_features = features
            exec(f'self.FC{idx+1} = nn.Linear({in_features}, {out_features})')
            exec(f'self.FCs.append(self.FC{idx+1})')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float()
        for i in range(len(self.FCs)-1):
            x = F.leaky_relu(self.FCs[i](x))
            x = self.dropout(x)
        x = self.FCs[-1](x)
        return x

    def inference(self, x):
        pass