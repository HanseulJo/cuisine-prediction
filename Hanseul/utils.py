import torch
import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(x):
    """ Convert int_data into bin_data, if needed. """
    if type(x) is not torch.Tensor:
        x = torch.LongTensor(x)
    if x.dim() > 2:
        x = x.squeeze()
        if x.dim() > 2:
            return False
    elif x.dim() < 2:
        x = x.unsqueeze(0)
    return F.one_hot(x).sum(1)[:,:-1]

class LogitSelector(nn.Module):
    """
    For each 3-tuple (output vector, label, data), choose 'important' model outputs.
    Specifically, choose a fixed number (e.g. rank = 100) of outputs consist of
    1) output for label index(=missing ingredient index),
    2) outputs for data indices(=given ingredient indices),  -- maybe unnecessary. can be included or not by 'contain_data' option.
    3) and several highest outputs for non-label indices.

    x: (batch, max_ingredient_num = 65 or 59), LongTensor
    output: (batch, num_items = 6714), FloatTensor
    labels: (batch, ), LongTensor
    rank: int
    """
    def __init__(self, rank=100, contain_data=False):
        super(LogitSelector, self).__init__()
        self.rank = rank
        self.contain_data=contain_data

    def forward(self, output, labels, x=None):
        num_items = output.size(1)
        if self.rank > num_items:
            raise ValueError
        target_indices = output.argsort(1)[:,-self.rank:]  # (batch, rank)
        label_where = (target_indices == labels.view(-1,1))  # target_indices의 각 batch마다 이미 label이 있으면 그 위치에 True
        no_label = torch.logical_not(label_where).all(dim=1)  # label 없는 batch에 대해 True
        yes_label, label_where = label_where.nonzero(as_tuple=True)  # label 있는 batch와 그 때 label의 위치를 long으로
        target_indices[no_label,0] = labels[no_label]  # label이 안 보였던 경우 맨 앞에 label 갖다 놓기
        if self.contain_data and x is not None:
            x_extended = F.pad(x, (1, self.rank-1-x.size(1)), 'constant', num_items)  # (batch, rank)
            target_indices[x_extended != num_items] = x_extended[x_extended != num_items]
        new_output = torch.gather(output, 1, target_indices) # (batch, rank)
        new_labels = torch.zeros_like(labels).long()  # label을 맨 앞에 갖다 놨음
        new_labels[yes_label] = label_where  # label이 이미 있었던 batch에 대해서만 수정
        return new_output, new_labels