import torch
import torch.nn.functional as F

def make_one_hot(x):
    if type(x) is not torch.Tensor:
        x = torch.LongTensor(x)
    if x.dim() > 2:
        x = x.squeeze()
        if x.dim() > 2:
            return False
    elif x.dim() < 2:
        x = x.unsqueeze(0)
    return F.one_hot(x).sum(1)[:,:-1]