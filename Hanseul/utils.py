import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def bin_to_int(x, num_col=None):
    """
    Given boolean vectors of recipes, create a matrix of integers.  
    Input: (batch_size, num_ingreds), Output: (batch_size, ?)  
    Each row should contain ingredient IDs for each recipe and padding indices.  
    For example, given
    [[0,0,1,0,1,1,1], [0,1,0,1,0,0,0]] ,
    since the number of columns is 7, padder outputs
    [[2,4,5,6], [1,3,7,7]].
    """
    batch_size, num_ingreds = x.size()
    if num_col is None:
        num_col = x.sum(1).long().max()
    out = torch.full((batch_size, num_col), num_ingreds).to(x.device)
    for i in range(batch_size):
        out[i][:x[i].sum().long()] = torch.arange(num_ingreds)[x[i]==1]
    return out

def _concatenate(running_v, new_v):
    if running_v is not None:
        return np.concatenate((running_v, new_v.clone().detach().cpu().numpy()), axis=0)
    else:
        return new_v.clone().detach().cpu().numpy()

