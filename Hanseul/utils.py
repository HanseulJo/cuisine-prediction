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


def get_variables(x, pad_idx=None, random_remove=True, random_mask=False):
    device = x.device
    if pad_idx is None:
        pad_idx = x.size(1)
    
    # remove an ingredient from each recipe.
    if random_remove:
        feasible = x.sum(1) > 1
        int_x = bin_to_int(x[feasible])
        batch_size = int_x.size(0)
    
        pad_mask = (int_x == pad_idx)

        _rand = torch.rand(int_x.size()).to(device)
        _rand[pad_mask] = -1
        masked_ingred = torch.argmax(_rand, 1)

        token_mask = torch.full(int_x.size(), False).to(device)
        token_mask[torch.arange(batch_size), masked_ingred] = True

        label = int_x[token_mask].clone().detach()
        
        if random_mask:
            how_mask = torch.rand(int_x.size()).to(device)
            int_x[token_mask * (how_mask<0.8)] = pad_idx
            int_x[token_mask * (how_mask>0.9)] = torch.randint(pad_idx, int_x.size())[token_mask * (how_mask>0.9)].to(x.device)
        else:
            int_x[token_mask] = pad_idx
        
        return int_x, pad_mask, token_mask.view(-1), label

    # add additional ingredient 'MASK' to each recipe.
    else:
        int_x = bin_to_int(x)
        batch_size, num_ingreds = int_x.size()
        int_x = torch.cat([int_x, torch.full((batch_size,1), pad_idx).to(device)], 1)
    
        pad_mask = int_x == pad_idx
        pad_mask[:,-1] = False
        
        token_mask = torch.full((batch_size, num_ingreds+1), False).to(device)
        token_mask[:, -1] = True

        return int_x, pad_mask, token_mask.view(-1)