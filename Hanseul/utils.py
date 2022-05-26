import random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fix_seed(seed=42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    

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


def get_variables(x, pad_idx=None, complete=True, phase='train', cpl_scheme='pooled', mask_scheme=None):
    """
    Inputs:
    - x: binary vectors of recipes. Size: (batchsize, 6714)
    
    Outputs:
    - int_x: recipes in integers. + paddings with `pad_idx'. Size: (batchsize, ?)
    - feasible: boolean vector. True if the corresponding recipe can be used for prediction.
    - pad_mask: boolean matrix. True if we should ignore that position in encoder. Size: same as int_x
    - token_mask: (optional) boolean matrix. True if we should infer the missing recipe at that position. Size: same as int_x
    - labels_cpl: (optional) the answers for recipe completions, associated with token_mask. Size: (batchsize, )
    """    
    int_x = bin_to_int(x)
    feasible = x.sum(1) > 0  # all True
    pad_mask = token_mask = labels_cpl = None

    device = x.device
    if pad_idx is None:
        pad_idx = x.size(1)

    if complete:
        assert cpl_scheme in ['pooled', 'encoded']

        if cpl_scheme == 'pooled' and phase == 'train':  # randomly drop an ingredient for each row
            int_x, feasible, labels_cpl = _drop_one(x, int_x, pad_idx, device)

        elif cpl_scheme == 'encoded':
            if phase == 'train':  # randomly change an ingredient into [MASK] token (= 6714)
                int_x, feasible, pad_mask, token_mask, labels_cpl = _mask_one(x, int_x, pad_idx, device)

            else:  # add additional [MASK] token to each recipe.
                batch_size, num_ingreds = int_x.size()
                int_x = torch.cat([int_x, torch.full((batch_size,1), pad_idx).to(device)], 1)
                pad_mask = (int_x == pad_idx)
                pad_mask[:,-1] = False
                token_mask = torch.full((batch_size, num_ingreds+1), False).to(device)
                token_mask[:, -1] = True
    
    elif phase == 'train':  # classification only
        if mask_scheme == 'drop_one':  # randomly drop an ingredient for each row
            int_x, feasible, _ = _drop_one(x, int_x, pad_idx, device)
        elif mask_scheme == 'mask_one':
            int_x, feasible, pad_mask, _, _ = _mask_one(x, int_x, pad_idx, device)
        elif mask_scheme == 'random_drop':
            int_x, feasible = _random_drop(x, int_x, pad_idx, device)
    return int_x, feasible, pad_mask, token_mask, labels_cpl


def _drop_one(x, int_x, pad_idx, device):
    feasible = x.sum(1) > 1  
    int_x = int_x[feasible]  # recipes with at least 2 ingredients
    batch_size = int_x.size(0)
    _rand = torch.rand(int_x.size()).to(device)
    _rand[int_x == pad_idx] = -1  # to ignore pad_idx
    masked_ingred = torch.argmax(_rand, 1)
    labels_cpl = int_x[torch.arange(batch_size), masked_ingred].clone().detach()
    int_x[torch.arange(batch_size), masked_ingred] = pad_idx  # drop ingredients
    return int_x, feasible, labels_cpl

def _mask_one(x, int_x, pad_idx, device):
    feasible = x.sum(1) > 1  
    int_x = int_x[feasible]  # recipes with at least 2 ingredients
    batch_size = int_x.size(0)
    pad_mask = (int_x == pad_idx)
    _rand = torch.rand(int_x.size()).to(device)
    _rand[pad_mask] = -1  # to ignore pad_idx
    masked_ingred = torch.argmax(_rand, 1)
    token_mask = torch.full(int_x.size(), False).to(device)
    token_mask[torch.arange(batch_size), masked_ingred] = True
    labels_cpl = int_x[token_mask].clone().detach()
    # BERT style random masking. 80%: change to [MASK]. 10%: change to random idx. 10%: not mask.
    how_mask = torch.rand(int_x.size()).to(device)
    int_x[token_mask * (how_mask<0.8)] = pad_idx
    int_x[token_mask * (how_mask>0.9)] = torch.randint(pad_idx, int_x.size())[token_mask * (how_mask>0.9)].to(x.device)
    return int_x, feasible, pad_mask, token_mask, labels_cpl

def _random_drop(x, int_x, pad_idx, device):
    pad_mask = (int_x == pad_idx)
    _rand = torch.rand(int_x.size()).to(device)
    mask_ingred = _rand < 0.15  # position of ingredients to drop: True (choose 15% randomly)
    while torch.all(mask_ingred[torch.logical_not(pad_mask)]):  # not to drop all existing ingredients
        _rand = torch.rand(int_x.size()).to(device)
        mask_ingred = _rand < 0.15
    int_x[mask_ingred] = pad_idx  # drop
    feasible = (int_x != pad_idx).sum(1) > 0  # recipes with at least 2 left ingredients
    int_x = int_x[feasible]  # remove redundant recipes
    return int_x, feasible

