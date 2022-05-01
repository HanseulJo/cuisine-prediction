import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## Residual Bock 

class ResBlock(nn.Module):
    """
    (Norm - GELU - Linear) * 2.
    Apply skip connection only when dim_input == dim_output.
    """
    def __init__(self, dim_input, dim_hidden, dim_output, norm='bn', dropout=0):
        super(ResBlock, self).__init__()
        self.use_skip_conn = (dim_input == dim_output)

        if norm == 'bn':
            norm_layer = nn.BatchNorm1d
        elif norm == 'ln':
            norm_layer = nn.LayerNorm
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_output)
        if norm in ['bn', 'ln']:
            self.norm1 = norm_layer(dim_input)
            self.norm2 = norm_layer(dim_hidden)
        self.p = dropout
        
    def forward(self, x, **kwargs):
        out = x
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.fc1(F.gelu(out))
        if self.p > 0:
            out = F.dropout(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.fc2(F.gelu(out))
        if self.use_skip_conn:
            return out + x
        return out


## Attention Layers (from "Set Transformers")

class MAB(nn.Module):
    """ Multi-head Attention + FeedForward """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.p = dropout
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_att = nn.Linear(dim_V, dim_V)
        self.fc_o = nn.Sequential(nn.Linear(dim_V, dim_V), nn.GELU(), nn.Linear(dim_V, dim_V))

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)  # (batch, q_len, d_hid == dim_V)
        K, V = self.fc_k(K), self.fc_v(K) # (batch, k_len or v_len, d_hid)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # (batch * num_heads, q_len, d_hid // num_heads)
        K_ = torch.cat(K.split(dim_split, 2), 0)  # (batch * num_heads, c_len, d_hid // num_heads)
        V_ = torch.cat(V.split(dim_split, 2), 0)  # (batch * num_heads, v_len, d_hid // num_heads)
        
        energy = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)  # (batch * num_heads, q_len, k_len)
        if mask is not None:  # mask: (batch, 1, k_len)
            energy.masked_fill_(mask.repeat(self.num_heads, 1, 1), float('-inf'))
        A = torch.softmax(energy, 2)  # (batch * num_heads, q_len, k_len)
            
        O = self.fc_att(torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2))  # (batch, q_len, d_hid), Multihead Attention
        if self.p > 0: 
            O = F.dropout(O)  # Dropout
        O = O + Q  # Add 
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)  # normalize
        _O = self.fc_o(O)  # FF
        if self.p > 0:
            _O = F.dropout(_O)  # Dropout
        O = O + _O # Add
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)  # normalize
        return O


class SAB(nn.Module):
    """ Self Attention Block """
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.2):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


class ISAB(nn.Module):
    """ Induced Self Attention Block (Set Transformers) """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.2):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask=mask)
        return self.mab1(X, H)


class PMA(nn.Module):
    """ Pooling by Multi-head Attention (Set Transformers) """
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.2):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, dropout=dropout)
        
    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask=mask)

