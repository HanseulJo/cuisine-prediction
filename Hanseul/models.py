import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_one_hot


class MAB(nn.Module):
    """ reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Sequential(
            nn.Linear(dim_V, 2*dim_V),
            nn.ReLU(),
            nn.Linear(2*dim_V, dim_V))

    def forward(self, Q, K, mask=None):
        # Q (batch, q_len, d_hid)
        # K (batch, k_len, d_hid)
        # V (batch, v_len, d_hid == dim_V)
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        
        # Q_ (batch * num_heads, q_len, d_hid // num_heads)
        # K_ (batch * num_heads, k_len, d_hid // num_heads)
        # V_ (batch * num_heads, v_len, d_hid // num_heads)
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        # energy (batch * num_heads, q_len, k_len)
        energy = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if mask is not None:
            energy.masked_fill_(mask, float('-inf'))
        A = torch.softmax(energy, 2)
        
        # O (batch, q_len, d_hid)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.fc_o(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """ reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


class ISAB(nn.Module):
    """ reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask=mask)
        return self.mab1(X, H)


class PMA(nn.Module):
    """ reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        
    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask=mask)


class CCNet(nn.Module):
    """ Classification + Completion based on Set Transformers """
    def __init__(self, dim_input=256,
                 dim_output=20,
                 num_items=6714+1, 
                 num_inds=32, 
                 dim_hidden=128, 
                 num_heads=4, 
                 num_outputs=1+1,  # class 1 + compl 1
                 num_enc_layers=4, 
                 num_dec_layers=2, 
                 ln=False, 
                 classify=True, 
                 complete=True):
        super(CCNet, self).__init__()
        
        self.num_heads = num_heads
        self.padding_idx = num_items-1
        self.classify, self.complete = classify, complete
        self.embedding =  nn.Embedding(num_embeddings=num_items, embedding_dim=dim_input, padding_idx=-1)
        self.encoder = nn.ModuleList(
            [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)] +
            [ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(num_enc_layers-1)])
        self.pooling = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.decoder1 = nn.Sequential(
            *[SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(num_dec_layers)])
        self.ff1 = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output))
        self.decoder2 = nn.ModuleList(
            [MAB(dim_hidden, dim_input, dim_hidden, num_heads, ln=ln) for _ in range(2)])
        self.ff2 = nn.Linear(dim_hidden, num_items-1)

    
    def forward(self, x):
        # x(=recipes): (batch, max_num_ingredient=65)
        # bin_x : (batch, num_items-1=6714)
        if not (self.classify or self.complete):
            return
        
        x = torch.LongTensor(x)
        feature = self.embedding(x)
        # feature: (batch, max_num_ingredient=65, dim_input=256)
        # cf. embedding.weight: (num_items=6715, dim_input=256)
        mask = (x == self.padding_idx).repeat(self.num_heads,1).unsqueeze(1)
        # mask: (batch*num_heads, 1, max_num_ingredient=65)
        code = feature.clone()
        for module in self.encoder:
            code = module(code, mask=mask)
        # code: (batch, max_num_ingredient=65, dim_hidden=128) : permutation-equivariant.
        
        pooled = self.pooling(code, mask=mask)
        # pooled: (batch, num_outputs=2, dim_hidden=128) : permutation-invariant.
        
        signals = self.decoder1(pooled)
        # no mask; signals: (batch, num_outputs=2, dim_hidden=128) : permutation-invariant.
        
        # split two signals: for classification & completion.
        signal_classification = signals[:][0]                   # (batch, dim_hidden=128)
        signal_completion = signals.clone()[:][1].unsqueeze(1)  # (batch, 1, dim_hidden=128)
        
        # Classification:
        if self.classify:
            logit_classification = self.ff1(signal_classification)  # (batch, dim_output)
            if not self.complete:
                return logit_clasification
        
        # Completion:
        if self.complete:
            bool_x = (make_one_hot(x) == True)
            used_ingred_mask = bool_x.repeat(self.num_heads,1).unsqueeze(1)
            # used_ingred_mask: (batch*num_heads, 1, num_items-1=6714)
            
            embedding_weight = self.embedding.weight[:-1].unsqueeze(0).repeat(feature.size(0),1,1)
            # embedding_weight: (batch, num_items=6715, dim_input=256)
            
            for module in self.decoder2:
                signal_completion = module(signal_completion, embedding_weight, mask=used_ingred_mask)
            logit_completion = self.ff2(signal_completion.squeeze()) # (batch, num_items-1=6714)
            logit_completion[bool_x] = float('-inf')
            if not self.classify:
                return logit_completion

        return logit_classification, logit_completion