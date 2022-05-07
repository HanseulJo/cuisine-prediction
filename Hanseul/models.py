import math
import torch
import torch.nn as nn

from utils import bin_to_int
from modules import ResBlock, ISAB, PMA

## Encoder.
## Given recipes (in integers), compute encoding matrix of ingredients for each recipe.  
class Encoder(nn.Module):
    def __init__(self, dim_embedding=256, dim_hidden=128,
                 num_items=6714, num_inds=10, num_heads=4, 
                 num_enc_layers=4, ln=True, dropout=0.2,
                 mode='HYBRID'):
        super(Encoder, self).__init__()
        assert mode in ['FC', 'ISA', 'HYBRID']
        self.mode = mode
        self.dropout = dropout
        self.padding_idx = num_items
        self.embedding = nn.Embedding(num_embeddings=num_items+1, embedding_dim=dim_embedding)
        layers = [nn.Linear(dim_embedding, dim_hidden)]
        if mode == 'FC':
            layers.extend([ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers)])
        elif mode == 'ISA':
            layers.extend([ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers)])
        elif mode == 'HYBRID':  # FC + ISA
            layers.extend([ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers-num_enc_layers//2)])
            layers.extend([ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers//2)])
        self.encoder = nn.ModuleList(layers)
        
    def forward(self, x, mask=None):
        """
        x: (batch, ?) integers.
        out: (batch, ?, dim_hidden)
        """
        self.out = self.embedding(x)  # (batch, ?, dim_embedding)
        for module in self.encoder:
            if isinstance(module, (ISAB)):
                self.out = module(self.out, mask=mask)  # (batch, ?, dim_hidden)
            else:
                self.out = module(self.out) # (batch, ?, dim_hidden)
        return self.out


## Pooler
## Given ingredient encoding matrix, compute recipe feature vectors
class Pooler(nn.Module):
    def __init__(self, dim_hidden=128, num_heads=4, num_outputs=1, ln=True, dropout=0.2, mode='PMA'):
        super(Pooler, self).__init__()
        assert mode in ['SumPool', 'PMA']
        self.mode = mode
        if mode == 'SumPool':
            if num_outputs>1:
                self.ff = nn.Linear(dim_hidden, dim_hidden*num_outputs, bias=False)
        elif mode == 'PMA':
            self.pooling = PMA(dim_hidden, num_heads, num_outputs, ln=ln, dropout=dropout)
    
    def forward(self, x, mask=None):
        if self.mode == 'SumPool':
            self.out = x.sum(1)  # Sumpool
            if hasattr(self, 'ff'):
                self.out = self.ff(self.out)
            return self.out  # (batch, dim_hidden*num_outputs)
        elif self.mode == 'PMA':
            self.out = self.pooling(x, mask=mask)
            return self.out.view(x.size(0), -1)  # (batch, dim_hidden*num_outputs)


## Decoder: Classifier
class Decoder(nn.Module):
    def __init__(self, dim_hidden=128, dim_outputs=20, # 20 (cuisines) or 6714 (ingredients)
                 num_dec_layers=4, dropout=0.2):
        super(Decoder, self).__init__()
        layers = [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout) for _ in range(num_dec_layers)]
        layers.append(nn.Linear(dim_hidden, dim_outputs))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


## Cuisine Classification + Recipe Completion by Neural Network : Full Model
# recipe completion using recipe feature vector
class CCNet(nn.Module):
    def __init__(self, dim_embedding=256, dim_hidden=128, dim_outputs=20, num_items=6714, 
                 num_inds=10, num_heads=8, num_enc_layers=8, num_dec_layers=4, ln=True, dropout=0.5,
                 classify=True, complete=True, encoder_mode = 'HYBRID', pooler_mode = 'PMA', cpl_scheme = 'pooled'):
        super(CCNet, self).__init__()
        assert classify or complete
        self.classify, self.complete = classify, complete
        self.cpl_scheme = cpl_scheme
        self.pad_idx = num_items
        self.dim_hidden = dim_hidden

        self.encoder = Encoder(dim_embedding=dim_embedding, dim_hidden=dim_hidden, num_items=num_items,
                               num_inds=num_inds, num_heads=num_heads, num_enc_layers=num_enc_layers,
                               ln=ln, dropout=dropout, mode=encoder_mode)
        if classify:
            self.pooler1 = Pooler(dim_hidden=dim_hidden, num_heads=num_heads, num_outputs=1, mode=pooler_mode)
            self.classifier = Decoder(dim_hidden=dim_hidden, dim_outputs=dim_outputs, num_dec_layers=num_dec_layers, dropout=dropout)
        if complete:
            assert cpl_scheme in ['pooled', 'encoded']
            self.completer = Decoder(dim_hidden=dim_hidden,dim_outputs=num_items, num_dec_layers=num_dec_layers, dropout=dropout)
            if cpl_scheme == 'pooled':  # use recipe featue vector to predict missing ingredient
                self.pooler2 = Pooler(dim_hidden=dim_hidden, num_heads=num_heads, num_outputs=1, mode=pooler_mode)

    def forward(self, x, pad_mask=None, token_mask=None):  # x: recipes in integers. (batch, ?)
        
        if pad_mask is None:
            pad_mask = (x == self.pad_idx)  # (batch, ?)
        
        encoded_recipe = self.encoder(x, mask=pad_mask)  # (batch, ?, dim_hidden)
        
        logit_classification, logit_completion = None, None

        # Classification:
        if self.classify:
            _pad_mask = pad_mask.clone()
            if self.complete and self.cpl_scheme == 'encoded':
                _pad_mask[token_mask] = True  # additional masks
                assert _pad_mask.sum()-pad_mask.sum() == x.size(0)
            recipe_feature1 = self.pooler1(encoded_recipe, mask=_pad_mask) 
            logit_classification = self.classifier(recipe_feature1)  # (batch, dim_output)
            
        # Completion:
        if self.complete:
            if self.cpl_scheme == 'pooled' and hasattr(self, 'pooler2'):  # cpl_scheme == 'pooled'
                recipe_feature2 = self.pooler2(encoded_recipe, mask=pad_mask)
                logit_completion = self.completer(recipe_feature2)  # (batch, num_items)
            elif self.cpl_scheme == 'encoded':
                assert token_mask is not None, 'Is get_variables working?'
                assert token_mask.sum() == x.size(0), 'Is there any size mismatch?'
                encodings_to_predict = encoded_recipe.view(-1, self.dim_hidden)[token_mask.view(-1)]  # (batch, dim_hidden)
                logit_completion = self.completer(encodings_to_predict)  # (batch, num_items)

        return logit_classification, logit_completion