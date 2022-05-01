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
                 mode='FC'):
        super(Encoder, self).__init__()
        assert mode in ['FC', 'ATT', 'HYBRID']
        self.mode = mode
        self.dropout = dropout
        self.padding_idx = num_items
        self.embedding = nn.Embedding(num_embeddings=num_items+1, embedding_dim=dim_embedding)
        if mode == 'FC':
            layers = [nn.Linear(dim_embedding, dim_hidden)]
            layers.extend([ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers)])
        elif mode == 'ATT':
            layers = [ISAB(dim_embedding, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout)]
            layers.extend([ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers)])
        elif mode == 'HYBRID':
            layers = [nn.Linear(dim_embedding, dim_hidden)]
            layers.extend([ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers-num_enc_layers//2)])
            layers.extend([ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers//2)])
        self.encoder = nn.ModuleList(layers)
        
    def forward(self, x, mask=None):
        """
        x: (batch, ?)
        out: (batch, ?, dim_hidden)
        """
        self.out = self.embedding(x)  # (batch, ?, dim_embedding)
        for module in self.encoder:
            if isinstance(module, ISAB):
                self.out = module(self.out, mask=mask)  # (batch, ?, dim_hidden)
            else:
                self.out = module(self.out) # (batch, ?, dim_hidden)
        return self.out


## Pooler
## Given ingredient encoding matrix, compute recipe feature vectors
class Pooler(nn.Module):
    def __init__(self, dim_hidden=128, num_heads=4, num_outputs=1, ln=True, dropout=0.2, mode='ATT'):
        super(Pooler, self).__init__()
        assert mode in ['deepSets', 'ATT']
        self.mode = mode
        if mode == 'deepSets':
            if num_outputs>1:
                self.ff = nn.Linear(dim_hidden, dim_hidden*num_outputs, bias=False)
        elif mode == 'ATT':
            self.pooling = PMA(dim_hidden, num_heads, num_outputs, ln=ln, dropout=dropout)
    
    def forward(self, x, mask=None):
        if self.mode == 'deepSets':
            self.out = x.sum(1)
            if hasattr(self, 'ff'):
                self.out = self.ff(self.out)
            return self.out  # (batch, dim_hidden*num_outputs)
        elif self.mode == 'ATT':
            self.out = self.pooling(x)
            return self.out.view(x.size(0), -1)  # (batch, dim_hidden*num_outputs)


## Decoder
class Decoder(nn.Module):
    def __init__(self, dim_hidden=128, dim_outputs=20, # 20 or 6714
                 num_dec_layers=4, dropout=0.2):
        super(Decoder, self).__init__()
        layers = [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout) for _ in range(num_dec_layers-1)]
        layers.append(nn.Linear(dim_hidden, dim_outputs))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


## Classification+Completion Network : Full Model
class CCNet(nn.Module):
    def __init__(self, dim_embedding=256, dim_hidden=128, dim_outputs=20, num_items=6714, 
                 num_inds=10, num_heads=4, num_enc_layers=4, num_dec_layers=2, num_outputs_cpl=2,
                 ln=True,          # LayerNorm option
                 dropout=0.5,      # Dropout option
                 classify=True,    # completion만 하고 싶으면 False로
                 complete=True,    # classification만 하고 싶으면 False로
                 freeze_classify=False, # classification만 관련된 parameter freeze
                 freeze_complete=False,  # completion만 관련된 parameter freeze
                 encoder_mode = 'FC',
                 pooler_mode = 'ATT',
                 #decoder_mode = 'FC',
                 ):
        super(CCNet, self).__init__()
        self.classify, self.complete = classify, complete

        self.encoder = Encoder(dim_embedding=dim_embedding, dim_hidden=dim_hidden, num_items=num_items,
                               num_inds=num_inds, num_heads=num_heads, num_enc_layers=num_enc_layers,
                               ln=ln, dropout=dropout, mode=encoder_mode)
        if classify:
            self.pooler1 = Pooler(dim_hidden=dim_hidden, num_heads=num_heads, num_outputs=1, mode=pooler_mode)
            self.classifier = Decoder(dim_hidden=dim_hidden, dim_outputs=dim_outputs, 
                                         num_dec_layers=num_dec_layers, dropout=dropout)
            if freeze_classify:
                for p in self.classifier.parameters():
                    p.requires_grad = False
        if complete:
            self.pooler2 = Pooler(dim_hidden=dim_hidden, num_heads=num_heads, num_outputs=num_outputs_cpl, mode=pooler_mode)
            self.completer = Decoder(dim_hidden=dim_hidden*num_outputs_cpl, dim_outputs=num_items,
                                       num_dec_layers=num_dec_layers, dropout=dropout)
            if freeze_complete:
                for p in self.completer.parameters():
                    p.requires_grad = False
    
    def forward(self, x):  # x: binary vectors.
        if not (self.classify or self.complete):
            return
        int_x = bin_to_int(x)  # (batch, ?)
        self.mask = (int_x == x.size(1)).unsqueeze(1)  # (batch, 1, ?)
        recipe_feature = self.encoder(int_x, self.mask)  # (batch, ?, dim_hidden)
        
        logit_classification, logit_completion = None, None

        # Classification:
        if self.classify:
            logit_classification = self.classifier(self.pooler1(recipe_feature))  # (batch, dim_output)
            
        # Completion:
        if self.complete:
            logit_completion = self.completer(self.pooler2(recipe_feature))  # (batch, num_items)

        return logit_classification, logit_completion