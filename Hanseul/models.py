import math
import torch
import torch.nn as nn

from .utils import make_one_hot

## Multi-head Attention Block
class MAB(nn.Module):
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
        self.fc_o = nn.Sequential(
            nn.Linear(dim_V, dim_V),
            nn.ReLU(),
            nn.Linear(dim_V, dim_V))
        self.dropout = nn.Dropout(p=dropout)

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
        
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)  # (batch, q_len, d_hid)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        _O = self.fc_o(O)
        if self.p > 0:
            _O = self.dropout(_O)
        O = O + _O 
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


## Self-Attention Block
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.2):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


## Induced Self-Attention Block
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.2):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask=mask)
        return self.mab1(X, H)


## Pooling by Multi-head Attention
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.2):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, dropout=dropout)
        
    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask=mask)

## Fully-connected Layer + Residual Connection (only if dim_input == dim_output)
class ResBlock(nn.Module):
    """
    (BatchNorm - LeakyReLU - Linear) * 2.
    Apply skip connection only when dim_input == dim_output.
    """
    def __init__(self, dim_input, dim_hidden, dim_output, norm='bn', dropout=0.2):
        super(ResBlock, self).__init__()
        self.use_skip_conn = (dim_input == dim_output)
        if norm == 'bn':
            norm_layer = nn.BatchNorm1d
        elif norm == 'ln':
            norm_layer = nn.LayerNorm
        ff = []
        if norm in ['bn', 'ln']:
            ff.append(norm_layer(dim_input))
        ff.extend([nn.LeakyReLU(), nn.Linear(dim_input, dim_hidden)])
        if norm in ['bn', 'ln']:
            ff.append(norm_layer(dim_hidden))
        ff.extend([nn.LeakyReLU(), nn.Linear(dim_hidden, dim_output)])
        if dropout > 0:
            ff.append(nn.Dropout(dropout))
        self.ff = nn.Sequential(*ff)
        
    def forward(self, x, **kwargs):
        if self.use_skip_conn:
            return self.ff(x) + x
        return self.ff(x)


## Encoder (DeepSets & SetTransformer Based.)
class Encoder(nn.Module):
    """ Create Feature Vector of Given Recipe. """
    def __init__(self, dim_embedding=256,
                 num_items=6714, 
                 num_inds=32,      # For ISAB
                 dim_hidden=128, 
                 num_heads=4, 
                 num_enc_layers=4,
                 ln=True,          # LayerNorm option
                 dropout=0.2,       # Dropout option
                 encoder_mode = 'set_transformer',
                 enc_pool_mode = 'set_transformer',
                ):
        super(Encoder, self).__init__()
        assert num_enc_layers % 2 == 0
        self.encoder_mode, self.enc_pool_mode = encoder_mode, enc_pool_mode
        self.padding_idx = num_items
        self.embedding = nn.Embedding(num_embeddings=num_items+1, embedding_dim=dim_embedding, padding_idx=-1)
        if encoder_mode == 'deep_sets':
            self.encoder = nn.ModuleList(
                [ResBlock(dim_embedding, dim_hidden, dim_hidden, norm='ln', dropout=dropout)] +
                [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers-1)])
        elif encoder_mode == 'set_transformer':
            self.encoder = nn.ModuleList(
                [ISAB(dim_embedding, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout)] +
                [ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers-1)])
        elif encoder_mode == 'fusion':
            self.encoder = nn.ModuleList(
                [ResBlock(dim_embedding, dim_hidden, dim_hidden, norm='ln', dropout=dropout)] +
                [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_enc_layers//2-1)] +
                [ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout=dropout) for _ in range(num_enc_layers//2)])
        if enc_pool_mode == 'deep_sets':
            def sumpool(x,**kwargs):
                return torch.sum(x, 1)
            self.pooling = sumpool
        elif enc_pool_mode == 'set_transformer':
            self.pooling = PMA(dim_hidden, num_heads, 1, ln=ln, dropout=dropout)
        self.out = self.mask = None
        
    def forward(self, x, mask):
        # x(=recipes): (batch, max_num_ingredient=65) : int_data.
        self.out = self.embedding(x)  # (batch, max_num_ingredient=65, dim_embedding=256)
        # cf. embedding.weight: (num_items+1=6715, dim_embedding=256)
        for module in self.encoder:
            self.out = module(self.out, mask=mask) # (batch, max_num_ingredient=65, dim_hidden=128) : permutation-equivariant.
        return self.pooling(self.out, mask=mask) # (batch, 1, dim_hidden=128) : permutation-invariant.


## Classification Model
class Classifier(nn.Module):
    def __init__(self, dim_hidden=128, dim_output=20, dropout=0.2, num_dec_layers=4):
        super(Classifier, self).__init__()
        self.classifier = nn.ModuleList(
                [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout) for _ in range(num_dec_layers-1)]
                +[ResBlock(dim_hidden, dim_hidden, dim_output, norm='bn', dropout=dropout)])
       
    def forward(self, x):
        # x: (batch, dim_hidden)
        assert x.ndim == 2
        self.out = x
        for module in self.classifier:
            self.out = module(self.out)
        return self.out  # (batch, dim_output)


## Completion Model
class Completer(nn.Module):
    def __init__(self, dim_embedding=256,
                 num_items=6714, 
                 #num_inds=32,      # For ISAB
                 dim_hidden=128, 
                 num_heads=4, 
                 num_dec_layers=4,
                 ln=True,          # LayerNorm option
                 dropout=0.2,      # Dropout option
                 mode = 'simple',
                ):
        super(Completer, self).__init__()

        assert mode in ['simple','concat','concat_attention','attention']
        
        self.num_items = num_items
        self.mode = mode
        # feedforward layer to process recipe representation
        self.ff = nn.Sequential(
                ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout),
                ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout))
        # 'simple': no need of embedding weight
        if mode == 'simple': 
            self.decoder = nn.ModuleList(
                [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='bn', dropout=dropout) for _ in range(num_dec_layers-3)]
                +[ResBlock(dim_hidden, dim_hidden, num_items, norm='bn', dropout=dropout)])
        # NCF style completer
        elif mode == 'concat':
            self.emb_encoder = nn.Sequential(
                ResBlock(dim_embedding, dim_hidden, dim_hidden, norm='ln', dropout=dropout),
                ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout))
            # decoding feedforward layer to deal with a concatenated feature (dim=2*dim_hidden)
            self.decoder = nn.ModuleList(
                [ResBlock(2*dim_hidden, dim_hidden, dim_hidden//2, norm='ln', dropout=dropout)]
                +[ResBlock(dim_hidden//2, dim_hidden//2, dim_hidden//2, norm='ln', dropout=dropout) for _ in range(num_dec_layers-4)]
                +[ResBlock(dim_hidden//2, dim_hidden//2, 1, norm='ln', dropout=dropout)])
        # completer based on concat + attention
        elif mode == 'concat_attention':
            self.emb_encoder = nn.Sequential(
                ResBlock(dim_embedding, dim_hidden, dim_hidden, norm='ln', dropout=dropout),
                ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout))
            self.new_set_encoder = nn.ModuleList(
                [SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout) for _ in range(num_dec_layers//2-1)])
            self.decoder= nn.ModuleList(
                [ResBlock(dim_hidden, dim_hidden, dim_hidden, norm='ln', dropout=dropout) for _ in range(num_dec_layers-num_dec_layers//2-2)]
                +[ResBlock(dim_hidden, dim_hidden, 1, norm='ln', dropout=dropout)])
        # completer based on attention
        elif mode == 'attention':
            pass
        
        self.out = self.emb_feature = None
        
    def forward(self, x, embedding_weight):
        # x: (batch, 1, dim_hidden) / embedding_weight: (num_items, dim_embedding)        
        self.out = self.ff(x.squeeze(1))  # (batch, dim_hidden=128)

        if self.mode == 'simple':
            for module in self.decoder:
                self.out = module(self.out)
            return self.out # (batch, num_items=6714)
        else:
            batch_size, num_items = x.size(0), embedding_weight.size(0)
            if self.mode == 'concat':
                self.emb_feature = self.emb_encoder(embedding_weight)  # (num_items, dim_hidden)
                self.out = torch.cat([self.out.unsqueeze(1).expand(-1,num_items,-1),
                    self.emb_feature.unsqueeze(0).expand(batch_size,-1,-1)], dim=2)  # (batch, num_items, 2*dim_hidden)
                for module in self.decoder:
                    self.out = module(self.out)  # (batch, num_items, 1)
                return self.out.squeeze(-1)  # (batch, num_items)
            elif self.mode == 'concat_attention':
                self.emb_feature = self.emb_encoder(embedding_weight)  # (num_items, dim_hidden)
                self.out = torch.cat([self.out.view(batch_size,1,1,-1).expand(-1,num_items,-1,-1),
                    self.emb_feature.view(1,num_items,1,-1).expand(batch_size,-1,-1,-1)], dim=2).view(batch_size*num_items,2,-1)  # (batch*num_items, 2, dim_hidden)
                for module in self.new_set_encoder:
                    self.out = module(self.out)  # (batch*num_items, 2, dim_hidden)
                self.out = self.out.sum(1).view(batch_size, num_items, -1)  # (batch, num_items, dim_hidden)
                for module in self.decoder:
                    self.out = module(self.out)  # (batch, num_items, 1)
                return self.out.squeeze(-1)  # (batch, num_items)
            elif self.mode == 'attention':
                pass


## Classification+Completion Network
class CCNet(nn.Module):
    def __init__(self, dim_embedding=256,
                 dim_output=20,
                 num_items=6714, 
                 num_inds=32, 
                 dim_hidden=128, 
                 num_heads=4, 
                 num_enc_layers=4, 
                 num_dec_layers=2,
                 ln=True,          # LayerNorm option
                 dropout=0.5,      # Dropout option
                 classify=True,    # completion만 하고 싶으면 False로
                 complete=True,    # classification만 하고 싶으면 False로
                 freeze_classify=False, # classification만 관련된 parameter freeze
                 freeze_complete=False,  # completion만 관련된 parameter freeze
                 encoder_mode = 'set_transformer',
                 enc_pool_mode = 'set_transformer',
                 decoder_mode = 'simple',
                 ):
        super(CCNet, self).__init__()
        self.padding_idx = num_items
        self.classify, self.complete = classify, complete

        self.encoder = Encoder(dim_embedding=dim_embedding,
                               num_items=num_items, 
                               num_inds=num_inds,
                               dim_hidden=dim_hidden, 
                               num_heads=num_heads, 
                               num_enc_layers=num_enc_layers,
                               ln=ln, dropout=dropout,
                               encoder_mode=encoder_mode,
                               enc_pool_mode=enc_pool_mode)
        if classify:
            self.classifier = Classifier(dim_hidden=dim_hidden,
                                         dim_output=dim_output,
                                         dropout=dropout)
            if freeze_classify:
                for p in self.classifier.parameters():
                    p.requires_grad = False
        if complete:
            self.completer = Completer(dim_embedding=dim_embedding,
                                       num_items=num_items, 
                                       #num_inds=num_inds,
                                       dim_hidden=dim_hidden, 
                                       num_heads=num_heads, 
                                       num_dec_layers=num_dec_layers,
                                       ln=ln, dropout=dropout,
                                       mode = decoder_mode)
            if freeze_complete:
                for p in self.completer.parameters():
                    p.requires_grad = False
    
    def forward(self, x, bin_x=None): 
        # x(=recipes): (batch, max_num_ingredients=65) : int_data.
        if not (self.classify or self.complete):
            return
        self.mask = (x == self.padding_idx).unsqueeze(1)  # (batch, 1, max_num_ingredients)
        recipe_feature = self.encoder(x, self.mask)  # (batch, 1, dim_hidden)
        
        logit_classification, logit_completion = None, None

        # Classification:
        if self.classify:
            logit_classification = self.classifier(recipe_feature.squeeze(1))  # (batch, dim_output)
            
        # Completion:
        if self.complete:
            embedding_weight = self.encoder.embedding.weight[:-1]  # (num_items=6714, dim_embedding=256)
            logit_completion = self.completer(recipe_feature, embedding_weight)  # (batch, num_items)

        return logit_classification, logit_completion
