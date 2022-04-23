import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torch.optim import lr_scheduler

from .dataset import RecipeDataset
from .models import CCNet
from .loss import MultiClassASLoss, MultiClassFocalLoss
from .train import train

LOSSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'FocalLoss': MultiClassFocalLoss,
    'ASLoss': MultiClassASLoss,
}
OPTIMIZERS = {
    'SGD': optim.SGD,
    'MomentumSGD': optim.SGD,
    'NestrovSGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}
OPTIMIZERS_ARG = {
    'SGD': {'weight_decay':0.2},
    'MomentumSGD': {'weight_decay':0.2, 'momentum':0.9},
    'NestrovSGD': {'weight_decay':0.2, 'momentum':0.9, 'nesterov':True},
    'Adam': {'weight_decay':0.2},
    'AdamW': {'weight_decay':0.2},
}


def main(data_dir='./Container/',
        dim_embedding=256,
        dim_hidden=128,
        dropout=0.5,
        subset_length=None,
        encoder_mode='deep_sets',
        enc_pool_mode='set_transformer',
        decoder_mode='simple',
        num_enc_layers=4,
        num_dec_layers=4,
        batch_size=16,
        n_epochs=50,
        loss='ASLoss',
        optimizer_name='AdamW',
        lr=1e-3,
        step_size=10,  # lr_scheduler
        step_factor=0.1, # lr_scheduler
        early_stop_patience=20,   # early stop
        seed=0,
        classify=True,
        complete=True,
        freeze_classify=False,
        freeze_complete=False,
        pretrained_model_path=None
         ):
    
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    

    # Datasets
    train_data_name = 'train_class' if classify and not complete else 'train_compl'
    dataset_names = [train_data_name, 'valid_class', 'valid_compl', 'test_class', 'test_compl']
    recipe_datasets = {x: RecipeDataset(os.path.join(data_dir, x), test='test' in x) for x in dataset_names}
    # DataLoaders
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if subset_length is None else subset_length)
                          ] for x in dataset_names}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]),
                                 batch_size=batch_size, shuffle=('train' in x)) for x in dataset_names}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    print(dataset_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    ## Get a batch of training data
    loaded_data = next(iter(dataloaders[train_data_name]))
    print('bin_inputs, int_inputs, *labels:', [x.shape for x in loaded_data])

    model_ft = CCNet(dim_embedding=dim_embedding, dim_output=20, dim_hidden=dim_hidden, num_items=len(loaded_data[0][0]),
                     num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, ln=True, dropout=dropout,
                     encoder_mode=encoder_mode, enc_pool_mode=enc_pool_mode, decoder_mode=decoder_mode,
                     classify=classify, complete=complete, freeze_classify=freeze_classify, freeze_complete=freeze_complete).to(device)
    if pretrained_model_path is not None:
        pretrained_dict = torch.load(pretrained_model_path)
        model_dict = model_ft.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model_ft.load_state_dict(model_dict)
    #print(model_ft)  # Model Info
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad).values())
    print("Total Number of Parameters", total_params)

    # Loss, Optimizer, LR Scheduler
    criterion = LOSSES[loss]().to(device)
    optimizer = OPTIMIZERS[optimizer_name]([p for p in model_ft.parameters() if p.requires_grad == True], lr=lr, **OPTIMIZERS_ARG[optimizer_name])
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=step_factor, patience=step_size, verbose=True)

    model_ft, best = train(model_ft, dataloaders, criterion, optimizer, exp_lr_scheduler,
                           dataset_sizes, device=device, num_epochs=n_epochs, early_stop_patience=early_stop_patience,
                           classify=classify, complete=complete, random_seed=seed)
    
    fname = ['ckpt', 'CCNet']
    if classify:
        fname.append('cls')
    if complete:
        fname.append('cmp')
    for k in best:
        if k == 'bestEpoch':
            fname.append(f'bestEpoch{int(best[k]):2d}')
        else:
            fname += [f"{k}{float(best[k]):.4f}"]
    fname += [f'bs{batch_size}',f'lr{lr}', f'seed{seed}',f'nEpochs{n_epochs}',]
    fname += ['encoder', encoder_mode, 'encPool', enc_pool_mode]
    if complete:
        fname += ['decoder', decoder_mode]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='.Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size for training.')
    parser.add_argument('-e', '--n_epochs', default=100, type=int,
                        help='number of epochs for training.')
    parser.add_argument('-lr', '--lr', default=1e-3, type=float,
                        help='learning rate for training optimizer.')
    parser.add_argument('-step', '--step_size', default=10, type=int,
                        help='step size for learning rate scheduler.')
    parser.add_argument('-factor', '--step_factor', default=0.1, type=float,
                        help='multiplicative factor for learning rate scheduler.')
    parser.add_argument('-earlystop', '--early_stop_patience', default=30, type=int,
                        help='patience for early stopping.')
    parser.add_argument('-seed', '--seed', default=0, type=int,
                        help='random seed number.')
    parser.add_argument('-subset', '--subset_length', default=None, type=int,
                        help='using a subset of dataset. how many?')
    parser.add_argument('-emb', '--dim_embedding', default=256, type=int,
                        help='embedding dimensinon.')
    parser.add_argument('-hid', '--dim_hidden', default=256, type=int,
                        help='hidden dimensinon.')
    parser.add_argument('-drop', '--dropout', default=0.2, type=float,
                        help='probability for dropout layers.')
    parser.add_argument('-encmode', '--encoder_mode', default='deep_sets', type=str,
                        help='encoder mode: "deep_sets", "fusion", "set_transformer"')
    parser.add_argument('-poolmode', '--enc_pool_mode', default='set_transformer', type=str,
                        help='encoder pooler mode: "deep_sets", "fusion", "set_transformer"')
    parser.add_argument('-decmode', '--decoder_mode', default='simple', type=str,
                        help='decoder pooler mode: "simple", "concat", "concat_attention",...')
    parser.add_argument('-numenc', '--num_enc_layers', default=4, type=int,
                        help='depth of encoder (number of Resblock/ISAB')
    parser.add_argument('-numdec', '--num_dec_layers', default=4, type=int,
                        help='depth of decoder (number of Resblock/ISAB')
    parser.add_argument('-loss', '--loss', default='ASLoss', type=str,
                        help=f"loss functions: {list(LOSSES.keys())}")
    parser.add_argument('-opt', '--optimizer_name', default='AdamW', type=str,
                        help=f"optimizers: {list(OPTIMIZERS.keys())}")
    parser.add_argument('-pretrained', '--pretrained_model_path', default=None, type=str,
                        help=f"path for pretrained model.")
    parser.add_argument('-cls', '--classify', action='store_true')
    parser.add_argument('-cmp', '--complete', action='store_true')
    parser.add_argument('-fcls', '--freeze_classify', action='store_true')
    parser.add_argument('-fcmp', '--freeze_complete', action='store_true')

    main(**vars(parser.parse_args()))
