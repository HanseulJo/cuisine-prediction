import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torch.optim import lr_scheduler

import wandb

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


def main(args):
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.wandb_log:
        proj_name = ''
        if args.classify:
            proj_name += 'Cuising Classification' + (' + ' if args.complete else '')
        if args.complete:
            proj_name += 'Recipe Completion'
        wandb.init(project=proj_name, config=args)
        args = wandb.config

    # Datasets
    train_data_name = 'train_class' if args.classify and not args.complete else 'train_compl'
    dataset_names = [train_data_name, 'valid_class', 'valid_compl', 'test_class', 'test_compl']
    recipe_datasets = {x: RecipeDataset(os.path.join(args.data_dir, x), test='test' in x) for x in dataset_names}
    # DataLoaders
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if args.subset_length is None else args.subset_length)
                          ] for x in dataset_names}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]),
                                 batch_size=args.batch_size, shuffle=('train' in x)) for x in dataset_names}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    print(dataset_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    ## Get a batch of training data
    loaded_data = next(iter(dataloaders[train_data_name]))
    print('bin_inputs, int_inputs, *labels:', [x.shape for x in loaded_data])

    model_ft = CCNet(dim_embedding=args.dim_embedding, dim_output=20, dim_hidden=args.dim_hidden, num_items=len(loaded_data[0][0]),
                     num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers, ln=True, dropout=args.dropout,
                     encoder_mode=args.encoder_mode, enc_pool_mode=args.enc_pool_mode, decoder_mode=args.decoder_mode,
                     classify=args.classify, complete=args.complete, freeze_classify=args.freeze_classify, freeze_complete=args.freeze_complete).to(device)
    if args.pretrained_model_path is not None:
        pretrained_dict = torch.load(args.pretrained_model_path)
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
    criterion = LOSSES[args.loss]().to(device)
    optimizer = OPTIMIZERS[args.optimizer_name]([p for p in model_ft.parameters() if p.requires_grad == True], lr=args.lr, **OPTIMIZERS_ARG[args.optimizer_name])
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.step_factor, patience=args.step_size, verbose=True)

    model_ft, best = train(model_ft, dataloaders, criterion, optimizer, exp_lr_scheduler,
                           dataset_sizes, device=device, num_epochs=args.n_epochs, early_stop_patience=args.early_stop_patience,
                           classify=args.classify, complete=args.complete, random_seed=args.seed, wandb_log=args.wandb_log)
    
    fname = ['ckpt', 'CCNet']
    if args.classify:
        fname.append('cls')
    if args.complete:
        fname.append('cmp')
    for k in best:
        if k == 'bestEpoch':
            fname.append(f'bestEpoch{int(best[k]):2d}')
        else:
            fname += [f"{k}{float(best[k]):.4f}"]
    fname += [f'bs{args.batch_size}',f'lr{args.lr}', f'seed{args.seed}',f'nEpochs{args.n_epochs}',]
    fname += ['encoder', args.encoder_mode, 'encPool', args.enc_pool_mode]
    if args.complete:
        fname += ['decoder', args.decoder_mode]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))
    if args.wandb_log:
        wandb.save(os.path.join('./weights/', 'ckpt*'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data', '--data_dir', default='.Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-bs', '--batch_size', default=64, type=int,
                        help='batch size for training.')
    parser.add_argument('-epochs', '--n_epochs', default=100, type=int,
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
    parser.add_argument('-logging', '--wandb_log', action='store_true')

    main(parser.parse_args())
