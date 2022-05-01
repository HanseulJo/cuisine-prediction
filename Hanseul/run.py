import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torch.optim import lr_scheduler

import wandb

from dataset import RecipeDataset
from models import CCNet
from loss import MultiClassASLoss, MultiClassFocalLoss
from train import train

LOSSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'MultiClassFocalLoss': MultiClassFocalLoss,
    'MultiClassASLoss': MultiClassASLoss,
}
OPTIMIZERS = {
    'SGD': optim.SGD,
    'MomentumSGD': optim.SGD,
    'NestrovSGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}
OPTIMIZERS_ARG = {
    'SGD': {},
    'MomentumSGD': {'momentum':0.9},
    'NestrovSGD': {'momentum':0.9, 'nesterov':True},
    'Adam': {},
    'AdamW': {},
}


def main(args):
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.seed_all()

    # Datasets
    dataset_names = ['train', 'valid_clf', 'valid_cpl', 'test_clf', 'test_cpl']
    if args.datasets is None:
        recipe_datasets = {x: RecipeDataset(os.path.join(args.data_dir, x)) for x in dataset_names}
    else:
        # Use Pre-loaded datasets.
        recipe_datasets = args.datasets
        args.datasets = None
    # DataLoaders
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if args.subset_length is None else args.subset_length)
                          ] for x in dataset_names}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]),
                                 batch_size=args.batch_size if ('train' in x) else args.batch_size_eval,
                                 shuffle=('train' in x)) for x in dataset_names}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    if args.verbose:
        print(dataset_sizes)
    dataloaders['train_eval'] = DataLoader(Subset(recipe_datasets['train'], subset_indices['train']),
                                 batch_size=args.batch_size_eval, shuffle=False)
    dataset_sizes['train_eval'] = dataset_sizes['train']
    
    # WandB
    if args.wandb_log:
        proj_name = ''
        if args.classify:
            proj_name += 'Cuisine Classification' + (' + ' if args.complete else '')
        if args.complete:
            proj_name += 'Recipe Completion'
        wandb.init(project=proj_name, config=args)
        args = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print('device: ', device)

    ## Get a batch of training data
    features_boolean, labels_one_hot, labels_int = next(iter(dataloaders['train']))
    if args.verbose:
        print('features_boolean {} labels_one_hot {} labels_int {}'.format(
            features_boolean.size(), labels_one_hot.size(), labels_int.size()))

    model_ft = CCNet(dim_embedding=args.dim_embedding, dim_hidden=args.dim_hidden,
                     dim_outputs=labels_one_hot.size(0), num_items=features_boolean.size(-1),
                     num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers, num_outputs_cpl=args.num_outputs_cpl,
                     ln=True, dropout=args.dropout, encoder_mode=args.encoder_mode, pooler_mode=args.pooler_mode, 
                     classify=args.classify, complete=args.complete).to(device)
    if args.pretrained_model_path is not None:
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model_ft.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model_ft.load_state_dict(model_dict)
        if args.freeze_encoder:
            for p in model_ft.encoder.parameters():
                p.requires_grad = False
    if args.verbose:
        #print(model_ft)  # Model Info
        total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad).values())
        print("Total Number of Parameters", total_params)

    # Loss, Optimizer, LR Scheduler
    criterion = LOSSES[args.loss]().to(device)
    optimizer = OPTIMIZERS[args.optimizer_name]([p for p in model_ft.parameters() if p.requires_grad == True],
                                                lr=args.lr, weight_decay=args.weight_decay, **OPTIMIZERS_ARG[args.optimizer_name])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.step_factor, patience=args.step_size, verbose=args.verbose)

    try:
        model_ft, best = train(model_ft, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
                               device=device, num_epochs=args.n_epochs, early_stop_patience=args.early_stop_patience,
                               random_seed=args.seed, wandb_log=args.wandb_log, verbose=args.verbose)
    except KeyboardInterrupt:
        if args.wandb_log:
            wandb.finish()
        print("Finished by KeyboardInterupt")
        return
    fname = ['ckpt', 'CCNet']
    if args.classify:
        fname.append('cls')
    if args.complete:
        fname.append('cmp')
    for k in best:
        if k == 'bestEpoch':
            fname.append(f'bestEpoch{int(best[k]):2d}')
        else:
            fname += [f"{k}{float(best[k]):.3f}"]
    fname += [f'bs{args.batch_size}',f'lr{args.lr}', f'seed{args.seed}',f'nEpochs{args.n_epochs}',]
    fname += ['Encodee', args.encoder_mode, 'Pool', args.pooler_mode]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))
    if args.wandb_log:
        wandb.save(os.path.join('./weights/', 'ckpt*'))
    wandb.finish()
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data', '--data_dir', default='./Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-bs', '--batch_size', default=64, type=int,
                        help='batch size for training.')
    parser.add_argument('-bseval', '--batch_size_eval', default=2048, type=int,
                        help='batch size for evaluation.')
    parser.add_argument('-epochs', '--n_epochs', default=100, type=int,
                        help='number of epochs for training.')
    parser.add_argument('-lr', '--lr', default=2e-4, type=float,
                        help='learning rate for training optimizer.')
    parser.add_argument('-l2', '--weight_decay', default=0.01, type=float,
                        help='l2 regularization for optimizer.')
    parser.add_argument('-step', '--step_size', default=10, type=int,
                        help='step size for learning rate scheduler.')
    parser.add_argument('-factor', '--step_factor', default=0.1, type=float,
                        help='multiplicative factor for learning rate scheduler.')
    parser.add_argument('-earlystop', '--early_stop_patience', default=20, type=int,
                        help='patience for early stopping.')
    parser.add_argument('-seed', '--seed', default=42, type=int,
                        help='random seed number.')
    parser.add_argument('-subset', '--subset_length', default=None, type=int,
                        help='using a subset of dataset. how many?')
    parser.add_argument('-emb', '--dim_embedding', default=256, type=int,
                        help='embedding dimensinon.')
    parser.add_argument('-hid', '--dim_hidden', default=256, type=int,
                        help='hidden dimensinon.')
    parser.add_argument('-drop', '--dropout', default=0.1, type=float,
                        help='probability for dropout layers.')
    parser.add_argument('-encmode', '--encoder_mode', default='HYBRID', type=str,
                        help='encoder mode: "FC", "ATT", "HYBRID"')
    parser.add_argument('-poolmode', '--pooler_mode', default='ATT', type=str,
                        help='encoder pooler mode: "deepSets", "ATT"')
    parser.add_argument('-noutcpl', '--num_outputs_cpl', default=1, type=int,
                        help='') 
    parser.add_argument('-numenc', '--num_enc_layers', default=4, type=int,
                        help='depth of encoder (number of Resblock/ISAB')
    parser.add_argument('-numdec', '--num_dec_layers', default=2, type=int,
                        help='depth of decoder (number of Resblock/ISAB')
    parser.add_argument('-loss', '--loss', default='ASLoss', type=str,
                        help=f"loss functions: {list(LOSSES.keys())}")
    parser.add_argument('-opt', '--optimizer_name', default='AdamW', type=str,
                        help=f"optimizers: {list(OPTIMIZERS.keys())}")
    parser.add_argument('-pretrained', '--pretrained_model_path', default=None, type=str,
                        help=f"path for pretrained model.")
    parser.add_argument('-cls', '--classify', action='store_true')
    parser.add_argument('-cmp', '--complete', action='store_true')
    parser.add_argument('-fenc', '--freeze_encoder', action='store_true')
    parser.add_argument('-logging', '--wandb_log', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-datasets', '--datasets', default=None)



    main(parser.parse_args())
