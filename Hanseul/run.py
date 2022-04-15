import os
from argparse import ArgumentParser
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torch.optim import lr_scheduler

from .dataset import RecipeDataset
from .models import CCNet
from .train import train


def main(data_dir='./Container/'
         subset_length=None,
         dim_embedding=300,
         dropout=0.2,
         batch_size=16,
         n_epochs=50,
         lr=1e-3,
         step_size=10,  # training scheduler
         seed=0,
         classify=True,
         complete=True,
         freeze_classify=False,
         freeze_complete=False
         ):
    
    dataset_name = ['train', 'valid_class', 'valid_compl', 'test_class', 'test_compl']

    recipe_datasets = {x: RecipeDataset(os.path.join(data_dir, x), test='test' in x) for x in dataset_name}

    exclude_idx = []
    if complete:
        for i in range(len(recipe_datasets['train'])):
            _bd, _,_,_ = recipe_datasets['train'][i]
            if _bd.sum()<2:
                exclude_idx.append(i)

    dataset_name = ['train', 'valid_class', 'valid_compl']
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if subset_length is None else subset_length)
                            if i not in exclude_idx] for x in dataset_name}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]),
                                 batch_size=batch_size, shuffle=True) for x in dataset_name}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_name}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # Get a batch of training data
    bin_inputs, int_inputs, bin_labels, int_labels = next(iter(dataloaders['train']))
    print('inputs.shape', bin_inputs.shape, int_inputs.shape)
    print('labels.shape', bin_labels.shape, int_labels.shape)

    model_ft = CCNet(dim_embedding=dim_embedding, dim_output=len(bin_labels[0]),
                     num_items=len(bin_inputs[0]), num_outputs=2 if classify and complete else 1,
                     num_enc_layers=4, num_dec_layers=2, ln=True, dropout=0.2,
                     classify=classify, complete=complete,
                     freeze_classify=freeze_classify, freeze_complete=freeze_complete).to(device)
    print(model_ft)
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad ).values())
    print("Total Number of Parameters", total_params)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optimizer = optim.AdamW([p for p in model_ft.parameters() if p.requires_grad == True],
                                        lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=step_size,
                                                      eps=1e-08, verbose=True)

    model_ft = train(model_ft, dataloaders, criterion, optimizer, exp_lr_scheduler, #metrics,
                     dataset_sizes, device=device, num_epochs=n_epochs, early_stop_patience=20,
                     classify=classify, complete=complete, random_seed=seed)
    
    fname = ['ckpt', 'CCNet', 'batch', str(batch_size), 'n_epochs', str(n_epochs), 
             'lr', str(lr), 'step_size', str(step_size),'seed', str(seed),
             'dim_embedding', str(dim_embedding), 'subset_length', str(subset_length)]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='.Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size for training.')
    parser.add_argument('-e', '--n_epochs', default=50, type=int,
                        help='number of epochs for training.')
    parser.add_argument('-lr', '--lr', default=1e-3, type=float,
                        help='learning rate for training optimizer.')
    parser.add_argument('-step', '--step_size', default=10, type=int,
                        help='step size for training scheduler.')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed number.')
    parser.add_argument('-subset', '--subset_length', default=None, type=int,
                        help='using a subset of dataset. how many?')
    parser.add_argument('-embed', '--dim_embedding', default=300, type=int,
                        help='embedding dimensinon.')
    parser.add_argument('-drop', '--dropout', default=0.2, type=float,
                        help='probability for dropout layers.')
    parser.add_argument('-cls', '--classify', action='store_true')
    parser.add_argument('-cmp', '--complete', action='store_true')
    parser.add_argument('-fcls', '--freeze_classify', action='store_true')
    parser.add_argument('-fcmp', '--freeze_complete', action='store_true')
    args = parser.parse_args()

    main(data_dir=args.data_dir,
         subset_length=args.subset_length,
         dim_embedding=args.dim_embedding,
         dropout=args.dropout,
         batch_size=args.batch_size,
         n_epochs=args.n_epochs,
         lr=args.lr,
         step_size=args.step_size,  # training scheduler
         seed=args.seed,
         classify=args.classify,
         complete=args.complete,
         freeze_classify=args.freeze_classify,
         freeze_complete=args.freeze_complete)