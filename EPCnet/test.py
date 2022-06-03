import os
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import numpy as np
import torch

from utils import get_variables
from dataset import RecipeDataset
from models import CCNet
from torch.utils.data import Subset, DataLoader

def inference(model, phase, dataloaders, device, k=10):
    recs = {}
    
    if phase in ['train_clf','valid_clf', 'test_clf']:
        model.classify, model.complete = True, False
    elif phase in ['train_cpl','valid_cpl', 'test_cpl']:
        model.classify, model.complete = False, True
    else:
        raise ValueError('dataloader should not be train set')

    curr_idx = 0
    for loaded_data in tqdm(dataloaders[phase], desc=phase, total=len(dataloaders[phase])):
        if 'test' in phase:
            feature_boolean = loaded_data.to(device)
        else:
            feature_boolean = loaded_data[0].to(device)
        int_x, _, pad_mask, token_mask, _ = get_variables(feature_boolean, complete=model.complete, phase=phase, cpl_scheme=model.cpl_scheme)
        batch_size = int_x.size(0)

        outputs_clf, outputs_cpl = model(int_x, pad_mask, token_mask)
        if phase in ['train_clf','valid_clf', 'test_clf']:
            scores = torch.softmax(outputs_clf, 1).cpu()
        elif phase in ['train_cpl','valid_cpl', 'test_cpl']:
            scores = torch.softmax(outputs_cpl, 1).cpu()
        
        topk_scores, topk_indices = torch.topk(scores, k, 1, largest=True, sorted=True)
        
        for i in range(batch_size):
            recs[curr_idx + i] = [(int(ind), float(score)) for ind, score in zip(topk_indices[i], topk_scores[i])]
        
        curr_idx += batch_size
    return recs


def save_inference(recs, path):
    with open(path, 'wb') as fw:
        pickle.dump(recs, fw)


def run_inference(args):
    assert args.classify or args.complete
    
    # Datasets
    dataset_names = ['train', 'train_cpl', 'valid_clf', 'valid_cpl', 'test_clf', 'test_cpl']
    if args.datasets is None:
        recipe_datasets = {x: RecipeDataset(os.path.join(args.data_dir, x)) for x in dataset_names}
        recipe_datasets['train_clf'] = recipe_datasets['train']
    else:
        # Use Pre-loaded datasets. (compatible with jupyter notebook)
        recipe_datasets = args.datasets
        args.datasets = None
    
    dataset_names = ['train_clf'] + dataset_names[1:]
    
    # DataLoaders
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if args.subset_length is None else args.subset_length)
                          ] for x in dataset_names}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]),
                                 batch_size=args.batch_size_eval,
                                 shuffle=False) for x in dataset_names}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    if args.verbose:
        print(dataset_sizes)

    if torch.cuda.is_available():
        if hasattr(args, 'gpu') and args.gpu is not None: device = torch.device(f"cuda:{args.gpu}")
        else: device = torch.device('cuda')
    else: device = torch.device('cpu')
    if args.verbose:
        print('device: ', device)

    ## Get a batch of training data
    features_boolean, labels_one_hot, labels_int = next(iter(dataloaders['train_clf']))
    if args.verbose:
        print('features_boolean {} labels_one_hot {} labels_int {}'.format(
            features_boolean.size(), labels_one_hot.size(), labels_int.size()))

    model_ft = CCNet(dim_embedding=args.dim_embedding, dim_hidden=args.dim_hidden,
                         dim_outputs=labels_one_hot.size(1), num_items=features_boolean.size(-1),
                         num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers, num_inds=args.num_inds,
                         ln=True, dropout=0, classify=args.classify, complete=args.complete,
                         encoder_mode=args.encoder_mode, pooler_mode=args.pooler_mode, cpl_scheme=args.cpl_scheme).to(device)

    if args.verbose:
        #print(model_ft)  # Model Info
        total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad).values())
        print("Total Number of Parameters", total_params)

    if args.pretrained_model_path is not None:
        if args.verbose:
            print("Model Filename:", args.pretrained_model_path.split('/')[-1])
        pretrained_dict = torch.load(args.pretrained_model_path, map_location=device)
        model_dict = model_ft.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model_ft.load_state_dict(model_dict)
    else:
        print(" ### Warning! You are using a randomly initialized model. ### ")

    fname = ['rec','CCNet']
    fname += ['Enc'+args.encoder_mode, 'Pool'+args.pooler_mode, 'Cpl'+str(args.cpl_scheme).title()]
    fname += [f'NumEnc{args.num_enc_layers}', f'NumDec{args.num_dec_layers}']
    fname += [f'Hid{args.dim_hidden}', f'Emb{args.dim_embedding}', f'Ind{args.num_inds}']
    if not os.path.isdir('../Recs/'):
        os.mkdir('../Recs/')    
    
    model_ft.eval()
    with torch.set_grad_enabled(False):
        phases = []
        if args.classify:
            phases.extend(['train_clf','valid_clf', 'test_clf'])
        if args.complete:
            phases.extend(['train_cpl','valid_cpl', 'test_cpl'])

        for phase in phases:
            recs = inference(model_ft, phase, dataloaders, device, k=10)
            fname_ = fname[:]
            fname_.insert(2, phase)
            fname_ = '_'.join(fname_) + '.pickle'
            save_inference(recs, os.path.join('../Recs', fname_))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='../Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-b', '--batch_size_eval', default=1024, type=int,
                        help='batch size for evaluation.')
    parser.add_argument('-e', '--dim_embedding', default=512, type=int,
                        help='embedding dimensinon.')
    parser.add_argument('-dh', '--dim_hidden', default=512, type=int,
                        help='hidden dimensinon.')
    parser.add_argument('-i', '--num_inds', default=10, type=int,
                        help='num_inds for ISA / HYBRID encoder.')
    parser.add_argument('-g', '--gpu', default=None,
                        help='gpu number. (cuda:#)')
    parser.add_argument('-o', '--dropout', default=0.1, type=float,
                        help='probability for dropout layers.')
    parser.add_argument('-E', '--encoder_mode', default='HYBRID', type=str,
                        help='encoder mode: "FC", "SA", "ISA", "HYBRID", "HYBRID_SA".\n Note: HYBRID==FC+ISA, HYBRID_SA==FC+SA.')
    parser.add_argument('-P', '--pooler_mode', default='PMA', type=str,
                        help='encoder pooler mode: "SumPool", "PMA"')
    parser.add_argument('-C', '--cpl_scheme', default=None, type=str,
                        help='completion scheme: (a)="pooled", (b)="endcoded"')
    parser.add_argument('-ne', '--num_enc_layers', default=4, type=int,
                        help='depth of encoder (number of encoder blocks)')
    parser.add_argument('-nd', '--num_dec_layers', default=2, type=int,
                        help='depth of decoder (number of decoder blocks(ResBlocks))')
    parser.add_argument('-p', '--pretrained_model_path', default=None, type=str,
                        help=f"path for pretrained model.")
    parser.add_argument('-ss', '--subset_length', default=None)
    parser.add_argument('-f', '--classify', action='store_true')
    parser.add_argument('-t', '--complete', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-ds', '--datasets', default=None)

    run_inference(parser.parse_args())