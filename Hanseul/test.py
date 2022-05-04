import os
import pickle
from argparse import ArgumentParser
import numpy as np
import torch

from utils import get_variables
from dataset import RecipeDataset
from models import CCNet
from torch.utils.data import Subset, DataLoader

def inference(model, phase, dataloaders, device, k=10, idx_start_with=0):
    recs = {}
    
    if phase in ['train_eval','valid_clf', 'test_clf']:
        model.classify, model.complete = True, False
    elif phase in ['valid_cpl', 'test_cpl']:
        model.classify, model.complete = False, True
    else:
        raise ValueError('dataloader should not be train set')

    curr_idx = 0
    for loaded_data in dataloaders[phase]:
        if 'test' in phase:
            feature_boolean = loaded_data.to(device)
        else:
            feature_boolean = loaded_data[0].to(device)
        int_x, _, pad_mask, token_mask, _ = get_variables(feature_boolean, complete=model.complete, phase=phase, cpl_scheme=model.cpl_scheme)
        batch_size = int_x.size(0)

        outputs_clf, outputs_cpl = model(int_x, pad_mask, token_mask)
        if phase in ['train_eval','valid_clf', 'test_clf']:
            scores = torch.softmax(outputs_clf, 1).cpu()
        elif phase in ['valid_cpl', 'test_cpl']:
            scores = torch.softmax(outputs_cpl, 1).cpu()
        
        topk_scores, topk_indices = torch.topk(scores, k, 1, largest=True, sorted=True)
        topk_indices += idx_start_with  # sometimes it should not start with 0
        
        for i in range(batch_size):
            recs[curr_idx + i] = [(int(ind), float(score)) for ind, score in zip(topk_indices[i], topk_scores[i])]
        
        curr_idx += batch_size
    return recs


def save_inference(recs, path):
    with open(path, 'wb') as fw:
        pickle.dump(recs, fw)


def run_inference(args):
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
                                 batch_size=args.batch_size_eval,
                                 shuffle=False) for x in dataset_names[1:]}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    dataloaders['train_eval'] = DataLoader(Subset(recipe_datasets['train'], subset_indices['train']),
                                 batch_size=args.batch_size_eval, shuffle=False)
    dataset_sizes['train_eval'] = dataset_sizes['train']
    if args.verbose:
        print(dataset_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print('device: ', device)

    ## Get a batch of training data
    features_boolean, labels_one_hot, labels_int = next(iter(dataloaders['train_eval']))
    if args.verbose:
        print('features_boolean {} labels_one_hot {} labels_int {}'.format(
            features_boolean.size(), labels_one_hot.size(), labels_int.size()))

    model_ft = CCNet(dim_embedding=args.dim_embedding, dim_hidden=args.dim_hidden,
                     dim_outputs=labels_one_hot.size(1), num_items=features_boolean.size(-1),
                     num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers,
                     ln=True, dropout=args.dropout, classify=True, complete=True,
                     encoder_mode=args.encoder_mode, pooler_mode=args.pooler_mode, cpl_scheme=args.cpl_scheme).to(device)
    if args.verbose:
        #print(model_ft)  # Model Info
        total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad).values())
        print("Total Number of Parameters", total_params)

    if args.pretrained_model_path is not None:
        pretrained_dict = torch.load(args.pretrained_model_path)
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
    fname += ['Enc', args.encoder_mode, 'Pool', args.pooler_mode, 'Cpl', args.cpl_scheme]
    fname += ['NumEnc', args.num_enc_layers, 'NumDec', args.num_dec_layers]
    if not os.path.isdir('./recs/'):
        os.mkdir('./recs/')
    
    model_ft.eval()
    with torch.set_grad_enabled(False):
        for phase in ['train_eval','valid_clf', 'test_clf']:
            recs = inference(model_ft, phase, dataloaders, device, k=10, idx_start_with=0)
            fname.insert(2, phase)
            fname = '_'.join(fname) + '.pickle'
            save_inference(recs, os.path.join('./recs', fname))
        for phase in ['valid_cpl', 'test_cpl']:
            recs = inference(model_ft, phase, dataloaders, device, k=10, idx_start_with=1)  # idx start with 1
            fname.insert(2, phase)
            fname = '_'.join(fname) + '.pickle'
            save_inference(recs, os.path.join('./recs', fname))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-dir', '--data_dir', default='./Container/', type=str,
                        help='path to the dataset.')
    parser.add_argument('-bsev', '--batch_size_eval', default=2048, type=int,
                        help='batch size for evaluation.')
    parser.add_argument('-emb', '--dim_embedding', default=256, type=int,
                        help='embedding dimensinon.')
    parser.add_argument('-hid', '--dim_hidden', default=256, type=int,
                        help='hidden dimensinon.')
    parser.add_argument('-dr', '--dropout', default=0.1, type=float,
                        help='probability for dropout layers.')
    parser.add_argument('-em', '--encoder_mode', default='HYBRID', type=str,
                        help='encoder mode: "FC", "ISA", "HYBRID"')
    parser.add_argument('-pm', '--pooler_mode', default='PMA', type=str,
                        help='encoder pooler mode: "SumPool", "PMA"')
    parser.add_argument('-cs', '--cpl_scheme', default='pooled', type=str,
                        help='completion scheme: (a)="pooled", (b)="endcoded"')
    parser.add_argument('-ne', '--num_enc_layers', default=4, type=int,
                        help='depth of encoder (number of Resblock/ISAB')
    parser.add_argument('-nd', '--num_dec_layers', default=2, type=int,
                        help='depth of decoder (number of Resblock/ISAB')
    parser.add_argument('-pt', '--pretrained_model_path', default=None, type=str,
                        help=f"path for pretrained model.")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-ds', '--datasets', default=None)

    run_inference(parser.parse_args())