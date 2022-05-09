import os, time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import accuracy_score
import wandb

from dataset import RecipeDataset
from models import Encoder
from utils import bin_to_int, get_variables
from run import LOSSES, OPTIMIZERS, OPTIMIZERS_ARG 


class PretrainNet(nn.Module):
    def __init__(self, dim_embedding=256, dim_hidden=128, num_items=6714, 
                 num_inds=10, num_heads=4, num_enc_layers=2,
                 ln=True,          # LayerNorm option
                 dropout=0.5,      # Dropout option
                 encoder_mode = 'ATT'
                 ):
        super(PretrainNet, self).__init__()
        self.encoder = Encoder(dim_embedding=dim_embedding, dim_hidden=dim_hidden, num_items=num_items,
                               num_inds=num_inds, num_heads=num_heads, num_enc_layers=num_enc_layers,
                               ln=ln, dropout=dropout, mode=encoder_mode)
        self.ff = nn.Linear(dim_hidden, num_items)
        self.pad_idx = num_items
        
    def forward(self, int_x, pad_mask, token_mask):  # x: binary vectors.
        out = self.encoder(int_x, mask=pad_mask.unsqueeze(1))  # (batch, ?, dim_hidden)
        out = self.ff(out).view(-1, self.pad_idx)[token_mask]
        return out
    

def statistics_pretrain(model, criterion, phase, dataloaders, device, verbose=True):
    running_loss = running_acc = 0.
    data_num = 0
    for idx, (feature_boolean, _, label_int) in enumerate(dataloaders[phase]):
        feature_boolean = feature_boolean.to(device)
        if 'train' in phase:
            int_x, pad_mask, token_mask, label = get_variables(feature_boolean, random_remove=True, random_mask=False)
        else:
            label = label_int.to(device)
            int_x, pad_mask, token_mask = get_variables(feature_boolean, random_remove=False)
        batch_size = int_x.size(0)
        
        outputs = model(int_x, pad_mask, token_mask)
        data_num += batch_size
        running_loss += criterion(outputs, label).item() * batch_size
        _, preds = torch.max(outputs, 1)
        #if verbose and idx == 0:
        #    print('label', label.cpu().numpy()[:12])
        #    print('preds', preds.cpu().numpy()[:12])
        running_acc += accuracy_score(label.cpu().numpy(), preds.cpu().numpy(), normalize=False)
    
    loss = float(running_loss / data_num)
    acc = float(running_acc / data_num)
    
    return loss, acc


def pretrain(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
              device='cpu', num_epochs=20, early_stop_patience=None,
              random_seed=1, wandb_log=False, verbose=True):
    
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.seed_all()
    
    since = time.time()
    
    # BEST MODEL SAVING
    best_loss = float('inf')
    best_acc = -1.
    best_model_wts = deepcopy(model.state_dict())

    if early_stop_patience is not None:
        patience_cnt = 0
    
    iterator, _temp = (range(1,num_epochs+1), None) if not verbose else (tqdm(range(1,num_epochs+1)), print('-' * 5 + 'Training the model' + '-' * 5))
    for epoch in iterator:
        None if not verbose else print(f'\nEpoch {epoch}/{num_epochs}')
        
        model.train()  # Set model to training mode
        
        for idx, (feature_boolean, _, _) in enumerate(dataloaders['train']):
            feature_boolean = feature_boolean.to(device)
            
            int_x, pad_mask, token_mask, label = get_variables(feature_boolean, random_remove=True, random_mask=True)
            
            optimizer.zero_grad()

            # forward
            outputs = model(int_x, pad_mask, token_mask)
            
            loss = criterion(outputs, label)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # gradient clipping
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            
            if verbose and idx % 100 == 0:
                if idx == 0:
                    print('label', label.cpu().numpy()[:10])
                    print('preds', preds.cpu().numpy()[:10])
                log_str = "    {:3d}% | Loss {:.4f} | Acc {:.4f}".format(
                    idx * 100 // len(dataloaders['train']), loss.item(), accuracy_score(label.cpu().numpy(), preds.cpu().numpy()))
                print(log_str)

        #if wandb_log:
        #    wandb.watch(model)
            
        # statistics
        model.eval()
        with torch.set_grad_enabled(False):
            train_loss, train_acc = statistics_pretrain(model, criterion, 'train_eval', dataloaders, device)
            if verbose:
                print(f"TRAIN Loss {train_loss:.4f} | Acc {train_acc:.4f}")
            valid_loss, valid_acc = statistics_pretrain(model, criterion, 'valid_cpl', dataloaders, device)
            if verbose:
                print(f"VALID Loss {valid_loss:.4f} | Acc {valid_acc:.4f}")

        is_new_best = valid_acc > best_acc
        if is_new_best:
            best_epoch = epoch
            best_loss = valid_loss
            best_acc = valid_acc
            best_model_wts = deepcopy(model.state_dict()) # deep copy the model
            if early_stop_patience is not None:
                patience_cnt = 0
        elif early_stop_patience is not None:
            patience_cnt += 1
        if verbose:
            print(f'best_valid_acc {best_acc:.4f} (epoch {best_epoch});')

        if wandb_log:
            log_dict = {'learning_rate': float(optimizer.param_groups[0]['lr']),  # scheduler.get_last_lr()[0] for CosineAnnealingWarmRestarts
                        'valid_acc': float(valid_acc), 'valid_loss': float(valid_loss)}
            wandb.log(log_dict)
        
        scheduler.step(-valid_acc)  # update learning rate after logging
        
        if early_stop_patience is not None and patience_cnt > early_stop_patience:
            if verbose:
                print(f'Early stop at epoch {epoch}.')
            break

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==== Best Result ====')
    print(f'bestEpoch: {best_epoch}')
    print(f'Loss: {best_loss}')
    print(f'Acc: {best_acc}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch, best_loss, best_acc


def run_pretrain(encoder_mode='HYBRID', batch_size=64, batch_size_eval=256, n_epochs=500, dropout=0.1,
               dim_embedding=256, dim_hidden=256, num_heads=8,
               subset_length=None, num_enc_layers=4, loss='CrossEntropyLoss', optimizer_name='AdamW',
               lr=1e-3, weight_decay=0.01, step_size=20, step_factor=0.25, patience=40, seed=42,
               data_dir='./Container/', pretrained_model_path=None, wandb_log=False):
    
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.seed_all()

    # Datasets
    dataset_names = ['train', 'valid_cpl']
    recipe_datasets = {x: RecipeDataset(os.path.join(data_dir, x)) for x in dataset_names}
    subset_indices = {x: [i for i in range(len(recipe_datasets[x]) if subset_length is None else subset_length)
                          ] for x in dataset_names}
    dataloaders = {x: DataLoader(Subset(recipe_datasets[x], subset_indices[x]), batch_size=batch_size if ('train' in x) else batch_size_eval,
                                 shuffle=('train' in x)) for x in dataset_names}
    dataset_sizes = {x: len(subset_indices[x]) for x in dataset_names}
    print(dataset_sizes)
    dataloaders['train_eval'] = DataLoader(Subset(recipe_datasets['train'], subset_indices['train']),
                                 batch_size=batch_size_eval, shuffle=False)
    dataset_sizes['train_eval'] = dataset_sizes['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # WandB
    if wandb_log:
        wandb.init(project='Pretraining Encoder by Completion',
                   config=dict(encoder_mode=encoder_mode, batch_size=batch_size, n_epochs=n_epochs, dropout=dropout,
                               dim_embedding=dim_embedding, dim_hidden=dim_hidden, num_heads=num_heads,
                               num_enc_layers=num_enc_layers, loss=loss, optimizer_name=optimizer_name,
                               lr=lr, weight_decay=weight_decay, step_size=step_size, step_factor=step_factor, patience=patience, seed=seed,))

    ## Get a batch of training data
    features_boolean, labels_one_hot, labels_int = next(iter(dataloaders['train']))
    print('features_boolean {} labels_one_hot {} labels_int {}'.format(
            features_boolean.size(), labels_one_hot.size(), labels_int.size()))
    
    model_ft = PretrainNet(dim_embedding=dim_embedding, dim_hidden=dim_hidden, num_items=features_boolean.size(-1),
                           num_heads=num_heads, num_enc_layers=num_enc_layers, ln=True, dropout=dropout,
                           encoder_mode=encoder_mode).to(device)
    if pretrained_model_path is not None:
        pretrained_dict = torch.load(pretrained_model_path)
        model_dict = model_ft.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model_ft.load_state_dict(model_dict)
        
    # Model Info
    #print(model_ft)
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model_ft.parameters() if p.requires_grad).values())
    print("Total Number of Parameters", total_params)

    # Loss, Optimizer, LR Scheduler
    criterion = LOSSES[loss]().to(device)
    optimizer = OPTIMIZERS[optimizer_name]([p for p in model_ft.parameters() if p.requires_grad == True],lr=lr, weight_decay=weight_decay, **OPTIMIZERS_ARG[optimizer_name])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=step_factor, patience=step_size, verbose=True)
    
    try:
        model_ft, best_epoch, best_loss, best_acc = pretrain(model_ft, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
                                                            device=device, num_epochs=n_epochs, early_stop_patience=patience,
                                                            random_seed=1, wandb_log=wandb_log)
    except KeyboardInterrupt:
        if wandb_log:
            wandb.finish()
        print("Finished by KeyboardInterrupt")
        raise Exception
        

    fname = ['ckpt', 'Pretrain', f'bestEpoch{best_epoch}', f'loss{best_loss:.3f}', f'acc{best_acc:.3f}']
    fname += [f'bs{batch_size}',f'lr{lr}', f'seed{seed}',f'nEpochs{n_epochs}',]
    fname += ['Encoder', encoder_mode]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))
    wandb.finish()


if __name__ == '__main__':
    run_pretrain(encoder_mode='ATT',
                 batch_size=32,
                 n_epochs=1000,
                 num_heads=8,
                 num_enc_layers=4,
                 lr=4e-4,
                 weight_decay=0.1,
                 dropout=0.1,
                 mask_prob=0.2,
                 wandb_log=True)