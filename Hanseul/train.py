import time
from copy import deepcopy
#from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import wandb

from utils import _concatenate, get_variables

def statistics(model, phase, dataloaders, dataset_sizes, device, k=5):
    celoss = nn.CrossEntropyLoss().to(device)  # universal criterion
    
    running_loss = running_acc = running_top_k_acc = 0.
    running_labels = running_preds = None
    
    temp_classify, temp_complete = model.classify, model.complete
    if phase in ['train_eval', 'valid_clf']:
        model.complete = False
    elif phase in ['valid_cpl']:
        model.classify = False

    for idx, (feature_boolean, _, labels_int) in enumerate(dataloaders[phase]):
        feature_boolean = feature_boolean.to(device)
        int_x, feasible, pad_mask, token_mask, _ = get_variables(feature_boolean, complete=(phase=='valid_cpl'), phase=phase, cpl_scheme=model.cpl_scheme)
        labels_int = labels_int[feasible].to(device)
        batch_size = int_x.size(0)

        outputs_clf, outputs_cpl = model(int_x, pad_mask, token_mask)
        if phase in ['train_eval', 'valid_clf']:
            _, preds = torch.max(outputs_clf, 1)
        elif phase in ['valid_cpl']:
            _, preds = torch.max(outputs_cpl, 1)
        
        #if verbose and idx == 0:
        #    print('label', labels_int.cpu().numpy()[:10])
        #    print('preds', preds.cpu().numpy()[:10])
        
        if phase in ['train_eval', 'valid_clf']:
            running_loss += celoss(outputs_clf, labels_int.long()) * batch_size
            running_top_k_acc += top_k_accuracy_score(labels_int.cpu().numpy(), outputs_clf.cpu().numpy(), k=k, labels=np.arange(outputs_clf.size(1)), normalize=False)
        elif phase in ['valid_cpl']:
            running_loss += celoss(outputs_cpl, labels_int.long()) * batch_size
            running_top_k_acc += top_k_accuracy_score(labels_int.cpu().numpy(), outputs_cpl.cpu().numpy(), k=k, labels=np.arange(outputs_cpl.size(1)), normalize=False)
        running_acc += accuracy_score(labels_int.cpu().numpy(), preds.cpu().numpy(), normalize=False)
        running_labels = _concatenate(running_labels, labels_int)
        running_preds = _concatenate(running_preds, preds)
    
    model.classify, model.complete = temp_classify, temp_complete

    stat = dict(
        Loss = running_loss / dataset_sizes[phase],
        Acc = running_acc / dataset_sizes[phase],
        Topk = running_top_k_acc / dataset_sizes[phase],
        F1macro = f1_score(running_labels, running_preds, average='macro'),
        F1micro = f1_score(running_labels, running_preds, average='micro')
    )
    return stat


# train classification only / completion only / simultaneously.
def train(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
         device='cpu', num_epochs=20, early_stop_patience=None, mask_scheme=None,
         random_seed=1, wandb_log=False, verbose=True):
    
    classify = model.classify
    complete = model.complete
    
    since = time.time()
    
    # BEST MODEL SAVING
    best = {}
    if classify:
        best['clf'] = dict(Loss=float('inf'), Acc=-1., Topk=-1., F1macro=-1., F1micro=-1., Model=deepcopy(model.state_dict()))
    if complete:
        best['cpl'] = dict(Loss=float('inf'), Acc=-1., Topk=-1., F1macro=-1., F1micro=-1., Model=deepcopy(model.state_dict()))

    if early_stop_patience is not None:
        patience_cnt = 0
    
    iterator, _temp = (range(1,num_epochs+1), None) if not verbose else (tqdm(range(1,num_epochs+1)), print('-' * 5 + 'Training the model' + '-' * 5))
    for epoch in iterator:
        None if not verbose else print(f'\nEpoch {epoch}/{num_epochs}')
        
        model.train()  # Set model to training mode
        
        for idx, (feature_boolean, _, labels_clf) in enumerate(dataloaders['train']):
            feature_boolean = feature_boolean.to(device)
            int_x, feasible, pad_mask, token_mask, labels_cpl = get_variables(feature_boolean, complete=complete, phase='train', cpl_scheme=model.cpl_scheme, mask_scheme=mask_scheme)
            labels_clf = labels_clf[feasible].to(device)

            optimizer.zero_grad()
            outputs_clf, outputs_cpl = model(int_x, pad_mask, token_mask)
            loss = 0.
            if classify:
                _, preds_clf = torch.max(outputs_clf, 1)
                loss_clf = criterion(outputs_clf, labels_clf.long())
                loss = loss + loss_clf
            if complete: 
                _, preds_cpl = torch.max(outputs_cpl, 1)
                loss_cpl = criterion(outputs_cpl, labels_cpl.long())
                loss = loss + loss_cpl
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # gradient clipping
            optimizer.step()
            
            if verbose and idx % 100 == 0:
                log_str = f"    {idx * 100 // len(dataloaders['train']):3d}% | "
                if classify:
                    if idx == 0:
                        print('    *label_clf', labels_clf.cpu().numpy()[:12])
                        print('    *preds_clf', preds_clf.cpu().numpy()[:12])
                    log_str += "Loss_clf {:.4f} | Acc_clf {:.4f} | ".format(
                        loss_clf.item(), accuracy_score(labels_clf.cpu().numpy(), preds_clf.cpu().numpy()))
                if complete:
                    if idx == 0:
                        print('    >label_cpl', labels_cpl.cpu().numpy()[:10])
                        print('    >preds_cpl', preds_cpl.cpu().numpy()[:10])
                    log_str += "Loss_cpl {:.4f} | Acc_cpl {:.4f}".format(
                        loss_cpl.item(), accuracy_score(labels_cpl.cpu().numpy(), preds_cpl.cpu().numpy()))
                print(log_str)

        #if wandb_log:
        #    wandb.watch(model)  # it sometimes take too much memory
            
        # statistics
        model.eval()
        with torch.set_grad_enabled(False):
            if classify:
                stat_train = statistics(model, 'train_eval', dataloaders, dataset_sizes, device, k=5)
                if verbose:
                    print("TRAIN_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_train.items()]))
                stat_valid_clf = statistics(model, 'valid_clf', dataloaders, dataset_sizes, device, k=5)
                if verbose:
                    print("VALID_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_clf.items()]))
                scheduler_criterion = float(stat_valid_clf['Loss'])
            if complete:
                stat_valid_cpl = statistics(model, 'valid_cpl', dataloaders, dataset_sizes, device, k=10)
                if verbose:
                    print("VALID_CPL", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_cpl.items()]))
                if classify:
                    scheduler_criterion += float(stat_valid_cpl['Loss'])
                else:
                    scheduler_criterion = float(stat_valid_cpl['Loss'])
        
        patience_add = True
        if classify:
            if stat_valid_clf['Loss'] < best['clf']['Loss']:
                best['clf'].update(stat_valid_clf)
                best['clf']['BestEpoch'] = int(epoch)
                best['clf']['Model'] = deepcopy(model.state_dict()) # deep copy the model
                if early_stop_patience is not None:
                    patience_cnt, patience_add = 0, False
        if complete:
            if stat_valid_cpl['Loss'] < best['cpl']['Loss']:
                best['cpl'].update(stat_valid_cpl)
                best['cpl']['BestEpoch'] = int(epoch)
                best['cpl']['Model'] = deepcopy(model.state_dict()) # deep copy the model
                if early_stop_patience is not None:
                    patience_cnt, patience_add = 0, False
        if early_stop_patience is not None:
            if patience_add:
                patience_cnt += 1
            print("patience_cnt", patience_cnt)

        if wandb_log:
            if scheduler is None or isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                log_dict = {'learning_rate': optimizer.param_groups[0]['lr']}   # for ReduceOnPlateau
            else:
                log_dict = {'learning_rate': scheduler.get_last_lr()[0]}        # for other schedulers
            
            if classify:
                log_dict.update({'Train/Classify/'+k: v for k, v in stat_train.items()})
                log_dict.update({'Valid/Classify/'+k: v for k, v in stat_valid_clf.items()})
                log_dict.update({'Best/Classify/'+k: v for k, v in best['clf'].items() if k != 'Model'})
            if complete:
                log_dict.update({'Valid/Complete/'+k: v for k, v in stat_valid_cpl.items()})
                log_dict.update({'Best/Complete/'+k: v for k, v in best['cpl'].items() if k != 'Model'})
            wandb.log(log_dict)
        
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(scheduler_criterion)  # validation completion loss if we do completion, else classification loss
        elif scheduler is not None:
            scheduler.step()

        if early_stop_patience is not None and patience_cnt > early_stop_patience:
            if verbose:
                print(f'Early stop at epoch {epoch}.')
            break

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    #if wandb_log:
    #    if classify:
    #        wandb.config.update({'BestClassify'+k: v for k, v in best['clf'].items() if k != 'Model'})
    #    if complete:
    #        wandb.config.update({'BestComplete'+k: v for k, v in best['cpl'].items() if k != 'Model'})
    
    print('==== Best Result ====')
    for k in best:
        for k1 in best[k]:
            if k1 == 'BestEpoch':
                print(f"{k}_{k1}: {best[k][k1]}")
            elif k1 != 'Model':
                print(f"{k}_{k1}: {float(best[k][k1]):.8f}")

    return best