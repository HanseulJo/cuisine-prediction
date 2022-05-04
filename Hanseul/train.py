import time
from copy import deepcopy
#from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import wandb

from utils import _concatenate, get_variables

def statistics(model, criterion, phase, dataloaders, dataset_sizes, device, k=5, verbose=True):
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
            running_loss += criterion(outputs_clf, labels_int.long()) * batch_size
            running_top_k_acc += top_k_accuracy_score(labels_int.cpu().numpy(), outputs_clf.cpu().numpy(), k=k, labels=np.arange(outputs_clf.size(1)), normalize=False)
        elif phase in ['valid_cpl']:
            # labels_int -= 1  # if completion answer index starts from 1, uncomment this line.
            running_loss += criterion(outputs_cpl, labels_int.long()) * batch_size
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
         device='cpu', num_epochs=20, early_stop_patience=None,
         random_seed=1, wandb_log=False, verbose=True):
    
    classify = model.classify
    complete = model.complete
    
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.seed_all()
    
    since = time.time()
    
    # BEST MODEL SAVING
    best = {}
    if classify:
        best['clf'] = dict(Loss=float('inf'), Acc=-1., Topk=-1., F1macro=-1., F1micro=-1.)
    if complete:
        best['cpl'] = dict(Loss=float('inf'), Acc=-1., Topk=-1., F1macro=-1., F1micro=-1.)
    best_model_wts = deepcopy(model.state_dict())

    if early_stop_patience is not None:
        patience_cnt = 0
    
    iterator, _temp = (range(1,num_epochs+1), None) if not verbose else (tqdm(range(1,num_epochs+1)), print('-' * 5 + 'Training the model' + '-' * 5))
    for epoch in iterator:
        None if not verbose else print(f'\nEpoch {epoch}/{num_epochs}')
        
        model.train()  # Set model to training mode
        
        for idx, (feature_boolean, _, labels_clf) in enumerate(dataloaders['train']):
            feature_boolean = feature_boolean.to(device)
            int_x, feasible, pad_mask, token_mask, labels_cpl = get_variables(feature_boolean, complete=complete, phase='train', cpl_scheme=model.cpl_scheme)
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
            val_loss, best_loss = 0., 0.
            if classify:
                stat_train = statistics(model, criterion, 'train_eval', dataloaders, dataset_sizes, device, k=5, verbose=verbose)
                if verbose:
                    print("TRAIN_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_train.items()]))
                stat_valid_clf = statistics(model, criterion, 'valid_clf', dataloaders, dataset_sizes, device, k=5, verbose=verbose)
                val_loss += stat_valid_clf['Loss']
                best_loss += best['clf']['Loss']
                if verbose:
                    print("VALID_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_clf.items()]))
            if complete:
                stat_valid_cpl = statistics(model, criterion, 'valid_cpl', dataloaders, dataset_sizes, device, k=10, verbose=verbose)
                if verbose:
                    print("VALID_CPL", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_cpl.items()]))
                val_loss += stat_valid_cpl['Loss']
                best_loss += best['cpl']['Loss']
        
        if val_loss < best_loss:
            if classify:
                best['clf'].update(stat_valid_clf)
            if complete:
                best['cpl'].update(stat_valid_cpl)
            best['bestEpoch'] = int(epoch)
            best_model_wts = deepcopy(model.state_dict()) # deep copy the model
            if early_stop_patience is not None:
                patience_cnt = 0
        elif early_stop_patience is not None:
            patience_cnt += 1

        if wandb_log:
            log_dict = {'learning_rate': optimizer.param_groups[0]['lr']} # scheduler.get_last_lr()[0] for CosineAnnealingWarmRestarts
            if classify:
                log_dict.update({'TrainClassify'+k: v for k, v in stat_train.items()})
                log_dict.update({'ValidClassify'+k: v for k, v in stat_valid_clf.items()})
            if complete:
                log_dict.update({'ValidComplete'+k: v for k, v in stat_valid_cpl.items()})
            wandb.log(log_dict)
        
        scheduler.step(val_loss)
        
        if early_stop_patience is not None and patience_cnt > early_stop_patience:
            if verbose:
                print(f'Early stop at epoch {epoch}.')
            break

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==== Best Result ====')
    for k in best:
        if k == 'bestEpoch':
            print(f"{k}: {int(best[k])}")
        else:
            for k1 in best[k]:
                print(f"{k}_{k1}: {float(best[k][k1]):.8f}")
    # load best model weights
    model.load_state_dict(best_model_wts)
    best_out = {'bestEpoch': best['bestEpoch']}
    if classify:
        best_out.update({'Clf'+k: v for k, v in stat_valid_clf.items()})
    if complete:
        best_out.update({'Cpl'+k: v for k, v in stat_valid_cpl.items()})
    return model, best_out