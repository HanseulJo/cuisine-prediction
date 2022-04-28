import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import wandb

from utils import _concatenate

def statistics(model, criterion, phase, dataloaders, dataset_sizes, device, k=5):
    running_loss = running_acc = running_top_k_acc = 0.
    running_labels = running_preds = None
    for idx, (feature_boolean, _, labels_int) in enumerate(dataloaders[phase]):
        batch_size = feature_boolean.size(0)
        feature_boolean = feature_boolean.to(device)
        labels_int = labels_int.to(device)
        outputs_clf, outputs_cpl = model(feature_boolean)
        if phase in ['train_eval', 'valid_clf']:
            _, preds = torch.max(outputs_clf, 1)
        elif phase in ['valid_cpl']:
            _, preds = torch.max(outputs_cpl, 1)
        
        if idx == 0:
            print('label', labels_int.cpu().numpy()[:10])
            print('preds', preds.cpu().numpy()[:10])
        
        if phase in ['train_eval', 'valid_clf']:
            running_loss += criterion(outputs_clf, labels_int.long()) * batch_size
            running_acc += accuracy_score(labels_int.cpu().numpy(), preds.cpu().numpy(), normalize=False)
            running_top_k_acc += top_k_accuracy_score(labels_int.cpu().numpy(), outputs_clf.cpu().numpy(), k=k, labels=np.arange(outputs_clf.size(1)), normalize=False)
            running_labels = _concatenate(running_labels, labels_int)
            running_preds = _concatenate(running_preds, preds)
        elif phase in ['valid_cpl']:
            # labels_int -= 1
            running_loss += criterion(outputs_cpl, labels_int.long()) * batch_size
            running_acc += accuracy_score(labels_int.cpu().numpy(), preds.cpu().numpy(), normalize=False)
            running_top_k_acc += top_k_accuracy_score(labels_int.cpu().numpy(), outputs_cpl.cpu().numpy(), k=k, labels=np.arange(outputs_cpl.size(1)), normalize=False)
            
    stat = dict(
        Loss = running_loss / dataset_sizes[phase],
        Acc = running_acc / dataset_sizes[phase],
        Topk = running_top_k_acc / dataset_sizes[phase]
    )
    if phase in ['train_eval', 'valid_clf']:
        stat.update(dict(
            F1macro = f1_score(running_labels, running_preds, average='macro'),
            F1micro = f1_score(running_labels, running_preds, average='micro')))
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
        best['clf'] = dict(Loss=float('inf'), Acc=-1., Topk=-1.)
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
            
            if complete:
                # 재료가 1개인 recipe는 없앤다.
                single_ingreds = feature_boolean.sum(1) > 1
                feature_boolean = feature_boolean[single_ingreds]
                labels_clf = labels_clf[single_ingreds]
                batch_size, num_ingreds = feature_boolean.size()
                # 재료 하나씩 없앤다.
                labels_cpl = torch.zeros(batch_size).long()
                for batch in range(batch_size):
                    recipe = np.arange(num_ingreds)[feature_boolean[batch] == 1]
                    removed_ingred = np.random.choice(recipe)
                    feature_boolean[batch][removed_ingred] = 0
                    labels_cpl[batch] = removed_ingred  # 없앤 재료 idx를 label로 학습
                labels_cpl = labels_cpl.to(device)

            feature_boolean = feature_boolean.to(device)
            labels_clf = labels_clf.to(device)
             
            optimizer.zero_grad()

            # forward
            outputs_clf, outputs_cpl = model(feature_boolean)

            loss = 0.
            if classify:
                _, preds_clf = torch.max(outputs_clf, 1)
                loss_clf = criterion(outputs_clf, labels_clf.long())
                loss = loss_clf
            if complete: 
                _, preds_cpl = torch.max(outputs_cpl, 1)
                loss_cpl = criterion(outputs_cpl, labels_cpl.long())
                loss += loss_cpl
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # gradient clipping
            optimizer.step()
            
            if verbose and idx % 100 == 0:
                log_str = f"    {idx * 100 // len(dataloaders['train']):3d}% | "
                if classify:
                    print('    *label_clf', labels_clf.cpu().numpy()[:12])
                    print('    *preds_clf', preds_clf.cpu().numpy()[:12])
                    log_str += " Loss_clf {:.4f} | Acc_clf {:.4f}".format(
                        loss_clf.item(), accuracy_score(labels_clf.cpu().numpy(), preds_clf.cpu().numpy()))
                if complete:
                    print('    >label_cpl', labels_cpl.cpu().numpy()[:12])
                    print('    >preds_cpl', preds_cpl.cpu().numpy()[:12])
                    log_str += " Loss_cpl {:.4f} | Acc_cpl {:.4f}".format(
                        loss_cpl.item(), accuracy_score(labels_cpl.cpu().numpy(), preds_cpl.cpu().numpy()))
                print(log_str)

        if wandb_log:
            wandb.watch(model)
            
        # statistics
        model.eval()
        with torch.set_grad_enabled(False):
            if classify:
                stat_train = statistics(model, criterion, 'train_eval', dataloaders, dataset_sizes, device, k=5)
                stat_valid_clf = statistics(model, criterion, 'valid_clf', dataloaders, dataset_sizes, device, k=5)
                if verbose:
                    print("TRAIN_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_train.items()]))
                    print("VALID_CLF", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_clf.items()]))
            if complete:
                stat_valid_cpl = statistics(model, criterion, 'valid_cpl', dataloaders, dataset_sizes, device, k=10)
                if verbose:
                    print("VALID_CPL", " ".join([f"{k} {v:.4f}" for k, v in stat_valid_cpl.items()]))
        
        scheduler.step(-stat_valid_cpl['Acc'] if complete else -stat_valid_clf['Acc'])
        is_new_best = stat_valid_cpl['Acc'] > best['cpl']['Acc'] if complete else stat_valid_clf['Acc'] > best['clf']['Acc']
        if is_new_best:
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
                log_dict.update({'ValidComplete'+k: v for k, v in stat_valid_clf.items()})
            wandb.log(log_dict)              
        
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