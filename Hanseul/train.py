import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score, top_k_accuracy_score
import wandb
#from .utils import LogitSelector


def train(model,
          dataloaders,
          criterion,
          optimizer,
          scheduler,
          dataset_sizes,
          device='cpu',
          num_epochs=20,
          #wandb_log=False,
          early_stop_patience=None,
          classify=True,
          complete=True,
          random_seed=1,
          wandb_log=False):

    def _concatenate(running_v, new_v):
        if running_v is not None:
            return np.concatenate((running_v, new_v.clone().detach().cpu().numpy()), axis=0)
        else:
            return new_v.clone().detach().cpu().numpy()
    
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    
    since = time.time()
    
    # Logit selector
    #logit_selector = LogitSelector(rank=100).to(device)
    
    # BEST MODEL SAVING
    best = {'loss': float('inf')}
    if classify:
        best['F1micro'] = -1.
        best['F1macro'] = -1.
        best['top5cls'] = -1.
    if complete:
        best['acc'] = -1.
        best['top10cmp'] = -1.
    best_model_wts = deepcopy(model.state_dict())

    if early_stop_patience is not None:
        if not isinstance(early_stop_patience, int):
            raise TypeError('early_stop_patience should be an integer.')
        patience_cnt = 0
    
    print('-' * 5 + 'Training the model' + '-' * 5)
    for epoch in tqdm(range(1,num_epochs+1)):
        print(f'\nEpoch {epoch}/{num_epochs}')

        val_loss = 0. # sum of classification and completion loss

        # Each epoch has a training phase and two validation phases
        for phase in ['train', 'valid_class', 'valid_compl']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if not classify and phase == 'valid_class':
                    continue
                elif not complete and phase == 'valid_compl':
                    continue
                model.eval()   # Set model to evaluate mode

            running_loss_class = 0.
            running_corrects_class = 0.
            running_labels_class = None
            running_preds_class = None
            running_top_k_class = 0.
            
            running_loss_compl = 0.
            running_corrects_compl = 0.
            running_top_k_compl = 0.
            
            dataset_name = phase
            if phase == 'train':
                dataset_name = 'train_class' if classify and not complete else 'train_compl'
            
            # Iterate over data.
            for idx, loaded_data in enumerate(dataloaders[dataset_name]):
                if phase == 'train':
                    if complete:
                        bin_inputs, int_inputs, label_class, label_compl = loaded_data
                    else:
                        bin_inputs, int_inputs, label_class = loaded_data
                elif phase == 'valid_class':
                        bin_inputs, int_inputs, label_class = loaded_data
                elif phase == 'valid_compl':
                        bin_inputs, int_inputs, label_compl = loaded_data
                
                batch_size, num_items = bin_inputs.size()
                if classify and phase in ['train', 'valid_class']:
                    labels_class = label_class.to(device)
                if complete and phase in ['train', 'valid_compl']:
                    labels_compl = label_compl.to(device)
                bin_inputs = bin_inputs.to(device)
                int_inputs = int_inputs.to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_class, outputs_compl = model(int_inputs, bin_x=bin_inputs)
                    new_outputs_compl, new_labels_compl = None, None
                    if classify and phase in ['train', 'valid_class']:
                        _, preds_class = torch.max(outputs_class, 1)
                    if complete and phase in ['train', 'valid_compl']:
                        _, preds_compl = torch.max(outputs_compl, 1)
                        #new_outputs_compl, new_labels_compl = logit_selector(outputs_compl, labels_compl)
                        
                    """if idx == 0:
                        if classify and phase in ['train', 'valid_class']:
                            print('labels_classification', labels_class.cpu().numpy())
                            print('preds_classification', preds_class.cpu().numpy())
                        if complete and phase in ['train', 'valid_compl']:
                            if new_labels_compl is not None:
                                print('new label', new_labels_compl.cpu().numpy())
                            print('labels_completion', labels_compl.cpu().numpy())
                            print('preds_completion', preds_compl.cpu().numpy())"""
                    
                    if classify and phase in ['train', 'valid_class']:
                        loss_class = criterion(outputs_class, labels_class.long())
                    if complete and phase in ['train', 'valid_compl']:
                        if new_outputs_compl is None:
                            loss_compl = criterion(outputs_compl, labels_compl.long())
                        else:
                            loss_compl = criterion(new_outputs_compl, new_labels_compl)

                    if classify and complete and phase == 'train':
                        loss = loss_class + loss_compl
                    elif classify and phase in ['train', 'valid_class']:
                        loss = loss_class
                    elif complete and phase in ['train', 'valid_compl']:
                        loss = loss_compl

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping
                        optimizer.step()

                if idx % 100 == 0 and phase == 'train':
                    log_str = f'    {phase} {idx * 100 // len(dataloaders[dataset_name]):3d}% of an epoch | '
                    if classify and phase in ['train', 'valid_class']:
                        log_str += f'Loss(classif.): {loss_class.item():.4f} | '
                    if complete and phase in ['train', 'valid_compl']:
                        log_str += f'Loss(complet.): {loss_compl.item():.4f} | '
                    print(log_str)

                # statistics
                if classify and phase in ['train', 'valid_class']: # for F1 score
                    running_loss_class += loss_class.item() * batch_size
                    running_labels_class = _concatenate(running_labels_class, labels_class)
                    running_preds_class = _concatenate(running_preds_class, preds_class)
                    running_top_k_class += top_k_accuracy_score(labels_class.cpu().numpy(), outputs_class.detach().cpu().numpy(), k=5, labels=np.arange(outputs_class.size(1)), normalize=False)
                if complete and phase in ['train', 'valid_compl']: # for accuracy
                    running_loss_compl += loss_compl.item() * batch_size
                    running_corrects_compl += torch.sum(preds_compl == labels_compl)
                    running_top_k_compl += top_k_accuracy_score(labels_compl.cpu().numpy(), outputs_compl.detach().cpu().numpy(), k=10, labels=np.arange(outputs_compl.size(1)), normalize=False)

            epoch_loss = 0.
            log_str = f'{phase.upper()} | '
            if classify and phase in ['train', 'valid_class']:
                epoch_loss_class = running_loss_class / dataset_sizes[dataset_name]
                epoch_loss += epoch_loss_class
                running_labels_class = torch.from_numpy(running_labels_class)
                running_preds_class = torch.from_numpy(running_preds_class)
                epoch_macro_f1 = f1_score(running_labels_class, running_preds_class, average='macro')  # classification: f1 scores.
                epoch_micro_f1 = f1_score(running_labels_class, running_preds_class, average='micro')  # micro f1 score == accuracy, for single-label classification.
                epoch_top_k_class = running_top_k_class / dataset_sizes[dataset_name]
                log_str += f'Loss(classif.): {epoch_loss_class:.3f} Macro-F1: {epoch_macro_f1:.3f} Micro-F1: {epoch_micro_f1:.3f} Top-5 Acc: {epoch_top_k_class:.3f} | '
            if complete and phase in ['train', 'valid_compl']:
                epoch_loss_compl = running_loss_compl / dataset_sizes[dataset_name]
                epoch_loss += epoch_loss_compl
                epoch_acc_compl = running_corrects_compl / dataset_sizes[dataset_name]  # completion task: accuracy.
                epoch_top_k_compl = running_top_k_compl / dataset_sizes[dataset_name]
                log_str += f'Loss(complet.): {epoch_loss_compl:.3f} Acc(complet.): {epoch_acc_compl:.3f} Top-10 Acc: {epoch_top_k_compl:.3f} | '
            print(log_str)
            
            if phase == 'train':
                train_loss = epoch_loss
                if classify:
                    train_macro_f1 = epoch_macro_f1
                    train_micro_f1 = epoch_micro_f1
                    train_top_k_class = epoch_top_k_class
                if complete:
                    train_acc = epoch_acc_compl
                    train_top_k_compl = epoch_top_k_compl
                if wandb_log:
                    wandb.watch(model)
            elif 'val' in phase:
                val_loss += epoch_loss
                if classify and phase == 'valid_class':
                    val_macro_f1 = epoch_macro_f1
                    val_micro_f1 = epoch_micro_f1
                    val_top_k_class = epoch_top_k_class
                if complete and phase == 'valid_compl':
                    val_acc = epoch_acc_compl
                    val_top_k_compl = epoch_top_k_compl
        
        scheduler.step(-val_micro_f1 if classify and not complete else -val_acc)
        is_new_best = val_micro_f1 > best['F1micro'] if classify and not complete else val_acc > best['acc']
        if is_new_best:
            best['bestEpoch'] = int(epoch)
            best['loss'] = val_loss
            if classify:
                best['F1micro'] = val_micro_f1
                best['F1macro'] = val_macro_f1
                best['top5cls'] = val_top_k_class
            if complete:
                best['acc'] = val_acc
                best['top10cmp'] = val_top_k_compl
            best_model_wts = deepcopy(model.state_dict()) # deep copy the model
            if early_stop_patience is not None:
                patience_cnt = 0
        elif early_stop_patience is not None:
            patience_cnt += 1

        if wandb_log:
            log_dict = dict(train_loss=train_loss, val_loss=val_loss,
                            learning_rate=optimizer.param_groups[0]['lr'])
                            # scheduler.get_last_lr()[0] for CosineAnnealingWarmRestarts
            if classify:
                log_dict.updata(dict(train_macro_f1=train_macro_f1, train_micro_f1=train_micro_f1, train_top_k_class=train_top_k_class,
                                     val_macro_f1=val_macro_f1, val_micro_f1=val_micro_f1, val_top_k_class=val_top_k_class))
            if complete:
                log_dict.updata(dict(train_acc=train_acc, train_top_k_compl=train_top_k_compl,
                                     val_acc=val_acc, val_top_k_compl=val_top_k_compl))

            wandb.log(log_dict)              
        
        if early_stop_patience is not None:
            if patience_cnt > early_stop_patience:
                print(f'Early stop at epoch {epoch}.')
                break
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==== Best Result ====')
    for k in best:
        if k == 'bestEpoch':
            print(f"{k}: {int(best[k])}")
        else:
            print(f"{k}: {float(best[k]):.8f}")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best