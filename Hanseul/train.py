import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score


def train(model,
          dataloaders,
          criterion,
          optimizer,
          scheduler,
          #metrics,
          dataset_sizes,
          device='cpu',
          num_epochs=20,
          #wandb_log=False,
          early_stop_patience=None,
          classify=True,
          complete=True,
          random_seed=1):

    def _concatenate(running_v, new_v):
        if running_v is not None:
            return np.concatenate((running_v, new_v.clone().detach().cpu().numpy()), axis=0)
        else:
            return new_v.clone().detach().cpu().numpy()
    
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    since = time.time()

    best_model_wts = deepcopy(model.state_dict())
    best_loss = 1e4
    
    if early_stop_patience is not None:
        if not isinstance(early_stop_patience, int):
            raise TypeError('early_stop_patience should be an integer.')
        patience_cnt = 0
    
    print('-' * 5 + 'Training the model' + '-' * 5)
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        val_loss = 0. # sum of classification and completion loss

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid_class', 'valid_compl']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                if not classify and phase == 'valid_class':
                    continue
                elif not complete and phase == 'valid_compl':
                    continue

            running_loss_class = 0.
            running_loss_compl = 0.
            running_corrects_compl = 0.
            running_corrects_class = 0.
            running_labels_class = None
            running_preds_class = None
            
            # Iterate over data.
            for idx, (bin_inputs, int_inputs, bin_labels, int_labels) in enumerate(dataloaders[phase]):
                batch_size = bin_inputs.size(0)
                if classify and phase in ['train', 'valid_class']:
                    labels_class = int_labels.to(device)
                if complete:
                    # randomly remove one ingredient for each recipe/batch
                    if phase == 'train':
                        labels_compl = torch.zeros_like(int_labels)
                        for batch in range(batch_size):
                            ingreds = torch.arange(bin_inputs.size(-1))[bin_inputs[batch]==1]
                            if len(ingreds) < 2:
                                raise RuntimeError("Train data has a single-ingredient recipe")
                            mask_ingred_idx = ingreds[np.random.randint(len(ingreds))]
                            bin_inputs[batch][mask_ingred_idx] = 0
                            int_inputs[batch][int_inputs[batch] == mask_ingred_idx] = int(bin_inputs.size(-1))
                            labels_compl[batch] = mask_ingred_idx
                        labels_compl = labels_compl.to(device)
                    elif phase == 'valid_compl':
                        labels_compl = int_labels.to(device)
                bin_inputs = bin_inputs.to(device)
                int_inputs = int_inputs.to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_class, outputs_compl = model(int_inputs, bin_x=bin_inputs)  # bin_x 없어도 작동은 가능
                    if classify and phase in ['train', 'valid_class']:
                        _, preds_class = torch.max(outputs_class, 1)
                    if complete and phase in ['train', 'valid_compl']:
                        _, preds_compl = torch.max(outputs_compl, 1)

                    if idx == 0 and phase == 'train':  # 원래 idx == 0 
                        if classify and phase in ['train', 'valid_class']:
                            print('labels_classification', labels_class.cpu().numpy())
                            #print('outputs_classification', outputs_class.clone().detach().cpu().numpy())
                            print('preds_classification', preds_class.cpu().numpy())
                        if complete and phase in ['train', 'valid_compl']:
                            print('labels_completion', labels_compl.cpu().numpy())
                            #print('outputs_completion', outputs_compl[0])
                            print('preds_completion', preds_compl.cpu().numpy())
                    
                    if classify and phase in ['train', 'valid_class']:
                        loss_class = criterion(outputs_class, labels_class.long())
                    if complete and phase in ['train', 'valid_compl']:
                        loss_compl = criterion(outputs_compl, labels_compl.long())

                    if classify and complete and phase == 'train':
                        loss = loss_class + loss_compl
                    elif classify and phase in ['train', 'valid_class']:
                        loss = loss_class
                    elif complete and phase in ['train', 'valid_compl']:
                        loss = loss_compl

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping
                        optimizer.step()

                if idx % 100 == 0:
                    log_str = f'    {phase} {idx * 100 // len(dataloaders[phase]):3d}% of an epoch | '
                    if classify and phase in ['train', 'valid_class']:
                        log_str += f'Loss(classif.): {loss_class.item():.4f} | '
                    if complete and phase in ['train', 'valid_compl']:
                        log_str += f'Loss(complet.): {loss_compl.item():.4f} | '
                    print(log_str)

                # statistics
                if classify and phase in ['train', 'valid_class']: # for F1 score & accuracy
                    running_loss_class += loss_class.item() * batch_size
                    running_labels_class = _concatenate(running_labels_class, labels_class)
                    running_preds_class = _concatenate(running_preds_class, preds_class)
                    running_corrects_class += torch.sum(preds_class == labels_class.data)
                if complete and phase in ['train', 'valid_compl']: # for accuracy
                    running_loss_compl += loss_compl.item() * batch_size
                    running_corrects_compl += torch.sum(preds_compl == labels_compl.data)


            epoch_loss = 0.
            log_str = f'{phase.upper()} | '
            if classify and phase in ['train', 'valid_class']:
                epoch_loss_class = running_loss_class / dataset_sizes[phase]
                epoch_loss += epoch_loss_class
                running_labels_class = torch.from_numpy(running_labels_class)
                running_preds_class = torch.from_numpy(running_preds_class)
                epoch_macro_f1 = f1_score(running_labels_class, running_preds_class, average='macro')  # classification: f1 scores.
                epoch_micro_f1 = f1_score(running_labels_class, running_preds_class, average='micro')
                epoch_acc_class = running_corrects_class / dataset_sizes[phase]
                log_str += f'Loss(classif.): {epoch_loss_class:.3f} Acc(classif.): {epoch_acc_class:.3f} Macro-F1: {epoch_macro_f1:.3f} Micro-F1: {epoch_micro_f1:.3f} | '
            if complete and phase in ['train', 'valid_compl']:
                epoch_loss_compl = running_loss_compl / dataset_sizes[phase]
                epoch_loss += epoch_loss_compl
                epoch_acc_compl = running_corrects_compl / dataset_sizes[phase]  # completion task: accuracy.
                log_str += f'Loss(complet.): {epoch_loss_compl:.3f} Acc(complet.): {epoch_acc_compl:.3f} | '
            print(log_str)
            
            if phase == 'train':
                train_loss = epoch_loss
                if classify:
                    train_macro_f1 = epoch_macro_f1
                    train_micro_f1 = epoch_micro_f1
                # if wandb_log:
                #     wandb.watch(model)
            elif 'val' in phase:
                val_loss += epoch_loss
                if classify and phase == 'valid_class':
                    val_macro_f1 = epoch_macro_f1
                    val_micro_f1 = epoch_micro_f1
            
        if 'val' in phase:
            scheduler.step(val_loss)
            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = deepcopy(model.state_dict())
                if early_stop_patience is not None:
                    patience_cnt = 0
            elif early_stop_patience is not None:
                patience_cnt += 1

        """
        if wandb_log:
            wandb.log({'train_loss': train_loss,
                       'val_loss': val_loss,
                       'train_macro_f1': train_macro_f1,
                       'train_micro_f1': train_micro_f1,
                       'val_macro_f1': val_macro_f1,
                       'val_micro_f1': val_micro_f1,
                       'best_val_loss': best_loss,
                       'learning_rate': optimizer.param_groups[0]['lr']})
                                        # scheduler.get_last_lr()[0] for CosineAnnealingWarmRestarts
        """
        if early_stop_patience is not None:
            if patience_cnt > early_stop_patience:
                print(f'Early stop at epoch {epoch}.')
                break
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model