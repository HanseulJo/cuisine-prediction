import torch
import time
import copy
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn.functional as F

def inference(model, dataset_name, dataloader, device='cpu', task='classification'):
    # if not isinstance(metrics, dict):
    #     raise TypeError(f'\'metrics\' argument should be a dictionary, but {type(metrics)}.')
    if task not in ['classification', 'completion']:
        raise ValueError(f'{task} should be either \'classification\' or \'completion\'.')

    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 1000.0

    # if early_stop_patience is not None:
    #     if not isinstance(early_stop_patience, int):
    #         raise TypeError('early_stop_patience should be an integer.')
    #     patience_cnt = 0

    print('-' * 5 + 'Model Inference' + '-' * 5)
    # for epoch in tqdm(range(num_epochs)):
    #     print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    # Each epoch has a training and validation phase
    # for phase in ['train', 'val']:
    #     if phase == 'train':
    #         model.train()  # Set model to training mode
    #     else:
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_labels = None
    # if task == 'classification':
    running_preds = None
    # elif task == 'completion':
    running_logits = None

    # Iterate over data.
    for idx, samples in enumerate(dataloader):
        if not 'test' in dataset_name:
            inputs, labels = samples
        else:
            inputs = samples
        inputs = inputs.to(device)
        # labels = labels.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward
        # track history if only in train
        # with torch.set_grad_enabled(phase == 'train'):
        with torch.no_grad():
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            topk_logits, topk_preds = torch.topk(F.softmax(outputs), 10, 1)
            # if task == 'completion':
            #     logits, preds_k = torch.topk(outputs, 10)
            if idx == 0:
                # print('labels', labels[0])
                print('outputs', outputs[0])
                print('topk_preds', topk_preds[0])
                print('topk_logits', topk_logits[0])
            # loss = criterion(outputs, labels.long())

            # backward + optimize only if in training phase
            # if phase == 'train':
            #     loss.backward()
            #     optimizer.step()

        # statistics
        # running_loss += loss.item() * inputs.size(0)
        # if running_labels is not None:
        #     # running_labels = np.concatenate((running_labels, labels.clone().detach().cpu().numpy()), axis=0)
        #     running_labels = torch.concat((running_labels, labels.clone().detach().cpu()), dim=0)
        # else:
        #     running_labels = labels.clone().detach().cpu() #.numpy()
        # if task == 'classification':
        if running_preds is not None:
            # running_preds = np.concatenate((running_preds, preds.clone().detach().cpu().numpy()), axis=0)
            running_preds = torch.concat((running_preds, topk_preds.clone().detach().cpu()), dim=0)
        else:
            running_preds = topk_preds.clone().detach().cpu() #.numpy()
        # elif task == 'completion':
        if running_logits is not None:
            # running_logits = np.concatenate((running_logits,
            #                                  F.softmax(outputs).clone().detach().cpu().numpy()), axis=0)
            running_logits = torch.concat((running_logits, topk_logits.clone().detach().cpu()), dim=0)
        else:
            running_logits = topk_logits.clone().detach().cpu() #.numpy()
            # running_corrects += torch.sum(preds == labels.data)


    # epoch_loss = running_loss / dataset_size
    # epoch_acc = running_corrects.double() / dataset_sizes[phase]
    # if task == 'classification':
    #     # running_labels = torch.from_numpy(running_labels)
    #     # running_preds = torch.from_numpy(running_preds)
    #     epoch_macro_f1 = metrics['macro_f1'](running_labels, running_preds)
    #     epoch_micro_f1 = metrics['micro_f1'](running_labels, running_preds)
    #     print('Test Loss: {:.4f} Macro-F1: {:.4f} Micro-F1: {:.4f}'.format(
    #         epoch_loss, epoch_macro_f1, epoch_micro_f1))
    # elif task == 'completion':
    #     epoch_topk_acc = metrics['topk_acc'](running_labels, running_logits)
    #     print('Test Loss: {:.4f} Top-k-Acc: {:.4f}'.format(
    #         epoch_loss, epoch_topk_acc))

    # log train results
    # if phase == 'train':
    #     train_loss = epoch_loss
    #     if task == 'classification':
    #         train_macro_f1 = epoch_macro_f1
    #         train_micro_f1 = epoch_micro_f1
    #     elif task == 'completion':
    #         train_topk_acc = epoch_topk_acc
    #     if wandb_log:
    #         wandb.watch(model)
    # elif phase == 'val':
    # test_loss = epoch_loss
    # if task == 'classification':
    #     test_macro_f1 = epoch_macro_f1
    #     test_micro_f1 = epoch_micro_f1
    # elif task == 'completion':
    #     test_topk_acc = epoch_topk_acc

    # if phase == 'train':
    #     scheduler.step()
    # if phase == 'val':
    #     scheduler.step(val_loss)

    # if phase == 'val':
    #     # deep copy the model
    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         best_model_wts = copy.deepcopy(model.state_dict())
    #         if early_stop_patience is not None:
    #             patience_cnt = 0
    #     elif early_stop_patience is not None:
    #         patience_cnt += 1

    # if wandb_log:
    #     log_dict = {'test_loss': test_loss,}
    #                # 'val_loss': val_loss,
    #                # 'best_val_loss': best_loss,
    #                # 'learning_rate': optimizer.param_groups[0]['lr']}
    #                 # scheduler.get_last_lr()[0] for CosineAnnealingWarmRestarts
    #     if task == 'classification':
    #         log_dict['test_macro_f1'] = test_macro_f1
    #         log_dict['test_micro_f1'] = test_micro_f1
    #         # log_dict['val_macro_f1'] = val_macro_f1
    #         # log_dict['val_micro_f1'] = val_micro_f1
    #     elif task == 'completion':
    #         log_dict['test_topk_acc'] = test_topk_acc
    #         # log_dict['val_topk_acc'] = val_topk_acc
    #     wandb.log(log_dict)

    # if early_stop_patience is not None:
    #     if patience_cnt > early_stop_patience:
    #         print(f'Early stop at epoch {epoch}.')
    #         break

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Loss: {:4f}'.format(best_loss))

    # after last epoch, generate confusion matrix of validation phase

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return running_preds.type(torch.int), running_logits
