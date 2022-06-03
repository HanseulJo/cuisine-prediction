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

    print('-' * 5 + 'Model Inference' + '-' * 5)
    model.eval()  # Set model to evaluate mode

    running_preds = None
    running_logits = None

    # Iterate over data.
    for idx, samples in enumerate(dataloader):
        if not 'test' in dataset_name:
            inputs, labels = samples
        else:
            inputs = samples
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            topk_logits, topk_preds = torch.topk(F.softmax(outputs), 10, 1)
            if idx == 0:
                # print('labels', labels[0])
                print('outputs', outputs[0])
                print('topk_preds', topk_preds[0])
                print('topk_logits', topk_logits[0])

        if running_preds is not None:
            running_preds = torch.concat((running_preds, topk_preds.clone().detach().cpu()), dim=0)
        else:
            running_preds = topk_preds.clone().detach().cpu() #.numpy()
        if running_logits is not None:
            running_logits = torch.concat((running_logits, topk_logits.clone().detach().cpu()), dim=0)
        else:
            running_logits = topk_logits.clone().detach().cpu() #.numpy()


    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return running_preds.type(torch.int), running_logits
