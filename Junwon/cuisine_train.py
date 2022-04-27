from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics import F1Score
import numpy as np
import torch
import os
import argparse
import wandb
# from sklearn.metrics import top_k_accuracy_score

from src.train import train
from src.data import RecipeDataset
from src.model.nn import DNN
from src.visualize import visualize_confusion_matrix
from src.metrics import top_k_accuracy



# class TargetTransform():
#     def __init__(self, classes, class_to_idx):
#         self.classes = list()
#         self.idx_to_class = {v: k for k, v in class_to_idx.items()}
#         for c in classes:
#             for word in c.split(' '):
#                 self.classes.append(word)
#         self.classes = list(set(self.classes))
#         self.classes.sort()
#         self.class_to_target = {_class: idx for idx, _class in enumerate(self.classes)}
#
#     def target_transformation(self, target):
#         _class = self.idx_to_class[target]
#         target = np.zeros(len(self.classes))
#         for c in _class.split(' '):
#             target[self.class_to_target[c]] = 1
#
#         return target.astype(np.float32)


def main(args):
    if args.wandb_log:
        wandb.init(project=f'cuisine_{args.task}', config=args)
        args = wandb.config

    task = args.task
    data_dir = args.data_dir
    seed = args.seed
    batch_size = args.batch_size
    step_size = args.step_size
    fc_layer_sizes = args.fc_layer_sizes

    dataset_name = ['train', 'val']

    recipe_datasets = {'train': RecipeDataset(os.path.join(data_dir, 'train'), task),
                       'val': RecipeDataset(os.path.join(data_dir,
                                                         'valid_clf' if task=='classification' else 'valid_cpl'), task)}
    # recipe_datasets = {x: RecipeDataset(os.path.join(data_dir, x))
    #                   for x in dataset_name}
    dataloaders = {x: torch.utils.data.DataLoader(recipe_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=8)
                  for x in dataset_name}
    dataset_sizes = {x: len(recipe_datasets[x]) for x in dataset_name}
    class_names = recipe_datasets['train'].classes
    print(f'{task} task: ', class_names)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    if not torch.cuda.is_available():
        raise EnvironmentError('Not using cuda!')

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    print('inputs.shape', inputs.shape)
    print('classes.shape', classes.shape)

    # 'og', 'transfer', 'finetune'
    if task == 'classification':
        output_size = len(recipe_datasets['train'].classes)
    elif task == 'completion':
        output_size = recipe_datasets['train'].classes
    model_ft = DNN(input_size=len(inputs[0]), output_size=output_size,
                   fc_layer_sizes=fc_layer_sizes, dropout=0.2).to(device)

    print(model_ft)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.AdamW(model_ft.parameters(),
                                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

    # LR scheduler
    # lr_scheduler.StepLR(optimizer_ft, args.step_size, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft,
    #                                                             T_0=10, T_mult=1, eta_min=0, last_epoch=-1)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=step_size,
                                                      eps=1e-08, verbose=True)

    # (metric) F1Score objects
    if task == 'classification':
        macro_f1 = F1Score(num_classes=len(recipe_datasets['train'].classes), average='macro')
        micro_f1 = F1Score(num_classes=len(recipe_datasets['train'].classes), average='micro')
        metrics = {'macro_f1': macro_f1, 'micro_f1': micro_f1}
    elif task == 'completion':
        topk_acc = top_k_accuracy
        metrics = {'topk_acc': topk_acc}

    model_ft = train(model_ft, dataloaders, criterion, optimizer, exp_lr_scheduler, metrics,
                     dataset_sizes, device, num_epochs=args.n_epochs, task=task,
                     wandb_log=args.wandb_log, early_stop_patience=20)


    fname = ['ckpt', task, 'DNN', 'batch', str(args.batch_size),
             'n_epochs', str(args.n_epochs), 'lr', str(args.lr), 'step_size', str(args.step_size),
             'seed', str(args.seed)]
    fname = '_'.join(fname) + '.pt'
    if not os.path.isdir('./weights/'):
        os.mkdir('./weights/')
    torch.save(model_ft.state_dict(), os.path.join('./weights/', fname))
    if args.wandb_log:
        wandb.save(os.path.join('./weights/', 'ckpt*'))

    if task == 'classification':
        visualize_confusion_matrix(model_ft, dataloaders['val'], class_names, args, device, data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thumbnail dataset image classification.')

    parser.add_argument('-t', '--task', default='completion', choices=['classification', 'completion'],
                        help='define the task of training: \'classification\' or \'completion\'.')
    parser.add_argument('-d', '--data_dir', default='../Chanho/Container/',
                        type=str,
                        help='path to the dataset.')
    # parser.add_argument('-l', '--learning_type', default='finetune', choices=['og', 'transfer', 'finetune'],
    #                     help='learning type: og (learning from scratch. original), transfer (transfer learning), finetune (fine tuning).')
    # parser.add_argument('-r', '--split_ratio', default=(0.8, 0.2), type=lambda x: len(x) in [2],# [2, 3],
    #                     help='ratio for train/validation(/test) dataset splitting.')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size for training.')
    parser.add_argument('-e', '--n_epochs', default=50
                        , type=int,
                        help='number of epochs for training.')
    parser.add_argument('-lr', '--lr', default=1e-3, type=float,
                        help='learning rate for training optimizer.')
    parser.add_argument('-step', '--step_size', default=10, type=int,
                        help='step size for training scheduler.')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed number.')
    parser.add_argument('-f', '--fc_layer_sizes', default=[1024, 1024, 512, 256], type=lambda x: len(x)>=1,  # [1024, 256, 256, 128]
                                            help='size of fc layers.')
    # parser.add_argument('-ext', '--image_extensions',
    #                     default=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
    #                     type=tuple, help='tuple of image extensions.')

    parser.add_argument('-w', '--wandb_log', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to log in wandb.')

    # parser.add_argument('-t', '--task', default='multiclass', choices=['multiclass', 'multilabel'],
    #                     help='define the task of training: \'multilabel\' or \'multiclass\' classification')

    main(parser.parse_args())