from __future__ import print_function, division

import torch
import os
import argparse
import pickle
# from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

from src.data import RecipeDataset
from src.model.nn import DNN
from src.inference import inference


def main(args):

    task = args.task
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    seed = args.seed
    batch_size = args.batch_size
    # step_size = args.step_size
    fc_layer_sizes = args.fc_layer_sizes

    test_recipe_dataset = RecipeDataset(os.path.join(data_dir, dataset_name), task) # RecipeDataset('./val_trai_cpl', task)

    dataloader = torch.utils.data.DataLoader(test_recipe_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)

    class_names = test_recipe_dataset.classes
    print(f'{task} task: ', class_names)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    if not torch.cuda.is_available():
        raise EnvironmentError('Not using cuda!')

    # Get a batch of training data
    if not 'test' in dataset_name:
        inputs, classes = next(iter(dataloader))
        print('inputs.shape', inputs.shape)
        print('classes.shape', classes.shape)
    else:
        inputs = next(iter(dataloader))
        print('inputs.shape', inputs.shape)

    if task == 'classification':
        output_size = len(test_recipe_dataset.classes)
    elif task == 'completion':
        output_size = test_recipe_dataset.classes
    model = DNN(input_size=len(inputs[0]), output_size=output_size,
                   fc_layer_sizes=fc_layer_sizes, dropout=0.2).to(device)
    for weight_dir, _, files in os.walk('./weights/'):
        for file in files:
            if task in file and str(batch_size) in file and str('-'.join(list(map(str, args.fc_layer_sizes)))) in file:
                model.load_state_dict(torch.load(os.path.join(weight_dir, file)))
                break

    print(model)

    preds, logits = inference(model, dataset_name, dataloader, device=device, task=task)
    # add dim 2 and concat
    inferences = torch.concat((preds.unsqueeze(dim=2), logits.unsqueeze(dim=2)), dim=2)

    fname = ['inference', args.dataset_name, args.task, 'DNN',
             'fc_layer_sizes', str('-'.join(list(map(str, args.fc_layer_sizes)))),
             'batch', str(args.batch_size), 'seed', str(args.seed)]
    fname = '_'.join(fname) + '.pkl'
    with open(os.path.join('../Recs', fname), 'wb') as f:
        recs = {}
        inferences_list = inferences.tolist()
        for query_id, rec_list in tqdm(enumerate(inferences_list)):
            recs[query_id] = [tuple(l) for l in rec_list]
        pickle.dump(recs, f)
    print('inferences.shape', inferences.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN Inference.')

    parser.add_argument('-t', '--task', default='completion', choices=['classification', 'completion'],
                        help='define the task of training: \'classification\' or \'completion\'.')
    parser.add_argument('-n', '--dataset_name', default='train',
                        choices=['train', 'valid_clf', 'valid_cpl', 'test_clf', 'test_cpl'],
                        help='define the task of training: \'classification\' or \'completion\'.')
    parser.add_argument('-d', '--data_dir', default='../Container/',
                        type=str,
                        help='path to the dataset.')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size for training.')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed number.')
    parser.add_argument('-f', '--fc_layer_sizes', default=[1024, 1024, 512, 512], type=lambda x: len(x)>=1,  # [2048, 1024, 512, 256]
                                            help='size of fc layers.')

    main(parser.parse_args())