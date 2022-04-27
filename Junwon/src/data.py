import os
from torch.utils.data import Dataset
import h5py
import numpy as np

class RecipeDataset(Dataset):
    def __init__(self, data_dir, task='classification', random_seed=0):
        if task not in ['classification', 'completion']:
            raise ValueError('Dataset: task should be whether \'classification\' or \'completion\'.')
        self.data_dir = data_dir
        self.data_file = h5py.File(data_dir, 'r')
        self.task = task
        self.random_seed = random_seed
        if 'train' in data_dir:
            self.dataset_type = 'train'
        elif 'val' in data_dir:
            self.dataset_type = 'val'
        else:
            raise ValueError('not train/val dataset!!! check data_dir..')

        if self.dataset_type == 'train':
            self.features = self.data_file['features_boolean'][:]
            self.labels = self.data_file['labels_int_enc'][:]
            if self.task == 'completion':
                # TODO: remove recipe len=1
                # remove recipies with ingredient length is 1
                indices_1 = np.where(np.add.reduce(self.data_file['features_boolean'][:], axis=1) == 1.)
                indices_1 = indices_1[0]
                for idx in indices_1:
                    print(f'recipe number {idx} deleted: only one ingredient.')
                    self.features = np.delete(self.features, idx, axis=0)
                    self.labels = np.delete(self.labels, idx, axis=0)

                self.features_one_indicies = list([] for _ in range(self.features.shape[0]))
                ones_idxs = np.nonzero(self.features)

                for recipe_idx, one_idx in zip(ones_idxs[0], ones_idxs[1]):
                    self.features_one_indicies[recipe_idx].append(one_idx)

        elif self.dataset_type == 'val':
            if task == 'classification':
                self.features = self.data_file['features_boolean'][:]
                self.labels = self.data_file['labels_int_enc'][:]
            elif task == 'completion':
                self.features = self.data_file['features_boolean'][:]
                self.labels = self.data_file['labels_id'][:]
                self.labels = self.labels.astype(int) - 1
                # idx_0dim, self.labels = np.nonzero(self.data_file['features_answer'][:])
                # assert len(idx_0dim) == self.data_file['features_answer'].shape[0]

        if task == 'classification':
            self.classes = sorted(list(set(self.labels)))
        elif task == 'completion':
            self.classes = len(self.labels[0]) if self.dataset_type == 'validation' else 6714
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dataset_type == 'train' and self.task == 'completion':
            np.random.seed(seed=self.random_seed)
            rand_idx = np.random.randint(0, len(self.features_one_indicies[idx]))
            feature = self.features[idx].copy()
            feature[rand_idx] = 0
            label = self.features_one_indicies[idx][rand_idx]
            # label = np.zeros(len(self.features[idx]), dtype=np.float64)
            # label[rand_idx] = 1
        else:
            feature = self.features[idx]
            label = self.labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return feature, label