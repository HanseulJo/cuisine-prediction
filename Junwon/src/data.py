import os
from torch.utils.data import Dataset
import h5py

class RecipeDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.data_file = h5py.File(data_dir, 'r')
        self.features = self.data_file['features_one_hot'][:]
        self.labels = self.data_file['labels_int_enc'][:]
        self.classes = sorted(list(set(self.labels)))
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return feature, label