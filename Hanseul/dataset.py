import h5py
import numpy as np
from torch.utils.data import Dataset


class RecipeDataset(Dataset):
    def __init__(self, data_dir, test=False):
        self.data_dir = data_dir
        self.test = test
        with h5py.File(data_dir, 'r') as data_file:
            self.bin_data = data_file['bin_data'][:]  # Size (num_recipes=23547, num_ingredients=6714)
            if not test:
                self.int_labels = data_file['int_labels'][:]  # Size (num_recipes=23547,), about cuisines
                self.bin_labels = data_file['bin_labels'][:]  # Size (num_recipes=23547, 20), about cuisines
        
        self.padding_idx = self.bin_data.shape[1]  # == num_ingredient == 6714
        self.max_num_ingredients_per_recipe = self.bin_data.sum(1).max()  # train의 경우 59, valid & test의 경우 65
        
        # (59나 65로) 고정된 길이의 row vector에 해당 recipe의 indices 넣고 나머지는 padding index로 채워넣기
        # self.int_data: Size (num_recipes=23547, self.max_num_ingredients_per_recipe=59 or 65)
        self.int_data = np.full((len(self.bin_data), self.max_num_ingredients_per_recipe), self.padding_idx) 
        for i, bin_recipe in enumerate(self.bin_data):
            recipe = np.arange(self.padding_idx)[bin_recipe==1]
            self.int_data[i][:len(recipe)] = recipe
        
    def __len__(self):
        return len(self.bin_data)

    def __getitem__(self, idx):
        bin_data = self.bin_data[idx]
        int_data = self.int_data[idx]
        bin_label = None if self.test else self.bin_labels[idx]
        int_label = None if self.test else self.int_labels[idx]
        
        return bin_data, int_data, bin_label, int_label