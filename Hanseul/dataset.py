import h5py
import numpy as np
from torch.utils.data import Dataset


class RecipeDataset(Dataset):
    def __init__(self, data_dir, test=False):
        self.data_dir = data_dir
        self.test = test
        self.classify, self.complete = False, False
        with h5py.File(data_dir, 'r') as data_file:
            self.bin_data = data_file['bin_data'][:]  # Size (num_recipes=23547, num_ingredients=6714)
            if 'label_class' in data_file.keys():
                self.classify = True
                self.label_class = data_file['label_class'][:]  
            if 'label_compl' in data_file.keys():
                self.complete = True
                self.label_compl = data_file['label_compl'][:]
        
        self.padding_idx = self.bin_data.shape[1]  # == num_ingredient == 6714
        self.max_num_ingredients_per_recipe = self.bin_data.sum(1).max()  # valid & test의 경우 65
        
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
        if self.test:
            return bin_data, int_data
        
        if self.classify:
            label_class = self.label_class[idx]
            if not self.complete:
                return bin_data, int_data, label_class
        if self.complete:
            label_compl = self.label_compl[idx]
            if not self.classify:
                return bin_data, int_data, label_compl
            else:
                return bin_data, int_data, label_class, label_compl