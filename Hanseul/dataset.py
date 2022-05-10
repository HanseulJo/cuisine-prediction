import h5py
from torch.utils.data import Dataset


class RecipeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with h5py.File(data_dir, 'r') as data_file:
            self.features_boolean = data_file['features_boolean'][:]  # recipes in boolean, size (recipes, ingreds)
            if 'labels_one_hot' in data_file.keys():
                self.labels_one_hot = data_file['labels_one_hot'][:]  # cuisines (or, answers for completion) in boolean, size (recipes, cuisines)
            if 'labels_int_enc' in data_file.keys():
                self.labels_int_enc = data_file['labels_int_enc'][:]  # cuisines in integers, size (recipes,)
            if 'labels_id' in data_file.keys():
                self.labels_id = data_file['labels_id'][:]  # answers for completion in integers starting from 0, size (recipes,)
        
    def __len__(self):
        return len(self.features_boolean)

    def __getitem__(self, idx):
        feature_boolean = self.features_boolean[idx]
        if hasattr(self, 'labels_one_hot'):
            label_one_hot = self.labels_one_hot[idx]
            if hasattr(self, 'labels_int_enc'):  # classification (train, valid_clf)
                label_int_enc = self.labels_int_enc[idx]
                return feature_boolean, label_one_hot, int(label_int_enc)
            if hasattr(self, 'labels_id'):       # completion (valid_cpl)
                label_id = self.labels_id[idx]
                return feature_boolean, label_one_hot, int(label_id)
        else:
            return feature_boolean # test_clf, test_cpl