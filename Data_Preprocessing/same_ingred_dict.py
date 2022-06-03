import numpy as np
import pickle
import os.path
from collections import defaultdict


def get_same_ingred(data_path='../', save_path='./container', save=True):
    ingred_dict = defaultdict(list)
    ingred_lst = []
    # ingred_dict: ingred name -> ids with same name
    with open(os.path.join(data_path, 'node_ingredient.txt'), 'rb') as fr:
        for i, line in enumerate(fr):
            ingred = line.strip()
            ingred_dict[ingred].append(i)
            ingred_lst.append(ingred)

    if save:
        with open(os.path.join(save_path, 'same_ingred_dict.pickle'), 'wb') as fw:
            pickle.dump(ingred_dict, fw)

        with open(os.path.join(save_path, 'same_ingred_lst.pickle'), 'wb') as fw:
            pickle.dump(ingred_lst, fw)

    return ingred_dict, ingred_lst

if __name__ == "__main__":
    get_same_ingred()