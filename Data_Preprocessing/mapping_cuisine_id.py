import numpy as np
import pickle
import os.path


def make_dict(data_path="../", save_path='./container', save=True):
    t_file = open(os.path.join(data_path, "train.csv"), 'r')

    cuisines = set()

    # Add cuisine to set
    lines = t_file.readlines()
    for line in lines:
        temp_lst = line.strip().split(",")
        cuisines.add(temp_lst[-1])

    # Sort cuisine in the set
    cuisines = list(cuisines)
    cuisines.sort(reverse=True)

    # Give id to each cuisine and make dict
    cuisine_id = 0
    cuisine_dict = {}
    cuisine_dict_inv = {}
    while len(cuisines) > 0:
        cuisine = cuisines.pop()
        cuisine_dict[cuisine] = cuisine_id
        cuisine_dict_inv[cuisine_id] = cuisine
        cuisine_id += 1

    if save:
        # cuisine -> id dict
        with open(os.path.join(save_path, 'cuisine_id_dict.pickle'), 'wb') as fw:
            pickle.dump(cuisine_dict, fw)

        # id -> cuisine dict
        with open(os.path.join(save_path, 'id_cuisine_dict.pickle'), 'wb') as fw:
            pickle.dump(cuisine_dict_inv, fw)

    return cuisine_dict, cuisine_dict_inv


if __name__ == "__main__":
    make_dict()