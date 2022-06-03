import os
import csv
import pickle
import numpy as np
import h5py

def read_train_data(path):
    """ Read train.csv and Return lists of data[int] / label[str]. """
    data = []
    with open(path, 'r') as f:
        for line in csv.reader(f):
            recipe = sorted(set(map(int, line[:-1])))  # a sorted list of recipe (integer) # 0417: 중복되는 재료 삭제 (3개 recipe)
            data.append(recipe)
    return data

def read_ingredient_names(ingredient_path):
    ingredients_names = []
    with open(ingredient_path, 'r', encoding='utf-8') as f:
        for line in csv.reader(f):
            ingredients_names.append(line[0])
    return ingredients_names

def data_to_binary_array(data, dim):
    """ convert data(list of lists) into a 2D binary array. (for dataset, row = recipe) """
    """ dim (int) : dimension of each row (of 'enc') that must be. """
    enc = np.zeros((len(data), dim), dtype=int) 
    for i in range(len(data)):
        recipe = data[i]
        enc[i][recipe] = 1
    return enc

def make_train_cpl(data_path, save_path):

    data_train_class = read_train_data(os.path.join(data_path, "train.csv"))  # classification-only dataset
    data_train_compl = []
    labels_train_compl = []
    for recipe in data_train_class:
        if len(recipe) > 1:
            for i in range(len(recipe)):
                data_train_compl.append(recipe[:i]+recipe[i+1:])
                labels_train_compl.append(recipe[i])
    
    ingredient_names = read_ingredient_names(os.path.join(data_path, "node_ingredient.txt"))
    num_ingredients = len(ingredient_names)  # 6714
    
    bin_data_train_compl = data_to_binary_array(data_train_compl, num_ingredients)
    int_labels_train_compl = np.array(labels_train_compl)
    bin_labels_train_compl = np.eye(num_ingredients)[labels_train_compl]

    with h5py.File(save_path + 'train_cpl', 'w') as h5f:
        h5f.create_dataset('features_boolean', data=bin_data_train_compl, compression="gzip")
        h5f.create_dataset('labels_one_hot', data=bin_labels_train_compl, compression="gzip")
        h5f.create_dataset('labels_id', data=int_labels_train_compl, compression="gzip")

if __name__ == '__main__':
    data_path = '../'
    save_path = '../Container/'
    make_train_cpl(data_path, save_path)