import numpy as np
import pickle
import h5py
import os.path

num_ingredients = 6714

def embedding(data_path="../", save_path="./container", rm_same=False):
    # Get cuisine -> id dict
    with open(os.path.join(save_path, 'cuisine_id_dict.pickle'), 'rb') as fr:
        cuisine_dict = pickle.load(fr)

    if rm_same:
        with open(os.path.join(save_path, 'same_ingred_dict.pickle'), 'rb') as fr:
            ingred_dict = pickle.load(fr)

        with open(os.path.join(save_path, 'same_ingred_lst.pickle'), 'rb') as fr:
            ingred_lst = pickle.load(fr)

    # Train file embedding
    target_file_name = os.path.join(data_path, "train.csv")
    t_file = open(target_file_name, 'r')

    features = []
    labels = []

    lines = t_file.readlines()
    for line in lines:
        feature = np.zeros(num_ingredients)
        temp_lst = line.strip().split(",")
        for ingred_id in temp_lst[:-1]:
            if rm_same:
                ingred_id = ingred_dict[ingred_lst[int(ingred_id) - 1]][0]
            feature[int(ingred_id)-1] = 1
        features.append(feature)
        labels.append(cuisine_dict[temp_lst[-1]])

    t_file.close()

    one_hot_labels = []
    for label in labels:
        one_hot_label = np.zeros(20)
        one_hot_label[label] = 1
        one_hot_labels.append(one_hot_label)

    if not rm_same:
        h5f = h5py.File(os.path.join(save_path, "train"), 'w')
    else:
        h5f = h5py.File(os.path.join(save_path, "train_rm_same"), 'w')
    h5f.create_dataset('features_boolean', data=features, compression="gzip")
    h5f.create_dataset('labels_int_enc', data=labels, compression="gzip")
    h5f.create_dataset('labels_one_hot', data=one_hot_labels, compression="gzip")

    h5f.close()

    # Valid_cpl file embedding
    if not rm_same:
        h5f = h5py.File(os.path.join(save_path, "valid_cpl"), 'w')
    else:
        h5f = h5py.File(os.path.join(save_path, "valid_cpl_rm_same"), 'w')

    target_file_name = os.path.join(data_path, "validation_completion_question.csv")
    t_file = open(target_file_name, 'r')

    features = []

    lines = t_file.readlines()
    for line in lines:
        feature = np.zeros(num_ingredients)
        temp_lst = line.strip().split(",")
        for ingred_id in temp_lst:
            if rm_same:
                ingred_id = ingred_dict[ingred_lst[int(ingred_id) - 1]][0]
            feature[int(ingred_id) - 1] = 1
        features.append(feature)

    h5f.create_dataset('features_boolean', data=features, compression="gzip")
    t_file.close()

    target_file_name = os.path.join(data_path, "validation_completion_answer.csv")
    t_file = open(target_file_name, 'r')

    labels = []
    features_temp = []
    lines = t_file.readlines()
    for i, line in enumerate(lines):
        feature = np.zeros(num_ingredients)
        temp_lst = line.strip().split(",")
        for ingred_id in temp_lst:
            if rm_same:
                ingred_id = ingred_dict[ingred_lst[int(ingred_id) - 1]][0]
            if features[i][int(ingred_id) - 1] == 0:
                labels.append(ingred_id)
                feature[int(ingred_id) - 1] = 1
                break
        features_temp.append(feature)

    h5f.create_dataset('labels_one_hot', data=features_temp, compression="gzip")
    h5f.create_dataset('labels_id', data=labels, compression="gzip")

    t_file.close()

    h5f.close()

    # Valid_clf file embedding
    if not rm_same:
        h5f = h5py.File(os.path.join(save_path, "valid_clf"), 'w')
    else:
        h5f = h5py.File(os.path.join(save_path, "valid_clf_rm_same"), 'w')

    target_file_name = os.path.join(data_path, "validation_classification_question.csv")
    t_file = open(target_file_name, 'r')

    features = []

    lines = t_file.readlines()
    for line in lines:
        feature = np.zeros(num_ingredients)
    temp_lst = line.strip().split(",")
    for ingred in temp_lst:
        if rm_same:
            ingred = ingred_dict[ingred_lst[int(ingred) - 1]][0]
        feature[int(ingred) - 1] = 1
    features.append(feature)

    h5f.create_dataset('features_boolean', data=features, compression="gzip")

    t_file.close()

    target_file_name = os.path.join(data_path, "validation_classification_answer.csv")
    t_file = open(target_file_name, 'r')

    labels = []

    lines = t_file.readlines()
    for line in lines:
        cuisine = line.strip()
    labels.append(cuisine_dict[cuisine])

    one_hot_labels = []
    for label in labels:
        one_hot_label = np.zeros(20)
    one_hot_label[label] = 1
    one_hot_labels.append(one_hot_label)

    h5f.create_dataset('labels_int_enc', data=labels, compression="gzip")
    h5f.create_dataset('labels_one_hot', data=one_hot_labels, compression="gzip")

    t_file.close()

    h5f.close()

    # test_cpl file embedding
    if not rm_same:
        h5f = h5py.File(os.path.join(save_path, "test_cpl"), 'w')
    else:
        h5f = h5py.File(os.path.join(save_path, "test_cpl_rm_same"), 'w')

    target_file_name = os.path.join(data_path, "test_completion_question.csv")
    t_file = open(target_file_name, 'r')

    features = []

    lines = t_file.readlines()
    for line in lines:
        feature = np.zeros(num_ingredients)
        temp_lst = line.strip().split(",")
        for ingred in temp_lst:
            if rm_same:
                ingred = ingred_dict[ingred_lst[int(ingred) - 1]][0]
            feature[int(ingred) - 1] = 1
        features.append(feature)

    h5f.create_dataset('features_boolean', data=features, compression="gzip")
    t_file.close()
    h5f.close()

    # test_clf file embedding
    if not rm_same:
        h5f = h5py.File(os.path.join(save_path, "test_clf"), 'w')
    else:
        h5f = h5py.File(os.path.join(save_path, "test_clf_rm_same"), 'w')

    target_file_name = os.path.join(data_path, "test_classification_question.csv")
    t_file = open(target_file_name, 'r')

    features = []

    lines = t_file.readlines()
    for line in lines:
        feature = np.zeros(num_ingredients)
    temp_lst = line.strip().split(",")
    for ingred in temp_lst:
        if rm_same:
            ingred = ingred_dict[ingred_lst[int(ingred) - 1]][0]
        feature[int(ingred) - 1] = 1
    features.append(feature)

    h5f.create_dataset('features_boolean', data=features, compression="gzip")

    t_file.close()
    h5f.close()

if __name__ == "__main__":
    embedding()