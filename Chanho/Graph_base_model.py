import numpy as np
import pickle
import h5py
import os.path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def topk_values(array, topk):
    temp_mat = np.zeros(len(array))
    topk = min(topk, np.count_nonzero(array > 0))
    temp_mat[np.argpartition(array, -topk)[-topk:]] = 1
    return temp_mat


def topk_per_row(matrix, topk):
    temp_mat = np.apply_along_axis(topk_values, 1, matrix, topk)
    return temp_mat


def graph_cpl(train_file, test_file, result_path, depth, weights, recipe_th, ingred_th, label_th, topk):

    recipe_threshold = recipe_th
    ingred_threshold = ingred_th
    label_threshold = label_th
    ingred_topk = topk

    w1, w2 = weights

    h5f_train = h5py.File(train_file, 'r')
    train_features = h5f_train['features_boolean'][:]
    train_labels = h5f_train['labels_one_hot'][:]
    h5f_train.close()

    h5f_valid = h5py.File(test_file, 'r')
    valid_features = h5f_valid['features_boolean'][:]
    h5f_valid.close()

    recipe_num, ingred_num = train_features.shape
    _, label_num = train_labels.shape

    adj_matrix = np.hstack([np.zeros((recipe_num, recipe_num)), w1*train_features, w2*train_labels])
    temp_matrix = np.hstack([w1*train_features.T, np.zeros((ingred_num, ingred_num)), np.zeros((ingred_num, label_num))])
    adj_matrix = np.vstack([adj_matrix, temp_matrix])
    temp_matrix = np.hstack([w2*train_labels.T, np.zeros((label_num, ingred_num)), np.zeros((label_num, label_num))])
    adj_matrix = np.vstack([adj_matrix, temp_matrix])
    adj_matrix = csr_matrix(adj_matrix)

    query_num, _ = valid_features.shape
    score_matrix = csr_matrix(np.hstack([np.zeros((query_num, recipe_num)), valid_features, np.zeros((query_num, label_num))]))
    src_matrix = score_matrix.copy()

    score_matrix = normalize(score_matrix, axis=1, norm='l1')
    exist_idx = score_matrix > 0

    for k in range(depth):
        # start node의 누적 점수 * 인접행렬 -> next node에 더해질 값
        src_matrix = np.multiply(score_matrix.toarray(), src_matrix.toarray()) @ adj_matrix
        score_matrix = csr_matrix(score_matrix + src_matrix)
        score_matrix = normalize(score_matrix, axis=1, norm='l1')
        # next_node의 score matrix
        src_matrix = score_matrix.toarray() * (src_matrix>0)
        src_recipe_matrix = topk_per_row(src_matrix[:, :recipe_num], recipe_threshold)
        src_ingred_matrix = topk_per_row(src_matrix[:, recipe_num:recipe_num+ingred_num], ingred_threshold)
        src_label_matrix = topk_per_row(src_matrix[:, recipe_num+ingred_num:], label_threshold)
        src_matrix = csr_matrix(np.hstack([src_recipe_matrix, src_ingred_matrix, src_label_matrix]))

        score_matrix_ = score_matrix.copy()
        score_matrix_[exist_idx] = 0

        ingred_recs = {}
        ingred_rec_idx_lst = np.argpartition(score_matrix_[:, recipe_num:recipe_num+ingred_num].toarray(), -ingred_topk)[:,-ingred_topk:]
        top_recommends = np.argmax(score_matrix_[:, recipe_num:recipe_num+ingred_num].toarray(), axis=1).flatten()

        for query, ingred_rec_idx in enumerate(ingred_rec_idx_lst):
            ingred_recs[query] = []

            for idx in ingred_rec_idx:
                ingred_recs[query].append((idx, score_matrix_[query, recipe_num + idx]))

            ingred_recs[query].sort(key=lambda x : x[1], reverse=True)

        with open(os.path.join(result_path, "Graph_rec_cpl_{}_{}_depth_{}.pickle".format(w1,w2,k)), 'wb') as f:
            pickle.dump(ingred_recs, f)

            
def graph_clf(train_file, test_file, result_path, depth, weights, recipe_th, ingred_th, label_th, topk):

    recipe_threshold = recipe_th
    ingred_threshold = ingred_th
    label_threshold = label_th
    label_topk = topk

    w1, w2 = weights

    h5f_train = h5py.File(train_file, 'r')
    train_features = h5f_train['features_boolean'][:]
    train_labels = h5f_train['labels_one_hot'][:]
    h5f_train.close()

    h5f_valid = h5py.File(test_file, 'r')
    valid_features = h5f_valid['features_boolean'][:]
    h5f_valid.close()

    recipe_num, ingred_num = train_features.shape
    _, label_num = train_labels.shape

    adj_matrix = np.hstack([np.zeros((recipe_num, recipe_num)), w1*train_features, w2*train_labels])
    temp_matrix = np.hstack([w1*train_features.T, np.zeros((ingred_num, ingred_num)), np.zeros((ingred_num, label_num))])
    adj_matrix = np.vstack([adj_matrix, temp_matrix])
    temp_matrix = np.hstack([w2*train_labels.T, np.zeros((label_num, ingred_num)), np.zeros((label_num, label_num))])
    adj_matrix = np.vstack([adj_matrix, temp_matrix])
    adj_matrix = csr_matrix(adj_matrix)
    
    query_num, _ = valid_features.shape
    score_matrix = csr_matrix(np.hstack([np.zeros((query_num, recipe_num)), valid_features, np.zeros((query_num, label_num))]))
    src_matrix = score_matrix.copy()
    
    score_matrix = normalize(score_matrix, axis=1, norm='l1')
    exist_idx = score_matrix > 0

    for k in range(depth):
        # start node의 누적 점수 * 인접행렬 -> next node에 더해질 값
        src_matrix = np.multiply(score_matrix.toarray(), src_matrix.toarray()) @ adj_matrix
        score_matrix = csr_matrix(score_matrix + src_matrix)
        score_matrix = normalize(score_matrix, axis=1, norm='l1')
        # next_node의 score matrix
        src_matrix = score_matrix.toarray() * (src_matrix>0)
        src_recipe_matrix = topk_per_row(src_matrix[:, :recipe_num], recipe_threshold)
        src_ingred_matrix = topk_per_row(src_matrix[:, recipe_num:recipe_num+ingred_num], ingred_threshold)
        src_label_matrix = topk_per_row(src_matrix[:, recipe_num+ingred_num:], label_threshold)
        src_matrix = csr_matrix(np.hstack([src_recipe_matrix, src_ingred_matrix, src_label_matrix]))

        score_matrix_ = score_matrix.copy()
        score_matrix_[exist_idx] = 0

        label_recs = {}
        label_rec_idx_lst = np.argpartition(score_matrix_[:, recipe_num+ingred_num:].toarray(), -label_topk)[:,-label_topk:]
        top_recommends = label_rec_idx_lst[:,0].flatten()

        for query, label_rec_idx in enumerate(label_rec_idx_lst):
            label_recs[query] = []

            for idx in label_rec_idx:
                label_recs[query].append((idx, score_matrix_[query, recipe_num + ingred_num + idx]))

            label_recs[query].sort(key=lambda x : x[1], reverse=True)

        with open(os.path.join(result_path, "Graph_rec_clf_{}_{}_depth_{}.pickle".format(w1,w2,k)), 'wb') as f:
            pickle.dump(label_recs, f)