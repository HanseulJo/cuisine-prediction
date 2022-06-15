import numpy as np
import pickle
import h5py
import os.path
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import pairwise_distances


def ALS_CF_cpl(train_file, test_file, result_path, data_type, k=2**6, topk=10):

    ingred_topk = topk

    h5f_train = h5py.File(train_file, 'r')
    train_features = h5f_train['features_boolean'][:]
    train_labels = h5f_train['labels_one_hot'][:]
    h5f_train.close()

    h5f_valid = h5py.File(test_file, 'r')
    valid_features = h5f_valid['features_boolean'][:]
    h5f_valid.close()

    features_csr = csr_matrix(train_features)
    features_csr_T = csr_matrix(features_csr.T)

    model = AlternatingLeastSquares(factors=k, regularization=0.01, calculate_training_loss=True)
    model.fit(features_csr)

    sim_matrix = 1 - pairwise_distances(model.item_factors, metric="cosine") - np.eye(len(model.item_factors))
    score_matrix = valid_features @ sim_matrix
    rec_indices = np.argpartition(score_matrix, -ingred_topk)[:,-ingred_topk:]

    ingred_recs = {}
    for query, ingred_rec_idx in enumerate(rec_indices):
        ingred_recs[query] = []
        for idx in ingred_rec_idx:
            ingred_recs[query].append((idx, score_matrix[query, idx]))
        ingred_recs[query].sort(key=lambda x : x[1], reverse=True)
    
    with open(os.path.join(result_path, "CF_rec_cpl_{}_dim_{}.pickle".format(data_type, k)), 'wb') as f:
        pickle.dump(ingred_recs, f)
        

def ALS_CF_clf(train_file, test_file, result_path, result_type, k=2**6, topk=10):

    cuisine_topk = topk

    h5f_train = h5py.File(train_file, 'r')
    train_features = h5f_train['features_boolean'][:]
    train_labels = h5f_train['labels_one_hot'][:]
    h5f_train.close()
    
    ingred_num = train_features.shape[1]
    cuisine_num = train_labels.shape[1]
    
    h5f_valid = h5py.File(test_file, 'r')
    valid_features = h5f_valid['features_boolean'][:]
    h5f_valid.close()

    features_csr = csr_matrix(np.hstack([train_features, train_labels]))
    features_csr_T = csr_matrix(features_csr.T)

    model = AlternatingLeastSquares(factors=k, regularization=0.01, calculate_training_loss=True)
    model.fit(features_csr)
    
    sim_matrix = 1 - pairwise_distances(model.item_factors, metric="cosine") - np.eye(len(model.item_factors))
    score_matrix = valid_features @ sim_matrix[:-cuisine_num,ingred_num:]
    rec_indices = np.argpartition(score_matrix, -cuisine_topk)[:,-cuisine_topk:]
    
    cuisine_recs = {}
    for query, cuisine_rec_idx in enumerate(rec_indices):
        cuisine_recs[query] = []
        for idx in cuisine_rec_idx:
            cuisine_recs[query].append((idx, score_matrix[query, idx]))
        cuisine_recs[query].sort(key=lambda x : x[1], reverse=True)
        
    with open(os.path.join(result_path, "CF_rec_clf_{}_dim_{}.pickle".format(result_type, k)), 'wb') as f:
        pickle.dump(cuisine_recs, f)