import numpy as np
from scipy.special import softmax
from collections import defaultdict
from sklearn.preprocessing import normalize


def ensemble(rec_list, weight=None, n=10):
    new_recs = {}
    query_set = set()
    if weight == None:
        weight = [1 for i in range(len(rec_list))]
    for recs in rec_list:
        query_set = query_set.union(set(recs.keys()))
    for query in query_set:
        en_rec = defaultdict(int)
        for recs, w in zip(rec_list, weight):
            try:
                rec = recs[query]
            except:
                continue
            rec_items, rec_scores = [rec_ for rec_, score in rec], [score for rec_, score in rec]
            #rec_scores = softmax(rec_scores)
            rec_scores = normalize(np.array(rec_scores)[:,np.newaxis], axis=0).ravel()
            for i, rec_item in enumerate(rec_items):
                en_rec[rec_item] += rec_scores[i] * w
        topk = min(n, len(en_rec))
        rec_lst = sorted(list(en_rec.items()), key= lambda x: x[1], reverse=True)[:topk]
        new_recs[query] = rec_lst
    return new_recs
