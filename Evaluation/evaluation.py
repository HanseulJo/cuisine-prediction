from sklearn.metrics import f1_score, accuracy_score

# recs = {query_id: [lst of (recommend item, score)]}
# answer = {query_id: correct item}


def get_f1_macro(recs, answer):
    return f1_score(answer, recs, average='macro')


def get_f1_micro(recs, answer):
    return f1_score(answer, recs, average='micro')


def get_acc(recs, answer):
    return accuracy_score(answer, recs)


def get_MAP(recs, answer):
    MAP = 0
    for query, ans in list(answer.items()):
        for i, rec in enumerate(recs[query]):
            item_idx, _ = rec
            if ans == item_idx:
                MAP += 1/(i+1)
    MAP = MAP / len(recs)
    return MAP


def get_recall_n(recs, answer, n):
    recall = 0
    for query, ans in list(answer.items()):
        for i, rec in enumerate(recs[query]):
            if i >= n:
                break
            item_idx, _ = rec
            if ans == item_idx:
                recall += 1
    recall = recall/len(recs)
    return recall

def get_recall_rank(recs, answer, n):
    recall = 0
    recall_rank = 0
    for query, ans in list(answer.items()):
        for i, rec in enumerate(recs[query]):
            if i >= n:
                break
            item_idx, _ = rec
            if ans == item_idx:
                recall += 1
                recall_rank += i + 1
    if recall != 0:
        recall_rank_avg = recall_rank / recall
    else: recall_rank_avg = 0
    return recall_rank_avg



def get_metric(recs, answer, mode='all', n=10):
    top_rec = [items[0][0] for query, items in sorted(list(recs.items()))]
    answer_ = [item for query, item in sorted(list(answer.items()))]
    metric = {}
    metric['macro'] = get_f1_macro(top_rec, answer_)
    metric['micro'] = get_f1_micro(top_rec, answer_)
    metric['accuracy'] = get_acc(top_rec, answer_)
    metric['map'] = get_MAP(recs, answer)
    metric['recall'] = get_recall_n(recs, answer, n)
    metric['recall_rank'] = get_recall_rank(recs, answer, n)
    return metric
