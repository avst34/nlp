from models.pss_func.calc_ss_per_pp_dist import calc_ss_per_pp_dist
from models.pss_func.most_frequent_from_dists import most_frequent_from_dists


def eval_func_pred(prediction, records):
    func_dist = calc_ss_per_pp_dist(records, pss='func')
    actual = most_frequent_from_dists(func_dist)
    actual = {k: v for k, v in actual.items() if k in prediction}
    assert(sorted(prediction.keys()) == sorted(actual.keys()))
    acc = len(set(prediction.items()) & set(actual.items())) / len(actual)
    return {
        'acc': acc,
        'actual': actual
    }

def eval_type_level_func_pred_token_level(prediction, tags):
    total = len([t for t in tags if t.prep in prediction])
    correct = len([t for t in tags if t.prep in prediction and t.ss_func == prediction[t.prep]])
    return {
        'acc': correct / total
    }