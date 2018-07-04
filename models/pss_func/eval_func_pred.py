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