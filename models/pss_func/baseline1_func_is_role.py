import copy

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.calc_ss_per_pp_dist import calc_ss_per_pp_dist
from models.pss_func.eval_func_pred import eval_func_pred
from models.pss_func.most_frequent_from_dists import most_frequent_from_dists


def predict_func_by_func_is_role(role_dist):
    return most_frequent_from_dists(role_dist)

if __name__ == '__main__':
    records = StreusleLoader().load()
    role_dist = calc_ss_per_pp_dist(records)
    prediction = predict_func_by_func_is_role(role_dist)
    e = eval_func_pred(prediction, records)
    print('acc:', e['acc'])
    for prp in e['actual'].keys():
        print('%s: %s -> %s' % (prp, e['actual'][prp], prediction[prp]))