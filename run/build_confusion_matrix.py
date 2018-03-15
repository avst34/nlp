import json
from collections import defaultdict
from itertools import chain


def build_confusion_matrix(sysf_path, goldf_path):

    with open(sysf_path) as sysf:
        sys_sents = json.load(sysf)
    with open(goldf_path) as goldf:
        gold_sents = json.load(goldf)

    def format_pair(s1, s2):
        return str(s1) + ',' + str(s2)

    role_mat = defaultdict(lambda: defaultdict(lambda: 0))
    fxn_mat = defaultdict(lambda: defaultdict(lambda: 0))
    exact_mat = defaultdict(lambda: defaultdict(lambda: 0))

    for sys_sent, gold_sent in zip(sys_sents, gold_sents):
        assert sys_sent['sent_id'] == gold_sent['sent_id']
        sys_wes = list(sys_sent['swes'].values()) + list(sys_sent['smwes'].values())
        gold_wes = list(gold_sent['swes'].values()) + list(gold_sent['smwes'].values())

        for gold_we in gold_wes:
            for sys_we in sys_wes:
                if set(sys_we['toknums']) == set(gold_we['toknums']):
                    if not (gold_we['ss'] or "").startswith('p.'):
                        continue
                    gold_ss, gold_ss2 = gold_we['ss'], gold_we['ss2']
                    sys_ss, sys_ss2 = sys_we['ss'], sys_we['ss2']
                    role_mat[gold_ss][sys_ss] += 1
                    fxn_mat[gold_ss2][sys_ss2] += 1
                    exact_mat[format_pair(gold_ss, gold_ss2)][format_pair(sys_ss, sys_ss2)] += 1

    def normalize(mat):
        return {
            k: {
                k2: {
                    'p': mat[k][k2] / (sum(mat[k].values()) - mat[k][k]) if (sum(mat[k].values()) - mat[k][k]) else 0,
                    'n': mat[k][k2]
                } for k2 in dict(mat[k]) if k != k2
            } for k in dict(mat)
        }

    role_mat = normalize(role_mat)
    fxn_mat = normalize(fxn_mat)
    exact_mat = normalize(exact_mat)
    mats = {
        'role': role_mat,
        'fxn': fxn_mat,
        'exact': exact_mat
    }

    return mats

from pprint import pprint

pprint(build_confusion_matrix(
    r"C:\temp\best_results\nn\goldid.goldsyn\goldid.goldsyn.test.sys.goldid.json",
    r"C:\temp\best_results\nn\goldid.goldsyn\goldid.goldsyn.test.gold.json"
))