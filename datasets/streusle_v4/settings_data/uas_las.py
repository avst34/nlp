import json
import os
from collections import Counter


def compare_sets_acc(gold, pred):
    c = Counter()
    assert len(gold)==len(pred)
    c['N'] = len(gold)
    c['correct'] = len(gold & pred)
    assert len(gold - pred)==len(pred - gold)
    c['incorrect'] = len(gold - pred)
    c['Acc'] = c['correct'] / c['N']
    return c


def calc_uas_las(sysf_path, goldf_path):
    with open(sysf_path) as sysf:
        sys = json.load(sysf)
    with open(goldf_path) as goldf:
        gold = json.load(goldf)

    sset = {(s['sent_id'], t['#'], t['head'], t['deprel']) for s in sys for t in s['toks']}
    gset = {(s['sent_id'], t['#'], t['head'], t['deprel']) for s in gold for t in s['toks']}

    c_las = compare_sets_acc(gset, sset)
    c_uas = compare_sets_acc({(id, tid, head) for (id, tid, head, dep) in gset}, {(id, tid, head) for (id, tid, head, dep) in sset})
    return {
        'las': c_las,
        'uas': c_uas
    }

if __name__ == '__main__':
    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'
    for s in ['train', 'dev', 'test']:
        ul = calc_uas_las(STREUSLE_BASE + '/' + s + '/streusle.ud_' + s + '.goldid.autosyn.json',
                     STREUSLE_BASE + '/' + s + '/streusle.ud_' + s + '.goldid.goldsyn.json')
        with open(STREUSLE_BASE + '/' + s + '/streusle.ud_' + s + '.uas_las.json', 'w') as f:
            json.dump(ul, f, indent=2)

