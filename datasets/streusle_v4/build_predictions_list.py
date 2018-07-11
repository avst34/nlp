import json
from collections import namedtuple

from models.pss_func.instance_level_stats import print_report

Annotation = namedtuple('Annotation', ['sent_id', 'we_id', 'prep', 'role', 'func', 'role_dist', 'func_dist'])

def load_annotations(streusle_json_path):
    annotations = []
    with open(streusle_json_path) as f:
        records = json.load(f)

    is_pss = lambda pss: pss is not None and pss.startswith('p.')

    for record in records:
        toks = {t['#']: t['word'] for t in record['toks']}
        for we_id, we in list(record['swes'].items()) + list(record['wmwes'].items()):
            ss = we.get('ss')
            ss2 = we.get('ss2')
            if is_pss(ss) or is_pss(ss2) or (ss is None and ss2 is None and we['lexcat'] in ['P', 'PP', 'INF.P', 'POSS', 'PRON.POSS']):
                annotations.append(Annotation(record['sent_id'], we_id, ' '.join([toks[tid] for tid in we['toknums']]), ss, ss2, we.get('ss_dist'), we.get('ss2_dist')))

    return annotations

Prediction = namedtuple('Prediction', ['sent_id', 'we_id', 'prep', 'pred_role', 'pred_func', 'pred_role_dist', 'pred_func_dist', 'gold_role', 'gold_func'])

def match_annotations(gold, pred):
    d = {(a.sent_id, a.we_id, a.prep): a for a in pred}
    return [(g, d[(g.sent_id, g.we_id, g.prep)]) for g in gold]


def build_predictions_list(gold_json, pred_json):
    gold = load_annotations(gold_json)
    pred = load_annotations(pred_json)
    return [
        Prediction(g.sent_id, g.we_id, g.prep, p.role, p.func, p.role_dist, p.func_dist, g.role, g.func)
        for g, p in match_annotations(gold, pred)
    ]

def dump_predictions(preds, outpath):
    with open(outpath, 'w') as f:
        json.dump([x._asdict() for x in preds], f, indent=2)


if __name__ == '__main__':
    preds = build_predictions_list('/tmp/role_only/goldid.goldsyn/goldid.goldsyn.dev.gold.json','/tmp/role_only/goldid.goldsyn/goldid.goldsyn.dev.sys.goldid.json')
    dump_predictions(preds, '/tmp/preds.json')
    print_report(preds)
