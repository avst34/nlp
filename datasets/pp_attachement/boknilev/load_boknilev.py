import json
import os
from itertools import chain


def load_boknilev():
    train_path = os.path.dirname(__file__) + '/data/pp-data-english/train.json'
    dev_path = os.path.dirname(__file__) + '/data/pp-data-english/dev.json'
    test_path = os.path.dirname(__file__) + '/data/pp-data-english/test.json'
    with open(train_path, 'r') as f:
        train_samples = json.load(f)
    with open(dev_path, 'r') as f:
        dev_samples = json.load(f)
    with open(test_path, 'r') as f:
        test_samples = json.load(f)
    psses = load_boknilev_pss()
    for s in chain(train_samples, dev_samples, test_samples):
        for pp in s['pps']:
            # print(s['tokens'], psses[s['sent_id']], pp['ind'])
            pp['pss_role'], pp['pss_func'] = psses[s['sent_id']][str(pp['ind'])]
    return train_samples, dev_samples, test_samples

def dump_boknilev_pss(predictions):
    # validate predictions match dataset
    samples = sum(load_boknilev(), [])
    for sample in samples:
        for pp in sample['pps']:
            assert predictions[sample['sent_id']][pp['ind']]

    pss_path = os.path.dirname(__file__) + '/data/pp-data-english/pss_predictions.json'
    with open(pss_path, 'w') as f:
        json.dump(predictions, f)

def load_boknilev_pss():
    pss_path = os.path.dirname(__file__) + '/data/pp-data-english/pss_predictions.json'
    with open(pss_path, 'r') as f:
        return json.load(f)

