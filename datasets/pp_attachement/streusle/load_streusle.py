import json
import os
from itertools import chain

def filter_non_pss_bearing(samples):
    for s in samples:
        s['pps'] = [pp for pp in s['pps'] if s['preprocessing']['gold_pss_role'][pp["ind"]] is not None]
    return samples


def load_streusle():
    train_path = os.path.dirname(__file__) + '/train.json'
    dev_path = os.path.dirname(__file__) + '/dev.json'
    test_path = os.path.dirname(__file__) + '/test.json'
    with open(train_path, 'r') as f:
        train_samples = json.load(f)
    with open(dev_path, 'r') as f:
        dev_samples = json.load(f)
    with open(test_path, 'r') as f:
        test_samples = json.load(f)
    train_samples, dev_samples, test_samples = [filter_non_pss_bearing(samples) for samples in (train_samples, dev_samples, test_samples)]
    return train_samples, dev_samples, test_samples

if __name__ == '__main__':
    tr, t, ts = load_streusle()
    print(len([p for s in tr for p in s['pps']]))

