import json
import os


def load_boknilev():
    train_path = os.path.dirname(__file__) + '/data/pp-data-english/train.json'
    test_path = os.path.dirname(__file__) + '/data/pp-data-english/test.json'
    with open(train_path, 'r') as f:
        train_samples = json.load(f)
    with open(test_path, 'r') as f:
        test_samples = json.load(f)
    return train_samples, test_samples
