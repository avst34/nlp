from collections import OrderedDict

from utils import f1_score
import random

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class PPAttEvaluator:

    ALL_CLASSES = '___ALL_CLASSES__'
    ALL_CLASSES_STRICT = '___ALL_CLASSES_STRICT__'

    def __init__(self, predictor=None):
        self.predictor = predictor

    def print_prediction(self, sample, predicted_y):
        try:
            print("Sample:")
            print("------")
            sample.pprint()
            print("Prediction:", end=' ')
            predicted_y.pprint()
        except UnicodeEncodeError:
            pass

    def update_counts(self, counts, klass, predicted, actual, strict=True):
        counts[klass] = counts.get(klass, {
            'p_none_a_none': 0,
            'p_none_a_value': 0,
            'p_value_a_none': 0,
            'p_value_a_value_eq': 0,
            'p_value_a_value_neq': 0,
            'total': 0
        })

        if strict:
            p, a = [predicted], [actual]
        else:
            p, a = [predicted, actual]

        if any([p, a]):
            if p is None:
                p = [None] * len(a)
            elif a is None:
                a = [None] * len(p)
        else:
            p, a = [[None], [None]]

        def isNone(x):
            return x is None or not any(x)

        for predicted, actual in zip(p, a):
            c = 1 / len(p)
            if isNone(predicted) and isNone(actual):
                counts[klass]['p_none_a_none'] += c
            else:
                counts[klass]['total'] += c
                if isNone(predicted) and not isNone(actual):
                    counts[klass]['p_none_a_value'] += c
                elif not isNone(predicted) and isNone(actual):
                    counts[klass]['p_value_a_none'] += c
                elif predicted == actual:
                    counts[klass]['p_value_a_value_eq'] += c
                else:
                    counts[klass]['p_value_a_value_neq'] += c
        return

    def evaluate(self, samples, examples_to_show=3, predictor=None):
        n_correct = 0
        samples = list(samples)
        random.shuffle(samples)
        for ind, s in enumerate(samples):
            predicted_y = predictor.predict(s.x)
            if ind < examples_to_show:
                self.print_prediction(s, predicted_y)
            if predicted_y.correct_head_cand == s.y.correct_head_cand:
                n_correct += 1
        acc = n_correct / len(samples)
        print("Accuracy: %1.2f" % acc)
        return {
            'acc': acc
        }

