import random

from collections import defaultdict

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class PairwiseFuncClustEvaluator:

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

    def evaluate(self, samples, examples_to_show=3, predictor=None):
        predictor = predictor or self.predictor

        class_total = defaultdict(lambda: 0)
        class_correct = defaultdict(lambda: 0)

        samples = list(samples)
        random.shuffle(samples)
        for ind, s in enumerate(samples):
            predicted_y = predictor.predict(s.x)
            if ind < examples_to_show:
                self.print_prediction(s, predicted_y)
            class_total[s.y.is_same_cluster] += 1
            if predicted_y.is_same_cluster == s.y.is_same_cluster:
                class_correct[s.y.is_same_cluster] += 1

        acc = sum(class_correct.values()) / len(samples)
        scores = {
            'acc': acc,
            'true_acc': class_correct[True] / class_total[True],
            'false_acc': class_correct[False] / class_total[False],
            'r': class_correct[True] / class_total[True],
            'p': class_correct[True] / (class_correct[True] + (class_total[False] - class_correct[False]))
        }
        scores['f1'] = 2 * scores['r'] * scores['p'] / (scores['r'] + scores['p'])

        print("P:  %2.2f" % (scores['p']*100))
        print("R:  %2.2f" % (scores['r']*100))
        print("F1: %2.2f" % (scores['f1']*100))
        return scores

