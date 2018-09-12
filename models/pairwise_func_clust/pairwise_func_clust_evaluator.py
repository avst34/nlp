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
        }

        print("Accuracy: %2.2f" % (acc*100))
        print("Accuracy (True): %2.2f" % (scores['true_acc']*100))
        print("Accuracy (False): %2.2f" % (scores['false_acc']*100))
        print("Accuracy (Weighted): %2.2f" % ((scores['true_acc'] + scores['false_acc'])*100/2))
        return scores

