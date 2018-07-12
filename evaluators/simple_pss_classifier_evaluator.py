import random
from pprint import pprint

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class SimplePSSClassifierEvaluator:

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

        counts = {
            'supersense_role': 0,
            'supersense_func': 0
        }
        samples = list(samples)
        random.shuffle(samples)
        for ind, s in enumerate(samples):
            predicted_y = predictor.predict(s.x)
            if ind < examples_to_show:
                self.print_prediction(s, predicted_y)
            for f in ['supersense_role', 'supersense_func']:
                if getattr(predicted_y, f) == getattr(s.y, f):
                    counts[f] += 1

        acc = {
            'supersense_role': counts['supersense_role'] / len(samples),
            'supersense_func': counts['supersense_func'] / len(samples)
        }
        pprint(acc)
        return acc

