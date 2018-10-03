from collections import Counter

import itertools

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class GmmFuncClustEvaluator:

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

    def evaluate(self, samples, predictor=None):
        predictor = predictor or self.predictor

        pys = [predictor.predict(s.x) for s in samples]
        outs = []
        for (s1, p1), (s2, p2) in itertools.combinations(zip(samples, pys), 2):
            outs.append((s1.y.func == s2.y.func, p1.cluster == p2.cluster))
        counts = Counter(outs)

        scores = {
            'r': counts[(True, True)] / (counts[(True, True)] + counts[(True, False)]),
            'p': counts[(True, True)] / (counts[(True, True)] + counts[(False, True)])
        }
        scores['f1'] = 2 * scores['r'] * scores['p'] / (scores['r'] + scores['p'])

        print("Accuracy (r): %2.2f" % (scores['r'] * 100))
        print("Accuracy (p): %2.2f" % (scores['p'] * 100))
        print("Accuracy (f1): %2.2f" % (scores['f1'] * 100))
        return scores

