import random

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

        n_correct = 0
        samples = list(samples)
        random.shuffle(samples)
        for ind, s in enumerate(samples):
            predicted_y = predictor.predict(s.x)
            if ind < examples_to_show:
                self.print_prediction(s, predicted_y)
            if predicted_y.is_same_cluster == s.y.is_same_cluster:
                n_correct += 1

        acc = n_correct / len(samples)
        print("Accuracy: %2.2f" % (acc*100))

        return {
            'acc': acc
        }

