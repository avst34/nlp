from utils import f1_score
import random

class ClassifierEvaluator:

    def __init__(self, predictor=None):
        self.predictor = predictor

    def print_prediction(self, sample, predicted_ys):
        for x, y_true, y_predicted in zip(sample.xs, sample.ys, predicted_ys):
            print(
                [x[f] for f in sorted(x.keys())],
                'correct: %s, predicted: %s    --- %s' % (y_true, y_predicted, 'X' if y_true != y_predicted else 'V') \
                    if y_true else ''
            )

    def evaluate(self, samples, examples_to_show=3, predictor=None):
        predictor = predictor or self.predictor
        p_none_a_none = 0
        p_none_a_value = 0
        p_value_a_none = 0
        p_value_a_value_eq = 0
        p_value_a_value_neq = 0
        total = 0
        samples = list(samples)
        random.shuffle(samples)
        for ind, sample in enumerate(samples):
            predicted_ys = predictor.predict(sample.xs, [True if y else False for y in sample.ys])
            if ind < examples_to_show:
                self.print_prediction(sample, predicted_ys)
            for p, a in zip(predicted_ys, sample.ys):
                if p is None and a is None:
                    p_none_a_none += 1
                elif p is None and a is not None:
                    p_none_a_value += 1
                elif p is not None and a is None:
                    p_value_a_none += 1
                elif p == a:
                    p_value_a_value_eq += 1
                else:
                    p_value_a_value_neq += 1
                total += 1

        precision = (p_value_a_value_eq / (p_value_a_value_eq + p_value_a_value_neq + p_value_a_none))
        recall = (p_value_a_value_eq / (p_value_a_value_eq + p_value_a_value_neq + p_none_a_value))
        f1 = f1_score(precision, recall)
        print('Evaluation on %d samples (%d predictions):' % (len(samples), p_value_a_value_eq + p_value_a_value_neq + p_none_a_value))
        print(' - precision: %1.4f' % precision)
        print(' - recall:    %1.4f' % recall)
        print(' - f1 score:  %1.4f' % f1)