from collections import OrderedDict

from utils import f1_score
import random

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class PSSClasifierEvaluator:

    ALL_CLASSES = '___ALL_CLASSES__'
    ALL_CLASSES_STRICT = '___ALL_CLASSES_STRICT__'

    def __init__(self, predictor=None):
        self.predictor = predictor

    def print_prediction(self, sample, predicted_ys):
        print("Sample:")
        print("------")
        for x, y_true, y_predicted in zip(sample.xs, sample.ys, predicted_ys):
            try:
                print(''.join(['{:<30}'.format("[%s] %s" % (f, x[f])) for f in sorted(x.keys())]))
                if any(y_true or []) or any(y_predicted or []):
                    print('^ [%s]  ACTUAL: %10s  PREDICTED: %10s' % ('X' if y_true != y_predicted else 'V', y_true, y_predicted) \
                              if y_true else ' ')
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
        ALL_CLASSES = PSSClasifierEvaluator.ALL_CLASSES
        ALL_CLASSES_STRICT = PSSClasifierEvaluator.ALL_CLASSES_STRICT
        predictor = predictor or self.predictor
        counts = {
            ALL_CLASSES: {
                'p_none_a_none': 0,
                'p_none_a_value': 0,
                'p_value_a_none': 0,
                'p_value_a_value_eq': 0,
                'p_value_a_value_neq': 0,
                'total': 0
            },
            ALL_CLASSES_STRICT: {
                'p_none_a_none': 0,
                'p_none_a_value': 0,
                'p_value_a_none': 0,
                'p_value_a_value_eq': 0,
                'p_value_a_value_neq': 0,
                'total': 0
            }
        }

        class_scores = {}

        for sample_ind, sample in enumerate(samples):
            predicted_ys = predictor.predict(sample.xs, sample.mask)
            if sample_ind < examples_to_show:
                self.print_prediction(sample, predicted_ys)
            for p, a in zip(predicted_ys, sample.ys):
                if not p:
                    p = tuple([None] * len(a))
                self.update_counts(counts, a, p, a)
                self.update_counts(counts, ALL_CLASSES, p, a, strict=False)
                self.update_counts(counts, ALL_CLASSES_STRICT, p, a)
                if a is not None and len(a) > 1:
                    for ind, klass in enumerate(a):
                        cklass = tuple([a[i] if i == ind else '*' for i in range(len(a))])
                        self.update_counts(counts, cklass, p[ind], klass)
                        cklass = tuple(['-- All --' if i == ind else '*' for i in range(len(a))])
                        self.update_counts(counts, cklass, p[ind], klass)

        for klass, class_counts in counts.items():
            if class_counts['total'] != 0:
                precision = (class_counts['p_value_a_value_eq'] / ((class_counts['p_value_a_value_eq'] + class_counts['p_value_a_value_neq'] + class_counts['p_value_a_none']) or 0.001))
                recall = (class_counts['p_value_a_value_eq'] / ((class_counts['p_value_a_value_eq'] + class_counts['p_value_a_value_neq'] + class_counts['p_none_a_value']) or 0.001))
                if precision + recall:
                    f1 = f1_score(precision, recall)
                else:
                    f1 = None
                class_scores[klass] = OrderedDict([
                    ('precision', to_precentage(precision)),
                    ('recall', to_precentage(recall)),
                    ('f1', to_precentage(f1)),
                    ('total', class_counts['total']),
                    ('correct/total', '%d / %d' % (class_counts['p_value_a_value_eq'], class_counts['total']))
                ])

        def report(klass, display_name):
            print(' - %s precision: %2.2f' % (display_name, class_scores[klass]['precision']))
            print(' - %s recall:    %2.2f' % (display_name, class_scores[klass]['recall']))
            if class_scores[klass]['f1'] is not None:
                print(' - %s f1 score:  %2.2f' % (display_name, class_scores[klass]['f1']))

        total_counts = counts[ALL_CLASSES]
        total_precision = class_scores[ALL_CLASSES]['precision']
        total_recall = class_scores[ALL_CLASSES]['recall']
        total_f1 = class_scores[ALL_CLASSES]['f1']

        print('Evaluation on %d samples (%d predictions):' % (len(samples),
                                                              total_counts['p_value_a_value_eq'] +
                                                              total_counts['p_value_a_value_neq'] +
                                                              total_counts['p_none_a_value'])
              )
        report(ALL_CLASSES, 'All Classes')
        report(ALL_CLASSES_STRICT, 'All Classes (strict)')

        return {
            'precision': total_precision,
            'recall': total_recall,
            'f1': total_f1,
            'total': counts[ALL_CLASSES]['total'],
            'class_scores': class_scores
        }