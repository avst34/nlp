import csv
from collections import OrderedDict

from utils import f1_score
import random

to_precentage = lambda s: int(s * 10000) / 100 if s is not None else None

class PSSDebugEvaluator:

    def __init__(self, predictor=None):
        self.predictor = predictor

    def print_prediction(self, sample, predicted_y):
        try:
            print("Sample:")
            print("------")
            print("Correct:   ", [sample.x.pp.pss_role, sample.x.pp.pss_func])
            print("Prediction:", predicted_y)
        except UnicodeEncodeError:
            pass

    def evaluate(self, samples, examples_to_show=3, predictor=None, predictions_csv_path=None):
        predictor = predictor or self.predictor

        csv_rows = []
        csv_rows.append(['sent_id', 'sent', 'pp', 'pp_ind', 'pss_role', 'pss_func', 'head cands', 'gold', 'predicted', 'predictions', 'is_correct'])

        n_correct = 0
        samples = list(samples)
        random.shuffle(samples)
        for ind, s in enumerate(samples):
            predicted_y = predictor.debug_predict_pss(s.x)
            if ind < examples_to_show:
                self.print_prediction(s, predicted_y)
            if predicted_y == [s.x.pp.pss_role, s.x.pp.pss_func]:
                n_correct += 1
            csv_rows.append([
                s.x.sent_id,
                ' '.join('%s_%d' % (t, ind) for ind, t in enumerate(s.x.tokens)),
                s.x.pp.word,
                s.x.pp.ind,
                s.x.pp.pss_role,
                s.x.pp.pss_func,
                ' '.join(['%s_%d' % (hc.word, hc.ind) for hc in s.x.head_cands]),
                '%s_%d' % (s.y.correct_head_cand.word, s.y.correct_head_cand.ind),
                '%s' % (repr(predicted_y)),
                ' '.join(['%s_%d_%2.2f' % (hc.word, hc.ind, score) for score, hc in s.y.scored_heads]),
                predicted_y == [s.x.pp.pss_role, s.x.pp.pss_func]
            ])

        acc = n_correct / len(samples)
        print("Accuracy: %2.2f" % (acc*100))

        if predictions_csv_path:
            with open(predictions_csv_path, 'w') as out_f:
                writer = csv.writer(out_f)
                for row in csv_rows:
                    writer.writerow(row)

        return {
            'acc': acc
        }

