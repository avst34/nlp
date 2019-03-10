import numbers
import sys
import csv
import os
import json
import time
import subprocess
from itertools import chain

from statistics import mean, stdev

from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

from datasets.streusle_v4.streusle import StreusleLoader

STREUSLE_BASE = os.path.dirname(__file__) + '/../datasets/streusle_v4/release'

gold_records = StreusleLoader().load(STREUSLE_BASE + '/streusle.conllulex')
gold_record_by_id = {r.id: r for r in gold_records}

def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False

class StreusleEvaluator:

    def  __init__(self, predictor, psseval_script_path=STREUSLE_BASE + '/psseval.py'):
        self.psseval_script_path = psseval_script_path
        self.predictor = predictor

    def evaluate(self, streusle_records, output_tsv_path=None, ident='autoid', gold_fname_out=None, sys_fname_out=None, streusle_record_to_model_sample=streusle_record_to_lstm_model_sample, all_depths=False):
        assert ident in ['autoid', 'goldid']
        rand = str(int(time.time() * 1000))
        gold_fname = gold_fname_out or 'gold_' + rand + '.json'
        sys_fname = sys_fname_out or 'sys_' + rand + '.' + ident + '.json'

        if output_tsv_path:
            keep_output_file = True
        else:
            keep_output_file = False
            output_tsv_path = 'output_' + rand + '.json'

        try:
            with open(sys_fname, 'w') as sys_f:
                sys_data = []
                for streusle_record in streusle_records:
                    sample = streusle_record_to_model_sample(streusle_record)
                    predictions = self.predictor.predict(sample.xs, mask=self.predictor.get_sample_mask(sample.xs))
                    dists_predictions = self.predictor.predict_dist(sample.xs, mask=self.predictor.get_sample_mask(sample.xs))
                    if predictions and type(predictions[0]) is not tuple:
                        predictions = [(p.supersense_role, p.supersense_func) for p in predictions]
                    sent_data = streusle_record.build_data_with_supersenses(predictions, ident, supersenses_dists=dists_predictions)
                    sys_data.append(sent_data)
                print("Dumping sys file")
                json.dump(sys_data, sys_f)
                print("Done")

            with open(gold_fname, 'w') as gold_f:
                print("Dumping gold file")
                json.dump([gold_record_by_id[record.id].data for record in streusle_records], gold_f)
                print("Done")

            class Args:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            output = subprocess.check_output([sys.executable, self.psseval_script_path, gold_fname, sys_fname])
            with open(output_tsv_path, 'wb') as output_f:
                output_f.write(output)
            if all_depths:
                for depth in [1,2,3]:
                    output = subprocess.check_output([sys.executable, self.psseval_script_path, gold_fname, sys_fname, '--depth', str(depth)])
                    with open(output_tsv_path.replace('.tsv', '.depth_' + str(depth) + '.tsv'), 'wb') as output_f:
                        output_f.write(output)
            return output
        finally:
            if os.path.exists(gold_fname) and not gold_fname_out:
                os.remove(gold_fname)
            if os.path.exists(sys_fname) and not sys_fname_out:
                os.remove(sys_fname)
            if not keep_output_file and os.path.exists(output_tsv_path):
                os.remove(output_tsv_path)

    @staticmethod
    def average_evaluations(csv_paths, out_csv_path):
        tables = []
        for csv_path in csv_paths:
            with open(csv_path) as in_f:
                tables.append(list(csv.reader(in_f)))

        with open(out_csv_path, 'w') as out_f:
            writer = csv.writer(out_f)
            for rows in zip(*tables):
                row = []
                for cols in zip(*rows):
                    if len(set(cols)) == 1:
                        row.append(cols[0])
                    else:
                        assert all(isfloat(c) for c in cols)
                        cols = [float(c) for c in cols]
                        row.append("%f+-%f" % (mean(cols), stdev(cols)))
                writer.writerow(row)


if __name__ == '__main__':
    from random import random
    for i in range(1,4):
        with open('/tmp/' + str(i) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['head1', 'head2', 'head3'])
            writer.writerow([str(i + random() * 2 - 1) for i in range(1,4)])

    StreusleEvaluator.average_evaluations(['/tmp/1.csv', '/tmp/2.csv', '/tmp/3.csv'], '/tmp/out.csv')

    with open('/tmp/out.csv') as f:
        lines = list(csv.reader(f))
        for l in lines:
            print(l)
