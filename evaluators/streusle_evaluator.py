import os
import json
import time
import subprocess
from itertools import chain

from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

from datasets.streusle_v4.streusle import StreusleLoader

STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'

gold_records = StreusleLoader().load(STREUSLE_BASE + '/streusle.conllulex')
gold_record_by_id = {r.id: r for r in gold_records}

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
                    if predictions and type(predictions[0]) is not tuple:
                        predictions = [(p.supersense_role, p.supersense_func) for p in predictions]
                    sent_data = streusle_record.build_data_with_supersenses(predictions, ident)
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

            output = subprocess.check_output(['python', self.psseval_script_path, gold_fname, sys_fname])
            with open(output_tsv_path, 'wb') as output_f:
                output_f.write(output)
            if all_depths:
                for depth in [1,2,3]:
                    output = subprocess.check_output(['python', self.psseval_script_path, gold_fname, sys_fname, '--depth', str(depth)])
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

