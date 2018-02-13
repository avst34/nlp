import os
import json
import time
from itertools import chain

from datasets.streusle_v4.release import psseval
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

from datasets.streusle_v4.streusle import StreusleLoader

STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'

gold_records = StreusleLoader().load(STREUSLE_BASE + '/streusle.conllulex')
gold_record_by_id = {r.id: r for r in gold_records}

class StreusleEvaluator:

    def  __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, streusle_records, output_tsv_path=None, ident='autoid', gold_fname_out=None, sys_fname_out=None):
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
            with open(gold_fname, 'w') as gold_f:
                print("Dumping gold file")
                json.dump([gold_record_by_id[record.id].data for record in streusle_records], gold_f)
                print("Done")
            with open(sys_fname, 'w') as sys_f:
                sys_data = []
                for streusle_record in streusle_records:
                    sample = streusle_record_to_lstm_model_sample(streusle_record)
                    predictions = self.predictor.predict(sample.xs, mask=self.predictor.get_sample_mask(sample.xs, sample.ys))
                    predictions = [(p.supersense_role, p.supersense_func) for p in predictions]
                    sent_data = streusle_record.build_data_with_supersenses(predictions, ident)
                    for we in chain(sent_data.get('wes', {}).values(), sent_data.get('smwes', {}).values()):
                        assert we['ss'] != 'p.X' and we['ss2'] != 'p.X'
                    sys_data.append(sent_data)
                print("Dumping sys file")
                json.dump(sys_data, sys_f)
                print("Done")

            class Args:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            with open(output_tsv_path, 'w') as output_f:
                with open(gold_fname, 'r') as gold_f:
                    with open(sys_fname, 'r') as sys_f:
                        def to_tsv_wrap(*args, **kwargs):
                            return psseval.to_tsv(*args, **kwargs, file=output_f)
                        psseval.main(Args(
                            goldfile=gold_f,
                            sysfile=[sys_f],
                            depth=4,
                            output_format=to_tsv_wrap
                        ))
            with open(output_tsv_path, 'r') as output_f:
                return output_f.read()
        finally:
            if os.path.exists(gold_fname) and not gold_fname_out:
                os.remove(gold_fname)
            if os.path.exists(sys_fname) and not sys_fname_out:
                os.remove(sys_fname)
            if not keep_output_file and os.path.exists(output_tsv_path):
                os.remove(output_tsv_path)