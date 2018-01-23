import os
import json
import time

from datasets.streusle_v4.streusle_4alpha import psseval
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

from datasets.streusle_v4.streusle import StreusleLoader

records = sum(StreusleLoader().load(), [])
record_by_id = {r.id: r for r in records}

class StreusleEvaluator:

    def  __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, streusle_samples, output_tsv_path=None, ident='goldid'):
        rand = str(int(time.time() * 1000))
        gold_fname = 'gold_' + rand + '.json'
        sys_fname = 'sys_' + rand + '.' + ident + '.json'

        if output_tsv_path:
            keep_output_file = True
        else:
            keep_output_file = False
            output_tsv_path = 'output_' + rand + '.json'

        try:
            with open(gold_fname, 'w') as gold_f:
                json.dump([record_by_id[sample.sample_id].data for sample in streusle_samples], gold_f)
            with open(sys_fname, 'w') as sys_f:
                sys_data = []
                for sample in streusle_samples:
                    streusle_record = record_by_id[sample.sample_id]
                    predictions = self.predictor.predict(sample.xs, mask=self.predictor.get_sample_mask(sample.xs, sample.ys))
                    predictions = [(p.supersense_role, p.supersense_func) for p in predictions]
                    sent_data = streusle_record.build_data_with_supersenses(predictions, allow_new=False)
                    sys_data.append(sent_data)
                json.dump(sys_data, sys_f)

            class Args:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            with open(output_tsv_path, 'w') as output_f:
                with open(gold_fname, 'r') as gold_f:
                    with open(sys_fname, 'r') as sys_f:
                        def to_tsv_wrap(*args, **kwargs):
                            return psseval.to_tsv(*args, **kwargs, file=output_f)
                        psseval.    main(Args(
                            goldfile=gold_f,
                            sysfile=[sys_f],
                            depth=4,
                            output_format=to_tsv_wrap
                        ))
            with open(output_tsv_path, 'r') as output_f:
                return output_f.read()
        finally:
            if os.path.exists(gold_fname):
                os.remove(gold_fname)
            if os.path.exists(sys_fname):
                os.remove(sys_fname)
            if not keep_output_file and os.path.exists(output_tsv_path):
                os.remove(output_tsv_path)