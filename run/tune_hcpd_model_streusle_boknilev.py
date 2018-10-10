import json
import math
import random
import sys

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from datasets.pp_attachement.streusle.load_streusle import load_streusle
from evaluators.ppatt_evaluator import PPAttEvaluator
from hyperparameters_tuner import HyperparametersTuner
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.hcpd_model_tuner import HCPDModelTuner

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()
strain_recs, sdev_recs, stest_recs = load_streusle()

ratio = math.ceil(len(train_recs) / len(strain_recs))
ratio_pps = math.ceil(len([pp for t in train_recs for pp in t['pps']]) / len([pp for t in strain_recs for pp in t['pps']]))
all_train_recs = train_recs + strain_recs * ratio_pps

print('Train')
print('Boknilev/Streusle - sentences: %d/%d' % (len(train_recs), len(strain_recs)))
print('Boknilev/Streusle - pps      : %d/%d' % (len([pp for t in train_recs for pp in t['pps']]), len([pp for t in strain_recs for pp in t['pps']])))
print('Dev')
print('Boknilev/Streusle - sentences: %d/%d' % (len(dev_recs), len(sdev_recs)))
print('Boknilev/Streusle - pps      : %d/%d' % (len([pp for t in dev_recs for pp in t['pps']]), len([pp for t in sdev_recs for pp in t['pps']])))
print('Test')
print('Boknilev/Streusle - sentences: %d/%d' % (len(test_recs), len(stest_recs)))
print('Boknilev/Streusle - pps      : %d/%d' % (len([pp for t in test_recs for pp in t['pps']]), len([pp for t in stest_recs for pp in t['pps']])))
print('Ratio: %d' % ratio)
print('Ratio (pps): %d' % ratio_pps)
print('Final training samples size: %d sents, %d pps' % (len(all_train_recs), len([pp for t in all_train_recs for pp in t['pps']])))

print("Converting to classifier samples")
mixed_train_samples = [s for r in all_train_recs for s in boknilev_record_to_hcpd_samples(r)]
train_samples_boknilev = [s for r in train_recs for s in boknilev_record_to_hcpd_samples(r)]
train_samples_streusle = [s for r in strain_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_boknilev = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_streusle = [s for r in sdev_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples_boknilev = [s for r in test_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples_streusle = [s for r in stest_recs for s in boknilev_record_to_hcpd_samples(r)]


print("Tuning..")

class Tuner(HCPDModelTuner):

    def build_csv_rows(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        result_data = result.result_data
        rows_tuples = [[]]
        row_tuples = rows_tuples[0]
        row_tuples.append(("mode", result_data['mode']))
        row_tuples.append(("Best Epoch", result_data['scopes']['train']['best_epoch']))
        row_tuples += [(k, str(v)) for k, v in sorted(params.items())]
        for scope, scope_data in sorted(result_data['scopes'].items()):
            row_tuples.append((scope + "_Acc", scope_data['acc']))
        row_tuples.append(("Hyperparams Json", json.dumps(params)))

        headers = [x[0] for x in rows_tuples[0]]
        rows = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
        return headers, rows


    def _execute(self, hyperparameters):
        modes = ['mix_scale_data', 'pipeline']
        mode = random.choice(modes)
        model = HCPDModel(
            hyperparameters=HCPDModel.HyperParameters(**hyperparameters)
        )

        print('mode:', mode)
        if mode == 'mix_scale_data':
            model.fit(mixed_train_samples, validation_samples=dev_samples_streusle, show_progress=True)
        elif mode == 'pipeline':
            print('training with boknilev')
            model.fit(train_samples_boknilev, validation_samples=dev_samples_boknilev)
            print('training with streusle')
            model.fit(train_samples_streusle, validation_samples=dev_samples_streusle, resume=True)
        else:
            raise Exception('No such mode:' + mode)

        return HyperparametersTuner.ExecutionResult(
            result_data={
                'mode': mode,
                'scopes': {
                    'train': PPAttEvaluator(model).evaluate(mixed_train_samples, examples_to_show=0),
                    'dev_boknilev':  PPAttEvaluator(model).evaluate(dev_samples_boknilev, examples_to_show=0),
                    'test_boknilev':  PPAttEvaluator(model).evaluate(test_samples_boknilev, examples_to_show=0),
                    'dev_streusle':  PPAttEvaluator(model).evaluate(dev_samples_streusle, examples_to_show=0),
                    'test_streusle':  PPAttEvaluator(model).evaluate(test_samples_streusle, examples_to_show=0),
                }
            },
            score=self.tuner_score_getter(model.test_set_evaluation),
            predictor=model
        )



model = Tuner(
    mixed_train_samples,
    validation_samples=dev_samples_streusle,
    results_csv_path=sys.argv[-1]
)
model.tune(1)
print("Done tuning")

