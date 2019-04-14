import json
import math
import random
import sys

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from datasets.pp_attachement.streusle.load_streusle import load_streusle
from evaluators.ppatt_evaluator import PPAttEvaluator
from hyperparameters_tuner import HyperparametersTuner, override_settings
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.hcpd_model_tuner import HCPDModelTuner
from models.hcpd.tuner_domains import TUNER_DOMAINS, PS

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()
strain_recs, sdev_recs, stest_recs = load_streusle()

ratio = math.ceil(len(train_recs) / len(strain_recs))
ratio_pps = math.ceil(len([pp for t in train_recs for pp in t['pps']]) / len([pp for t in strain_recs for pp in t['pps']]))
all_strain_recs = train_recs + strain_recs * ratio_pps
all_sdev_recs = dev_recs + sdev_recs * ratio_pps

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
# print('Final training samples size: %d sents, %d pps' % (len(all_train_recs), len([pp for t in all_train_recs for pp in t['pps']])))

print("Converting to classifier samples")
train_samples_boknilev = [s for r in train_recs for s in boknilev_record_to_hcpd_samples(r)]
train_samples_streusle = [s for r in strain_recs for s in boknilev_record_to_hcpd_samples(r)]
train_samples_streusle_boknilev = [s for r in all_strain_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_boknilev = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_streusle = [s for r in sdev_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_streusle_boknilev = [s for r in all_sdev_recs for s in boknilev_record_to_hcpd_samples(r)]
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
        row_tuples.append(("dataset", result_data['dataset']))
        row_tuples.append(("Best Epoch", result_data['scopes']['train']['best_epoch']))
        row_tuples.append(("Tuner Score", result.score))
        for scope, scope_data in sorted(result_data['scopes'].items()):
            row_tuples.append((scope + "_Acc", scope_data['acc']))
        row_tuples += [(k, str(v)) for k, v in sorted(params.items())]
        row_tuples.append(("Hyperparams Json", json.dumps(params)))

        headers = [x[0] for x in rows_tuples[0]]
        rows = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
        return headers, rows

    def _execute(self, hyperparameters):
        # modes = ['mix_scale_data', 'pipeline']
        modes = ['mix_scale_data']
        # modes = ['pipeline']
        datasets = ['boknilev', 'streusle']
        mode = random.choice(modes)
        dataset = random.choice(datasets)

        if dataset == 'boknilev':
            train = train_samples_boknilev
            dev = dev_samples_boknilev
            # test = test_samples_boknilev
        elif dataset == 'streusle':
            train = train_samples_streusle_boknilev
            dev = dev_samples_streusle_boknilev
            # test = test_samples_streusle
        else:
            raise Exception("Unknown dataset: " + dataset)

        model = HCPDModel(
            hyperparameters=HCPDModel.HyperParameters(**hyperparameters)
        )

        print('mode:', mode, 'dataset:', dataset)
        if mode == 'mix_scale_data':
            model.fit(train, validation_samples=dev, show_progress=True)
        elif mode == 'pipeline':
            print('training with boknilev')
            model.hyperparameters.mask_pss = True
            model.fit(train_samples_boknilev, validation_samples=dev_samples_boknilev)
            print('training with streusle')
            model.hyperparameters.mask_pss = False
            model.fit(train_samples_streusle, validation_samples=dev_samples_streusle, resume=True)
            # raise Exception("not supported")
        else:
            raise Exception('No such mode:' + mode)

        return HyperparametersTuner.ExecutionResult(
            result_data={
                'mode': mode,
                'dataset': dataset,
                'scopes': {
                    'train': self.tuner_results_getter(model.train_set_evaluation),
                    'dev': self.tuner_results_getter(model.dev_set_evaluation),
                    'dev_boknilev':  PPAttEvaluator(model).evaluate(dev_samples_boknilev, examples_to_show=0),
                    'test_boknilev':  PPAttEvaluator(model).evaluate(test_samples_boknilev, examples_to_show=0),
                    'dev_streusle':  PPAttEvaluator(model).evaluate(dev_samples_streusle, examples_to_show=0),
                    'test_streusle':  PPAttEvaluator(model).evaluate(test_samples_streusle, examples_to_show=0),
                }
            },
            score=self.tuner_score_getter(model.dev_set_evaluation),
            predictor=model
        )



model = Tuner(
    None,
    validation_samples=None,
    results_csv_path=sys.argv[-1],
    tuner_domains=override_settings([
        TUNER_DOMAINS
        # ,
        # [
        #     PS(name='trainer', values=["AdagradTrainer"]),
        # ]
    ])
)
# model.tune(1)
model.sample_execution(params=json.loads(random.sample([
    """
    {"mask_pss": false, "learning_rate": 0.1, "p1_mlp_layers": 1, "p2_vec_dim": 100, "use_pss": false, "dropout_p": 0.5, "update_embeddings": true, "fallback_to_lemmas": false, "pss_embd_type": "binary", "p2_mlp_layers": 1, "epochs": 100, "pss_embd_dim": 5, "p1_vec_dim": 100, "max_head_distance": 5, "use_verb_noun_ss": false, "learning_rate_decay": 0, "trainer": "SimpleSGDTrainer", "activation": "tanh"}
    """,
    """
    {"mask_pss": false, "dropout_p": 0.5, "fallback_to_lemmas": true, "activation": "tanh", "update_embeddings": true, "pss_embd_type": "lookup", "epochs": 100, "trainer": "SimpleSGDTrainer", "use_pss": true, "max_head_distance": 5, "p1_mlp_layers": 1, "p2_mlp_layers": 1, "p1_vec_dim": 100, "pss_embd_dim": 50, "learning_rate": 0.1, "learning_rate_decay": 0, "use_verb_noun_ss": false, "p2_vec_dim": 100}
    """
], 1)[0]))
print("Done tuning")

