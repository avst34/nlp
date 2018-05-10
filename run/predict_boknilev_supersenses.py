import os
import copy
from collections import defaultdict

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev, dump_boknilev_pss
from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.boknilev_integration import boknilev_record_to_lstm_model_sample_xs
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
import json
evaluator = PSSClasifierEvaluator()

def run():
    loader = StreusleLoader()
    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'
    task = 'goldid.goldsyn'
    train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
    dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
    test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')

    train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    test_features()

    GOLD_ID_AUTO_PREP = json.loads("""{
  "mask_mwes": false,
  "learning_rate_decay": 0.0001,
  "lstm_h_dim": 100,
  "mlp_layers": 2,
  "is_bilstm": true,
  "num_lstm_layers": 2,
  "dynet_random_seed": "7564313",
  "use_ud_xpos": true,
  "ner_embd_dim": 10,
  "allow_empty_prediction": false,
  "learning_rate": 0.15848931924611143,
  "mlp_activation": "relu",
  "use_lexcat": true,
  "use_govobj": true,
  "token_embd_dim": 300,
  "update_lemmas_embd": true,
  "govobj_config_embd_dim": 3,
  "ud_deps_embd_dim": 10,
  "mlp_layer_dim": 100,
  "mlp_dropout_p": 0.37,
  "ud_xpos_embd_dim": 25,
  "use_ner": true,
  "update_token_embd": false,
  "epochs": 1,
  "lstm_dropout_p": 0.38,
  "use_ud_dep": true,
  "lexcat_embd_dim": 3,
  "use_prep_onehot": false,
  "use_token": true,
  "use_token_internal": true,
  "token_internal_embd_dim": 10,
  "labels_to_predict": [
    "supersense_role",
    "supersense_func"
  ]
}""")

    print('Training model..')
    model = LstmMlpSupersensesModel(
        hyperparameters=LstmMlpSupersensesModel.HyperParameters(**GOLD_ID_AUTO_PREP),
    )
    predictor = model.fit(train_samples, dev_samples)
    evaluator = PSSClasifierEvaluator(predictor.model)
    evaluator.evaluate([model.sample_to_lowlevel(s) for s in test_samples])

    btrain, bdev, btest = load_boknilev()
    all_samples = btrain + bdev + btest
    predictions = {}
    for ind, sample in enumerate(all_samples):
        print("%d/%d" % (ind, len(all_samples)))
        lm_sample_xs = boknilev_record_to_lstm_model_sample_xs(sample)
        lm_sample_ys = model.predict(lm_sample_xs)
        predictions[sample['sent_id']] = {}
        for ind, (sx, sy) in enumerate(zip(lm_sample_xs, lm_sample_ys)):
            if sx.identified_for_pss:
                predictions[sample['sent_id']][ind] = (sy.supersense_role, sy.supersense_func)

    dump_boknilev_pss(predictions)

if __name__ == '__main__':
    run()