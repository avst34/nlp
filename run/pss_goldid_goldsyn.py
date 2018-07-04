import os
import copy
from collections import defaultdict

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
import json

from process_tuner_result import evaluate_model_on_task

evaluator = PSSClasifierEvaluator()

def run():
    loader = StreusleLoader()
    # STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'
    task = 'goldid.goldsyn'
    # train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
    # dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
    # test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')
    #
    # train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    # dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    # # test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    test_features()

    GOLD_ID_GOLD_PREP = json.loads("""{
  "mask_mwes": false,
  "learning_rate_decay": 0.00031622776601683794,
  "lstm_h_dim": 100,
  "mlp_layers": 2,
  "is_bilstm": true,
  "num_lstm_layers": 2,
  "dynet_random_seed": "3857654",
  "use_ud_xpos": true,
  "ner_embd_dim": 5,
  "allow_empty_prediction": false,
  "learning_rate": 0.15848931924611143,
  "mlp_activation": "relu",
  "use_lexcat": true,
  "use_govobj": true,
  "token_embd_dim": 300,
  "update_lemmas_embd": false,
  "govobj_config_embd_dim": 3,
  "ud_deps_embd_dim": 25,
  "mlp_layer_dim": 100,
  "mlp_dropout_p": 0.42,
  "ud_xpos_embd_dim": 5,
  "use_ner": true,
  "update_token_embd": false,
  "epochs": 80,
  "lstm_dropout_p": 0.49,
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

    model = LstmMlpSupersensesModel(
        hyperparameters=LstmMlpSupersensesModel.HyperParameters(**GOLD_ID_GOLD_PREP),
    )
    # predictor = model.fit(train_samples, dev_samples)
    evaluate_model_on_task('goldid.goldsyn', model, streusle_record_to_lstm_model_sample, '/tmp/dists', save_model=True)



if __name__ == '__main__':
    run()