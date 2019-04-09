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
    btrain, bdev, btest = load_boknilev()
    all_samples = btrain + bdev + btest

    task = 'goldid.autosyn'
    loader = StreusleLoader(load_elmo=True, task_name=task)
    train_records = loader.load_train()
    dev_records = loader.load_dev()
    test_records = loader.load_test()

    train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    test_features()

    GOLD_ID_AUTO_PREP = {
 'allow_empty_prediction': False,
 'dynet_random_seed': 'None',
 'elmo_layer': 1,
 'embd_type': 'elmo',
 'epochs': 130,
 'govobj_config_embd_dim': 3,
 'grandparent_dropout_p': 0.0,
 'is_bilstm': True,
 'labels_to_learn': ('supersense_role', 'supersense_func'),
 'labels_to_predict': ('supersense_role', 'supersense_func'),
 'learning_rate': 0.2,
 'learning_rate_decay': 0.0001,
 'lexcat_embd_dim': 3,
 'lstm_dropout_p': 0.42,
 'lstm_h_dim': 200,
 'mask_mwes': False,
 'mlp_activation': 'relu',
 'mlp_dropout_p': 0.14,
 'mlp_layer_dim': 200,
 'mlp_layers': 2,
 'ner_embd_dim': 10,
 'num_lstm_layers': 1,
 'parent_dropout_p': 0.0,
 'prep_dropout_p': 0.01,
 'pss_embd_dim': 5,
 'token_embd_dim': 300,
 'token_internal_embd_dim': 10,
 'trainer': 'sgd',
 'ud_deps_embd_dim': 10,
 'ud_xpos_embd_dim': 25,
 'update_lemmas_embd': False,
 'update_token_embd': True,
 'use_capitalized_word_follows': True,
 'use_func': False,
 'use_govobj': True,
 'use_grandparent': False,
 'use_instance_embd': False,
 'use_lemma': False,
 'use_lexcat': True,
 'use_ner': False,
 'use_parent': False,
 'use_prep': True,
 'use_prep_onehot': False,
 'use_role': False,
 'use_token': True,
 'use_token_internal': True,
 'use_ud_dep': False,
 'use_ud_xpos': False,
 }

    print('Training model..')
    model = LstmMlpSupersensesModel(
        hyperparameters=LstmMlpSupersensesModel.HyperParameters(**GOLD_ID_AUTO_PREP),
    )
    predictor = model.fit(train_samples, dev_samples, test_samples)
    evaluator = PSSClasifierEvaluator(predictor.model)
    evaluator.evaluate([model.sample_to_lowlevel(s) for s in test_samples])

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