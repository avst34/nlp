import os

import time

from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.settings import GOLD_ID_GOLD_PREP, GOLD_ID_AUTO_PREP, AUTO_ID_AUTO_PREP
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
from models.supersenses.tuner_domains import PS
from run.dump_vocabs import dump_vocabs
from vocabulary import Vocabulary
import supersense_repo
import json
from hyperparameters_tuner import union_settings, override_settings
evaluator = PSSClasifierEvaluator()

def run(train_records, dev_records, test_records, streusle_loader):
    train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    # test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    test_features()

    RESULTS_BASE = os.environ.get('RESULTS_BASE') or '/cs/labs/oabend/aviramstern'
    BASELINE_RESULTS_BASE = RESULTS_BASE + '/baseline_' + time.strftime("%y%m%d_%H%M%S")

    if not os.path.exists(BASELINE_RESULTS_BASE):
        os.mkdir(BASELINE_RESULTS_BASE)

    task_hp = {
        'AUTO_ID_AUTO_PREP': {},
        'AUTO_ID_GOLD_PREP': {},
        'GOLD_ID_AUTO_PREP': json.loads("""{"mlp_activation": "tanh", "mlp_dropout_p": 0.15, "deps_from": "spacy", "token_internal_embd_dim": 100, "use_token_internal": true, "dynet_random_seed": "5604335", "use_prep_onehot": false, "labels_to_predict": ["supersense_role", "supersense_func"], "spacy_deps_embd_dim": 25, "mlp_layers": 2, "update_lemmas_embd": true, "lstm_h_dim": 20, "lstm_dropout_p": 0.2, "learning_rate": 0.15848931924611143, "token_embd_dim": 300, "use_dep": true, "ud_pos_embd_dim": 5, "spacy_pos_embd_dim": 5, "epochs": 80, "is_bilstm": true, "lemmas_from": "spacy", "update_token_embd": false, "spacy_ner_embd_dim": 10, "use_pos": true, "use_token": true, "mask_mwes": false, "use_spacy_ner": true, "ud_deps_embd_dim": 5, "mask_by": "sample-ys", "mlp_layer_dim": 40, "num_lstm_layers": 2, "learning_rate_decay": 0.0, "pos_from": "spacy"}"""),
        'GOLD_ID_GOLD_PREP': json.loads("""{"spacy_pos_embd_dim": 5, "spacy_ner_embd_dim": 10, "mask_by": "sample-ys", "spacy_deps_embd_dim": 5, "token_embd_dim": 300, "num_lstm_layers": 2, "use_spacy_ner": true, "labels_to_predict": ["supersense_role", "supersense_func"], "learning_rate": 0.15848931924611143, "use_token": true, "mlp_layers": 2, "update_token_embd": true, "lemmas_from": "ud", "ud_pos_embd_dim": 25, "lstm_h_dim": 40, "update_lemmas_embd": false, "mlp_layer_dim": 80, "epochs": 80, "use_prep_onehot": false, "mask_mwes": false, "lstm_dropout_p": 0.18, "use_pos": true, "deps_from": "ud", "dynet_random_seed": "854137", "mlp_dropout_p": 0.3, "learning_rate_decay": 0.0001, "is_bilstm": true, "pos_from": "ud", "ud_deps_embd_dim": 25, "mlp_activation": "tanh", "token_internal_embd_dim": 100, "use_dep": true, "use_token_internal": true}""")
    }

    for task, hp in task_hp.items():
        TASK_RESULTS_BASE = BASELINE_RESULTS_BASE + '/' + task
        os.mkdir(TASK_RESULTS_BASE)
        tuner = LstmMlpSupersensesModelHyperparametersTuner(
            results_csv_path=TASK_RESULTS_BASE,
            samples=train_samples, # use all after testing
            validation_samples=dev_samples,
            show_progress=True,
            show_epoch_eval=True,
            dump_models=True
        )

        tuner.sample_execution(hp)


    # best_params, best_results = tuner.tune(n_executions=1)
    # predictor = best_results.predictor
    #
    # predictor.save('/tmp/predictor')
    # loaded_predictor = LstmMlpSupersensesModel.load('/tmp/predictor')
    #
    # print('Predictor: original')
    #
    # evaluator = PSSClasifierEvaluator(predictor=predictor.model)
    # ll_samples = [predictor.sample_to_lowlevel(x) for x in dev_samples]
    # evaluator.evaluate(ll_samples, examples_to_show=5)
    #
    # print('Predictor: loaded')
    #
    # evaluator = PSSClasifierEvaluator(predictor=loaded_predictor.model)
    # ll_samples = [loaded_predictor.sample_to_lowlevel(x) for x in dev_samples]
    # evaluator.evaluate(ll_samples, examples_to_show=5)


    # tuner.sample_execution(json.loads(
    #     """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_role", "supersense_func"], "use_prep_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_spacy_ner": false, "deps_from": "ud", "lstm_dropout_p": 0.1, "use_dep": true}"""
    #     # """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_role"], "use_prep_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_spacy_ner": false, "deps_from": "ud", "lstm_dropout_p": 0.1, "use_dep": true}"""
    #     # """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_func"], "use_prep_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_spacy_ner": false, "deps_from": "ud", "lstm_dropout_p": 0.1, "use_dep": true}"""
    # ))
    # tuner.sample_execution()

    # print('LSTM-MLP evaluation:')
    # lstm_mlp_model = LstmMlpSupersensesModel(
    #     token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
    #     token_vocab=token_vocab,
    #     token_onehot_vocab=pp_vocab,
    #     pos_vocab=pos_vocab,
    #     dep_vocab=dep_vocab,
    #     supersense_vocab=pss_vocab,
    #     hyperparameters=LstmMlpSupersensesModel.HyperParameters(**json.loads(
    #         """ {"use_token_internal": false, "token_internal_embd_dim": 73, "mlp_layers": 2, "epochs": 100, "is_bilstm": true, "labels_to_predict": ["supersense_func"], "use_head": false, "mlp_activation": "relu", "learning_rate_decay": 0.01, "learning_rate": 0.1, "num_lstm_layers": 2, "mlp_dropout_p": 0.12, "pos_embd_dim": 50, "use_token": true, "use_prep_onehot": true, "use_pos": false, "update_token_embd": true, "mlp_layer_dim": 98, "token_embd_dim": 300, "use_spacy_ner": false, "deps_from": "ud", "lstm_dropout_p": 0.1, "use_dep": true, "lstm_h_dim": 100, "mask_by": "pos:IN,PRP$,RB,TO"}"""
    #     ))
    # )
    #
    # lstm_mlp_model.fit(train_samples,
    #                    dev_samples,
    #                    evaluator=evaluator)
    #
    # # evaluator = PSSClasifierEvaluator(predictor=lstm_mlp_model.model)
    # # ll_samples = [LstmMlpSupersensesModel.sample_to_lowlevel(x) for x in test_samples]
    # # evaluator.evaluate(ll_samples, examples_to_show=5)
