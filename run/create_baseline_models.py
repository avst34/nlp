import os

import time

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
import json
evaluator = PSSClasifierEvaluator()

def run():

    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'

    test_features()

    RESULTS_BASE = os.environ.get('RESULTS_BASE') or '/cs/labs/oabend/aviramstern'
    BASELINE_RESULTS_BASE = RESULTS_BASE + '/baseline_' + time.strftime("%y%m%d_%H%M%S")

    if not os.path.exists(BASELINE_RESULTS_BASE):
        os.mkdir(BASELINE_RESULTS_BASE)

    task_hp = {
        'autoid.autosyn': json.loads("""{  "mask_mwes": false,  "learning_rate_decay": 0.0,  "lstm_h_dim": 80,  "mlp_layers": 2,  "is_bilstm": true,  "num_lstm_layers": 2,  "dynet_random_seed": "3592752",  "use_ud_xpos": true,  "ner_embd_dim": 5,  "allow_empty_prediction": false,  "learning_rate": 0.15848931924611143,  "mlp_activation": "tanh",  "use_lexcat": true,  "use_govobj": true,  "token_embd_dim": 300,  "update_lemmas_embd": true,  "govobj_config_embd_dim": 3,  "ud_deps_embd_dim": 5,  "mlp_layer_dim": 80,  "mlp_dropout_p": 0.32,  "ud_xpos_embd_dim": 5,  "use_ner": true,  "update_token_embd": false,  "epochs": 80,  "lstm_dropout_p": 0.45,  "use_ud_dep": true,  "lexcat_embd_dim": 3,  "use_prep_onehot": false,  "use_token": true,  "use_token_internal": true,  "token_internal_embd_dim": 50,  "labels_to_predict": [    "supersense_role",    "supersense_func"  ]}"""),
        'autoid.goldsyn': json.loads("""{  "mask_mwes": false,  "learning_rate_decay": 0.0,  "lstm_h_dim": 100,  "mlp_layers": 2,  "is_bilstm": true,  "num_lstm_layers": 2,  "dynet_random_seed": "1434844",  "use_ud_xpos": true,  "ner_embd_dim": 5,  "allow_empty_prediction": false,  "learning_rate": 0.15848931924611143,  "mlp_activation": "tanh",  "use_lexcat": true,  "use_govobj": true,  "token_embd_dim": 300,  "update_lemmas_embd": true,  "govobj_config_embd_dim": 3,  "ud_deps_embd_dim": 25,  "mlp_layer_dim": 80,  "mlp_dropout_p": 0.31,  "ud_xpos_embd_dim": 25,  "use_ner": true,  "update_token_embd": false,  "epochs": 80,  "lstm_dropout_p": 0.24,  "use_ud_dep": true,  "lexcat_embd_dim": 3,  "use_prep_onehot": false,  "use_token": true,  "use_token_internal": true,  "token_internal_embd_dim": 100,  "labels_to_predict": [    "supersense_role",    "supersense_func"  ]}"""),
        'goldid.autosyn': json.loads("""{"mask_mwes": false,"learning_rate_decay": 0.0001,"lstm_h_dim": 100,"mlp_layers": 2,"is_bilstm": true,"num_lstm_layers": 2,"dynet_random_seed": "7564313","use_ud_xpos": true,"ner_embd_dim": 10,"allow_empty_prediction": false,"learning_rate": 0.15848931924611143,"mlp_activation": "relu","use_lexcat": true,"use_govobj": true,"token_embd_dim": 300,"update_lemmas_embd": true,"govobj_config_embd_dim": 3,"ud_deps_embd_dim": 10,"mlp_layer_dim": 100,"mlp_dropout_p": 0.37,"ud_xpos_embd_dim": 25,"use_ner": true,"update_token_embd": false,"epochs": 80,"lstm_dropout_p": 0.38,"use_ud_dep": true,"lexcat_embd_dim": 3,"use_prep_onehot": false,"use_token": true,"use_token_internal": true,"token_internal_embd_dim": 10,"labels_to_predict": [  "supersense_role",  "supersense_func"]}"""),
        'goldid.goldsyn': json.loads("""{"mask_mwes": false,"learning_rate_decay": 0.00031622776601683794,"lstm_h_dim": 100,"mlp_layers": 2,"is_bilstm": true,"num_lstm_layers": 2,"dynet_random_seed": "3857654","use_ud_xpos": true,"ner_embd_dim": 5,"allow_empty_prediction": false,"learning_rate": 0.15848931924611143,"mlp_activation": "relu","use_lexcat": true,"use_govobj": true,"token_embd_dim": 300,"update_lemmas_embd": false,"govobj_config_embd_dim": 3,"ud_deps_embd_dim": 25,"mlp_layer_dim": 100,"mlp_dropout_p": 0.42,"ud_xpos_embd_dim": 5,"use_ner": true,"update_token_embd": false,"epochs": 80,"lstm_dropout_p": 0.49,"use_ud_dep": true,"lexcat_embd_dim": 3,"use_prep_onehot": false,"use_token": true,"use_token_internal": true,"token_internal_embd_dim": 10,"labels_to_predict": [  "supersense_role",  "supersense_func"]}""")
    }

    for task, hp in task_hp.items():
        loader = StreusleLoader()
        train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
        dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
        # test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')

        train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
        dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
        # test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

        TASK_RESULTS_BASE = BASELINE_RESULTS_BASE + '/' + task
        os.mkdir(TASK_RESULTS_BASE)
        tuner = LstmMlpSupersensesModelHyperparametersTuner(
            results_csv_path=BASELINE_RESULTS_BASE + '/results.csv',
            samples=train_samples, # use all after testing
            validation_samples=dev_samples,
            show_progress=True,
            show_epoch_eval=True,
            report_epoch_scores=True,
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

if __name__ == '__main__':
    run()