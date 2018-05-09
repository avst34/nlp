import os
import random

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.settings import GOLD_ID_GOLD_PREP, GOLD_ID_AUTO_PREP, AUTO_ID_AUTO_PREP, AUTO_ID_GOLD_PREP, \
    TASK_SETTINGS
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

def print_samples_statistics(name, samples):
    sentences = len(samples)
    tokens = len([tok for s in samples for tok in s.tagged_tokens])
    prepositions = len([tok for s in samples for tok in s.tagged_tokens if tok.ud_xpos in ['IN', 'PRP$', 'RB', 'TO']])
    labeled_prepositions = len([tok for s in samples for tok in s.tagged_tokens if tok.ud_xpos in ['IN', 'PRP$', 'RB', 'TO'] and tok.supersense_combined])
    labels_lost = len([tok for s in samples for tok in s.tagged_tokens if tok.ud_xpos not in ['IN', 'PRP$', 'RB', 'TO'] and tok.supersense_combined])

    print("Set: %s, Sentences: %d, Tokens: %d, Prepositions: %d, Labeled prepositions: %d, Labels Lost: %d" % (name,
                                                                                                               sentences,
                                                                                                               tokens,
                                                                                                               prepositions,
                                                                                                               labeled_prepositions,
                                                                                                               labels_lost)
          )


def run():

    tasks = ['.'.join([id, syn]) for id in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    task = random.choice(tasks)
    task = 'autoid.autosyn'
    for task in [task]:
        loader = StreusleLoader()
        STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'
        train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
        dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
        test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')

        print_samples_statistics('train', train_records)
        print_samples_statistics('dev', dev_records)
        print_samples_statistics('test', test_records)

        train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
        dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
        test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

        test_features()

        tuner = LstmMlpSupersensesModelHyperparametersTuner(
            task_name=task,
            results_csv_path=os.environ.get('RESULTS_PATH') or '/cs/labs/oabend/aviramstern/results.csv',
            samples=train_samples, # use all after testing
            validation_samples=dev_samples,
            show_progress=True,
            show_epoch_eval=True,
            tuner_domains=override_settings([
                TASK_SETTINGS[task],
                # [PS(name='epochs', values=[1])] # remove after testing
            ]),
            dump_models=False
        )

        best_params, best_results = tuner.tune(n_executions=1)
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
