import json
import os
import random

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from hyperparameters_tuner import override_settings
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.settings import ELMO_FASTTEXT_TASK_SETTINGS, PS
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

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

    # tasks = ['.'.join([id, syn]) for id in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    # tasks = list(FASTTEXT_TASK_SETTINGS.keys())
    tasks = {
        'goldid.goldsyn': json.loads("""{"learning_rate": 0.06309573444801933, "lstm_dropout_p": 0.27, "use_ud_xpos": true, "use_ner": true, "grandparent_dropout_p": 0.3, "use_parent": true, "parent_dropout_p": 0.2, "allow_empty_prediction": false, "embd_type": "fasttext_en", "num_lstm_layers": 1, "use_func": false, "mlp_dropout_p": 0.3, "use_token_internal": false, "dynet_random_seed": null, "elmo_layer": 2, "use_role": false, "use_instance_embd": false, "prep_dropout_p": 0.2, "is_bilstm": true, "ud_deps_embd_dim": 25, "use_ud_dep": true, "labels_to_predict": ["supersense_role", "supersense_func"], "lstm_h_dim": 20, "use_prep": true, "mlp_layer_dim": 100, "labels_to_learn": ["supersense_role", "supersense_func"], "use_lexcat": true, "learning_rate_decay": 0.0001, "token_embd_dim": 300, "token_internal_embd_dim": 100, "update_lemmas_embd": false, "mlp_layers": 2, "epochs": 100, "lexcat_embd_dim": 3, "ner_embd_dim": 5, "use_govobj": false, "pss_embd_dim": 10, "mask_mwes": false, "update_token_embd": false, "govobj_config_embd_dim": 3, "ud_xpos_embd_dim": 5, "use_prep_onehot": false, "use_grandparent": true, "use_token": true, "mlp_activation": "relu"}"""),
        'goldid.goldsyn.goldrole': json.loads("""{"use_instance_embd": false, "mask_mwes": false, "ud_xpos_embd_dim": 25, "use_func": false, "labels_to_predict": ["supersense_func"], "use_govobj": false, "prep_dropout_p": 0.2, "use_token": true, "token_internal_embd_dim": 25, "token_embd_dim": 300, "mlp_activation": "cube", "grandparent_dropout_p": 0.2, "lstm_dropout_p": 0.36, "use_token_internal": false, "mlp_dropout_p": 0.17, "elmo_layer": 0, "lexcat_embd_dim": 3, "ner_embd_dim": 10, "update_token_embd": true, "ud_deps_embd_dim": 5, "update_lemmas_embd": false, "dynet_random_seed": null, "use_ud_dep": true, "use_ud_xpos": false, "labels_to_learn": ["supersense_func"], "use_lexcat": true, "learning_rate_decay": 3.1622776601683795e-05, "govobj_config_embd_dim": 3, "use_role": true, "epochs": 100, "pss_embd_dim": 10, "allow_empty_prediction": false, "lstm_h_dim": 80, "parent_dropout_p": 0.3, "is_bilstm": true, "use_parent": true, "mlp_layers": 2, "use_prep": true, "mlp_layer_dim": 80, "learning_rate": 0.01, "use_grandparent": true, "use_ner": true, "use_prep_onehot": false, "num_lstm_layers": 1, "embd_type": "fasttext_en"}"""),
        'goldid.goldsyn.goldfunc': json.loads("""{"use_govobj": false, "use_prep_onehot": false, "update_lemmas_embd": false, "ner_embd_dim": 10, "govobj_config_embd_dim": 3, "num_lstm_layers": 2, "embd_type": "fasttext_en", "use_token": true, "ud_deps_embd_dim": 5, "use_token_internal": false, "elmo_layer": 0, "parent_dropout_p": 0.2, "use_grandparent": true, "use_lexcat": true, "update_token_embd": false, "learning_rate": 0.15848931924611143, "dynet_random_seed": null, "use_ner": true, "is_bilstm": true, "use_func": true, "mlp_dropout_p": 0.16, "lstm_dropout_p": 0.39, "prep_dropout_p": 0.2, "allow_empty_prediction": false, "labels_to_learn": ["supersense_role"], "mask_mwes": false, "use_instance_embd": false, "lstm_h_dim": 100, "mlp_layer_dim": 100, "labels_to_predict": ["supersense_role"], "use_parent": true, "use_ud_xpos": false, "lexcat_embd_dim": 3, "pss_embd_dim": 5, "token_internal_embd_dim": 300, "mlp_layers": 3, "ud_xpos_embd_dim": 25, "use_role": false, "mlp_activation": "tanh", "epochs": 100, "learning_rate_decay": 0.0031622776601683794, "use_ud_dep": true, "token_embd_dim": 300, "grandparent_dropout_p": 0.3, "use_prep": true}""")
    }
    # task = 'goldid.goldsyn'
    for task, params in tasks.items():
        loader = StreusleLoader(load_elmo=True)
        train_records = loader.load_train()
        dev_records = loader.load_dev()
        test_records = loader.load_test()
        # chinese_test_records = StreusleLoader(load_elmo=True).load(conllulex_path=os.path.dirname(__file__) + '/../datasets/streusle_v4/chinese/lp.eng.zh_pss.all.json', input_format='json')
        # records = StreusleLoader().load(conllulex_path='/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/chinese/lp.chinese.all.json', input_format='json')
        # train_records = records[0::3]
        # dev_records = records[1::3]
        # test_records = records[2::3]

        print_samples_statistics('train', train_records)
        print_samples_statistics('dev', dev_records)
        print_samples_statistics('test', test_records)

        train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
        dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
        test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

        test_features()

        tuner = LstmMlpSupersensesModelHyperparametersTuner(
            task_name=task,
            results_csv_path=os.environ.get('RESULTS_PATH') or '/tmp/eval.csv',
            samples=train_samples, # use all after testing
            validation_samples=dev_samples,
            test_samples=test_samples,
            show_progress=True,
            show_epoch_eval=True,
            tuner_domains=override_settings([
                ELMO_FASTTEXT_TASK_SETTINGS[task],
                [
                    # PS(name='epochs', values=[10])
                ]
            ]),
            dump_models=False
        )

        best_params, best_results = tuner.sample_execution(params=params)
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