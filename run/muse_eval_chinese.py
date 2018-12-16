import json
import os

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from hyperparameters_tuner import override_settings
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.settings import MUSE_TASK_SETTINGS
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

evaluator = PSSClasifierEvaluator()

best_params_per_task = {
    'goldid.goldsyn': json.loads("""{"use_parent": true, "lstm_h_dim": 100, "allow_empty_prediction": false, "use_lexcat": true, "labels_to_learn": ["supersense_role", "supersense_func"], "token_embd_dim": 300, "mlp_layers": 2, "lstm_dropout_p": 0.47000000000000003, "use_ud_xpos": false, "is_bilstm": true, "mlp_activation": "cube", "use_instance_embd": false, "use_ner": true, "learning_rate_decay": 0.001, "embd_type": "muse", "use_prep_onehot": false, "use_govobj": false, "ud_deps_embd_dim": 25, "dynet_random_seed": null, "use_token": true, "token_internal_embd_dim": 100, "govobj_config_embd_dim": 3, "mlp_layer_dim": 200, "pss_embd_dim": 10, "mask_mwes": false, "epochs": 100, "use_grandparent": true, "use_func": false, "update_token_embd": true, "ner_embd_dim": 5, "lexcat_embd_dim": 3, "labels_to_predict": ["supersense_role", "supersense_func"], "ud_xpos_embd_dim": 5, "num_lstm_layers": 1, "use_role": false, "update_lemmas_embd": true, "use_token_internal": false, "use_ud_dep": true, "mlp_dropout_p": 0.36, "learning_rate": 0.06309573444801933}"""),
    'goldid.goldsyn.goldrole': json.loads("""{"mask_mwes": false, "learning_rate_decay": 0.0031622776601683794, "mlp_activation": "cube", "use_parent": true, "govobj_config_embd_dim": 3, "is_bilstm": true, "use_token_internal": false, "mlp_dropout_p": 0.25, "ud_deps_embd_dim": 25, "lexcat_embd_dim": 3, "allow_empty_prediction": false, "use_ner": true, "embd_type": "muse", "labels_to_predict": ["supersense_func"], "use_instance_embd": false, "learning_rate": 0.06309573444801933, "lstm_h_dim": 100, "use_role": true, "epochs": 100, "mlp_layers": 2, "lstm_dropout_p": 0.5, "labels_to_learn": ["supersense_func"], "token_internal_embd_dim": 25, "use_prep_onehot": false, "use_ud_xpos": false, "ud_xpos_embd_dim": 5, "use_govobj": false, "use_func": false, "use_lexcat": true, "use_token": true, "num_lstm_layers": 1, "dynet_random_seed": null, "pss_embd_dim": 10, "ner_embd_dim": 5, "mlp_layer_dim": 200, "update_token_embd": true, "token_embd_dim": 300, "use_grandparent": true, "use_ud_dep": true, "update_lemmas_embd": true}"""),
    'goldid.goldsyn.goldfunc': json.loads("""{"use_role": false, "lstm_dropout_p": 0.5, "pss_embd_dim": 5, "lstm_h_dim": 40, "mlp_layers": 3, "ner_embd_dim": 10, "update_token_embd": true, "use_ner": true, "use_func": true, "use_grandparent": true, "use_parent": true, "embd_type": "muse", "update_lemmas_embd": true, "ud_deps_embd_dim": 5, "use_ud_xpos": false, "mlp_dropout_p": 0.09, "token_embd_dim": 300, "use_ud_dep": true, "ud_xpos_embd_dim": 10, "mlp_layer_dim": 200, "govobj_config_embd_dim": 3, "is_bilstm": true, "mask_mwes": false, "labels_to_learn": ["supersense_role"], "learning_rate_decay": 0.0001, "epochs": 100, "dynet_random_seed": null, "labels_to_predict": ["supersense_role"], "allow_empty_prediction": false, "token_internal_embd_dim": 300, "use_lexcat": true, "use_govobj": false, "use_prep_onehot": false, "use_token_internal": false, "use_instance_embd": false, "use_token": true, "num_lstm_layers": 1, "learning_rate": 0.15848931924611143, "lexcat_embd_dim": 3, "mlp_activation": "tanh"}"""),
}

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
    # tasks = list(MUSE_TASK_SETTINGS.keys())
    # task = 'goldid.goldsyn'
    loader = StreusleLoader(load_elmo=False)
    train_records = loader.load_train()
    dev_records = loader.load_dev()
    test_records = loader.load_test()

    chinese_test_records = StreusleLoader().load(conllulex_path='/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/chinese/lp.chinese.all.json', input_format='json')

    for task, params in best_params_per_task.items():

        print_samples_statistics('train', train_records)
        print_samples_statistics('dev', dev_records)
        print_samples_statistics('test', test_records)

        train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
        dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
        test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

        chinese_test_samples = [streusle_record_to_lstm_model_sample(r) for r in chinese_test_records]

        test_features()

        tuner = LstmMlpSupersensesModelHyperparametersTuner(
            task_name=task,
            results_csv_path=os.environ.get('RESULTS_PATH') or '/cs/labs/oabend/aviramstern/muse_chiense_results.csv',
            samples=train_samples, # use all after testing
            validation_samples=dev_samples,
            test_samples=chinese_test_samples,
            show_progress=True,
            show_epoch_eval=True,
            tuner_domains=override_settings([
                MUSE_TASK_SETTINGS[task],
                [
                    # PS(name='epochs', values=[1]),
                    # PS(name='embd_type', values=['muse_dict'])
                ]
            ]),
            dump_models=False
        )
        # params['epochs'] = 1
        # params['embd_type'] = 'muse_dict'
        best_params, best_results = tuner.sample_execution(params)
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