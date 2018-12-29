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
import random

evaluator = PSSClasifierEvaluator()

best_params_per_task = {
    'goldid.goldsyn': json.loads("""{"use_token_internal": false, "use_lexcat": true, "use_ud_dep": true, "learning_rate": 0.06309573444801933, "is_bilstm": true, "use_parent": true, "prep_dropout_p": 0.5, "update_token_embd": false, "num_lstm_layers": 1, "use_prep_onehot": false, "use_grandparent": true, "update_lemmas_embd": true, "use_role": false, "use_ud_xpos": true, "lexcat_embd_dim": 3, "mlp_dropout_p": 0.11, "epochs": 100, "mlp_layer_dim": 80, "ud_xpos_embd_dim": 10, "use_func": false, "lstm_h_dim": 40, "mask_mwes": false, "use_instance_embd": false, "mlp_activation": "tanh", "allow_empty_prediction": false, "use_token": true, "govobj_config_embd_dim": 3, "use_prep": true, "embd_type": "muse", "labels_to_learn": ["supersense_role", "supersense_func"], "labels_to_predict": ["supersense_role", "supersense_func"], "token_embd_dim": 300, "mlp_layers": 2, "dynet_random_seed": null, "pss_embd_dim": 5, "use_govobj": false, "ner_embd_dim": 10, "lstm_dropout_p": 0.5, "token_internal_embd_dim": 25, "ud_deps_embd_dim": 10, "learning_rate_decay": 0.00031622776601683794, "use_ner": true}"""),
    'goldid.goldsyn.goldrole': json.loads("""{"use_role": true, "use_lexcat": true, "use_govobj": false, "lstm_dropout_p": 0.02, "update_token_embd": false, "prep_dropout_p": 0, "pss_embd_dim": 10, "govobj_config_embd_dim": 3, "use_prep_onehot": false, "lstm_h_dim": 20, "labels_to_learn": ["supersense_func"], "use_token": true, "learning_rate_decay": 0.001, "lexcat_embd_dim": 3, "use_grandparent": true, "num_lstm_layers": 1, "mlp_dropout_p": 0.09, "use_prep": true, "mlp_layers": 3, "use_parent": true, "allow_empty_prediction": false, "mask_mwes": false, "learning_rate": 0.01, "ud_deps_embd_dim": 10, "dynet_random_seed": null, "ud_xpos_embd_dim": 25, "mlp_activation": "tanh", "use_token_internal": false, "use_instance_embd": false, "ner_embd_dim": 10, "mlp_layer_dim": 100, "use_ud_dep": true, "update_lemmas_embd": false, "is_bilstm": true, "use_ud_xpos": true, "use_ner": true, "epochs": 100, "embd_type": "muse", "token_embd_dim": 300, "use_func": false, "token_internal_embd_dim": 10, "labels_to_predict": ["supersense_func"]}"""),
    'goldid.goldsyn.goldfunc': json.loads("""{"use_token": true, "ud_deps_embd_dim": 25, "lexcat_embd_dim": 3, "labels_to_predict": ["supersense_role"], "learning_rate_decay": 0.00031622776601683794, "use_role": false, "use_govobj": false, "epochs": 100, "learning_rate": 0.15848931924611143, "use_ud_xpos": true, "use_ud_dep": true, "ud_xpos_embd_dim": 10, "use_prep_onehot": false, "lstm_h_dim": 200, "mlp_activation": "relu", "mask_mwes": false, "embd_type": "muse", "use_ner": true, "token_internal_embd_dim": 25, "ner_embd_dim": 5, "pss_embd_dim": 10, "use_token_internal": false, "use_lexcat": true, "govobj_config_embd_dim": 3, "num_lstm_layers": 1, "use_grandparent": true, "prep_dropout_p": 0.3, "update_lemmas_embd": false, "mlp_layers": 2, "is_bilstm": true, "lstm_dropout_p": 0.13, "labels_to_learn": ["supersense_role"], "allow_empty_prediction": false, "mlp_dropout_p": 0.22, "use_instance_embd": false, "token_embd_dim": 300, "mlp_layer_dim": 200, "dynet_random_seed": null, "use_parent": true, "use_func": true, "use_prep": true, "update_token_embd": true}"""),
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

    chinese_test_records = StreusleLoader().load(conllulex_path=os.path.dirname(__file__) + '../datasets/streusle_v4/chinese/lp.eng.zh_pss.all.json', input_format='json')

    tasks = list(best_params_per_task.items())
    random.shuffle(tasks)
    for task, params in tasks:
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
                    # PS(name='embd_type', values=['fasttext_en'])
                ]
            ]),
            dump_models=False
        )
        params['epochs'] = 130
        # params['embd_type'] = 'fasttext_en'
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