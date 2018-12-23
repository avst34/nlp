import json

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.preprocessing import preprocess_sentence
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

evaluator = PSSClasifierEvaluator()

best_params_per_task = {
    'goldid.goldsyn': json.loads("""{"mask_mwes": false, "use_ud_xpos": true, "use_instance_embd": false, "ud_deps_embd_dim": 25, "use_lexcat": true, "token_embd_dim": 300, "use_token_internal": false, "dynet_random_seed": null, "use_parent": true, "update_lemmas_embd": true, "lstm_dropout_p": 0.36, "use_govobj": false, "learning_rate": 0.15848931924611143, "token_internal_embd_dim": 25, "govobj_config_embd_dim": 3, "allow_empty_prediction": false, "labels_to_learn": ["supersense_role", "supersense_func"], "use_prep": true, "ud_xpos_embd_dim": 5, "learning_rate_decay": 0.00031622776601683794, "ner_embd_dim": 5, "mlp_activation": "tanh", "num_lstm_layers": 2, "use_grandparent": true, "mlp_layers": 2, "is_bilstm": true, "use_token": true, "use_role": false, "use_ud_dep": true, "labels_to_predict": ["supersense_role", "supersense_func"], "mlp_dropout_p": 0.32, "use_func": false, "use_prep_onehot": false, "prep_dropout_p": 0.3, "lstm_h_dim": 80, "update_token_embd": false, "use_ner": true, "pss_embd_dim": 10, "epochs": 100, "embd_type": "fasttext_en", "mlp_layer_dim": 100, "lexcat_embd_dim": 3}"""),
    # 'goldid.goldsyn': json.loads("""{"use_parent": true, "lstm_h_dim": 100, "allow_empty_prediction": false, "use_lexcat": true, "labels_to_learn": ["supersense_role", "supersense_func"], "token_embd_dim": 300, "mlp_layers": 2, "lstm_dropout_p": 0.47000000000000003, "use_ud_xpos": false, "is_bilstm": true, "mlp_activation": "cube", "use_instance_embd": false, "use_ner": true, "learning_rate_decay": 0.001, "embd_type": "muse", "use_prep_onehot": false, "use_govobj": false, "ud_deps_embd_dim": 25, "dynet_random_seed": null, "use_token": true, "token_internal_embd_dim": 100, "govobj_config_embd_dim": 3, "mlp_layer_dim": 200, "pss_embd_dim": 10, "mask_mwes": false, "epochs": 100, "use_grandparent": true, "use_func": false, "update_token_embd": true, "ner_embd_dim": 5, "lexcat_embd_dim": 3, "labels_to_predict": ["supersense_role", "supersense_func"], "ud_xpos_embd_dim": 5, "num_lstm_layers": 1, "use_role": false, "update_lemmas_embd": true, "use_token_internal": false, "use_ud_dep": true, "mlp_dropout_p": 0.36, "learning_rate": 0.06309573444801933}"""),
    # 'goldid.goldsyn.goldrole': json.loads("""{"mask_mwes": false, "learning_rate_decay": 0.0031622776601683794, "mlp_activation": "cube", "use_parent": true, "govobj_config_embd_dim": 3, "is_bilstm": true, "use_token_internal": false, "mlp_dropout_p": 0.25, "ud_deps_embd_dim": 25, "lexcat_embd_dim": 3, "allow_empty_prediction": false, "use_ner": true, "embd_type": "muse", "labels_to_predict": ["supersense_func"], "use_instance_embd": false, "learning_rate": 0.06309573444801933, "lstm_h_dim": 100, "use_role": true, "epochs": 100, "mlp_layers": 2, "lstm_dropout_p": 0.5, "labels_to_learn": ["supersense_func"], "token_internal_embd_dim": 25, "use_prep_onehot": false, "use_ud_xpos": false, "ud_xpos_embd_dim": 5, "use_govobj": false, "use_func": false, "use_lexcat": true, "use_token": true, "num_lstm_layers": 1, "dynet_random_seed": null, "pss_embd_dim": 10, "ner_embd_dim": 5, "mlp_layer_dim": 200, "update_token_embd": true, "token_embd_dim": 300, "use_grandparent": true, "use_ud_dep": true, "update_lemmas_embd": true}"""),
    # 'goldid.goldsyn.goldfunc': json.loads("""{"use_role": false, "lstm_dropout_p": 0.5, "pss_embd_dim": 5, "lstm_h_dim": 40, "mlp_layers": 3, "ner_embd_dim": 10, "update_token_embd": true, "use_ner": true, "use_func": true, "use_grandparent": true, "use_parent": true, "embd_type": "muse", "update_lemmas_embd": true, "ud_deps_embd_dim": 5, "use_ud_xpos": false, "mlp_dropout_p": 0.09, "token_embd_dim": 300, "use_ud_dep": true, "ud_xpos_embd_dim": 10, "mlp_layer_dim": 200, "govobj_config_embd_dim": 3, "is_bilstm": true, "mask_mwes": false, "labels_to_learn": ["supersense_role"], "learning_rate_decay": 0.0001, "epochs": 100, "dynet_random_seed": null, "labels_to_predict": ["supersense_role"], "allow_empty_prediction": false, "token_internal_embd_dim": 300, "use_lexcat": true, "use_govobj": false, "use_prep_onehot": false, "use_token_internal": false, "use_instance_embd": false, "use_token": true, "num_lstm_layers": 1, "learning_rate": 0.15848931924611143, "lexcat_embd_dim": 3, "mlp_activation": "tanh"}"""),
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
        params['epochs'] = 100
        params['embd_type'] = 'fasttext_en'
        model = LstmMlpSupersensesModel(LstmMlpSupersensesModel.HyperParameters(**params))
        model.fit(
            samples=train_samples,
            validation_samples=dev_samples,
            test_samples=test_samples,
            show_progress=True
        )

        model.save(r'/tmp/pssmodel')
        loaded = model.load(r'/tmp/pssmodel')

        p = loaded.predict(preprocess_sentence('Hello, my name is Aviram and I am from Mishmar HaShiva').xs)
        print(p)


if __name__ == '__main__':
    run()