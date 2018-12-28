import json

from datasets.streusle_v4 import StreusleLoader
from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.preprocessing import preprocess_sentence
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample

evaluator = PSSClasifierEvaluator()

best_params_per_task = {
    'goldid.goldsyn': json.loads("""{"use_token_internal": false, "use_lexcat": true, "use_ud_dep": true, "learning_rate": 0.06309573444801933, "is_bilstm": true, "use_parent": true, "prep_dropout_p": 0.5, "update_token_embd": false, "num_lstm_layers": 1, "use_prep_onehot": false, "use_grandparent": true, "update_lemmas_embd": true, "use_role": false, "use_ud_xpos": true, "lexcat_embd_dim": 3, "mlp_dropout_p": 0.11, "epochs": 100, "mlp_layer_dim": 80, "ud_xpos_embd_dim": 10, "use_func": false, "lstm_h_dim": 40, "mask_mwes": false, "use_instance_embd": false, "mlp_activation": "tanh", "allow_empty_prediction": false, "use_token": true, "govobj_config_embd_dim": 3, "use_prep": true, "embd_type": "muse", "labels_to_learn": ["supersense_role", "supersense_func"], "labels_to_predict": ["supersense_role", "supersense_func"], "token_embd_dim": 300, "mlp_layers": 2, "dynet_random_seed": null, "pss_embd_dim": 5, "use_govobj": false, "ner_embd_dim": 10, "lstm_dropout_p": 0.5, "token_internal_embd_dim": 25, "ud_deps_embd_dim": 10, "learning_rate_decay": 0.00031622776601683794, "use_ner": true}"""),
    'goldid.goldsyn.goldrole': json.loads("""{"use_role": true, "use_lexcat": true, "use_govobj": false, "lstm_dropout_p": 0.02, "update_token_embd": false, "prep_dropout_p": 0, "pss_embd_dim": 10, "govobj_config_embd_dim": 3, "use_prep_onehot": false, "lstm_h_dim": 20, "labels_to_learn": ["supersense_func"], "use_token": true, "learning_rate_decay": 0.001, "lexcat_embd_dim": 3, "use_grandparent": true, "num_lstm_layers": 1, "mlp_dropout_p": 0.09, "use_prep": true, "mlp_layers": 3, "use_parent": true, "allow_empty_prediction": false, "mask_mwes": false, "learning_rate": 0.01, "ud_deps_embd_dim": 10, "dynet_random_seed": null, "ud_xpos_embd_dim": 25, "mlp_activation": "tanh", "use_token_internal": false, "use_instance_embd": false, "ner_embd_dim": 10, "mlp_layer_dim": 100, "use_ud_dep": true, "update_lemmas_embd": false, "is_bilstm": true, "use_ud_xpos": true, "use_ner": true, "epochs": 100, "embd_type": "muse", "token_embd_dim": 300, "use_func": false, "token_internal_embd_dim": 10, "labels_to_predict": ["supersense_func"]}"""),
    'goldid.goldsyn.goldfunc': json.loads("""{{"use_token": true, "ud_deps_embd_dim": 25, "lexcat_embd_dim": 3, "labels_to_predict": ["supersense_role"], "learning_rate_decay": 0.00031622776601683794, "use_role": false, "use_govobj": false, "epochs": 100, "learning_rate": 0.15848931924611143, "use_ud_xpos": true, "use_ud_dep": true, "ud_xpos_embd_dim": 10, "use_prep_onehot": false, "lstm_h_dim": 200, "mlp_activation": "relu", "mask_mwes": false, "embd_type": "muse", "use_ner": true, "token_internal_embd_dim": 25, "ner_embd_dim": 5, "pss_embd_dim": 10, "use_token_internal": false, "use_lexcat": true, "govobj_config_embd_dim": 3, "num_lstm_layers": 1, "use_grandparent": true, "prep_dropout_p": 0.3, "update_lemmas_embd": false, "mlp_layers": 2, "is_bilstm": true, "lstm_dropout_p": 0.13, "labels_to_learn": ["supersense_role"], "allow_empty_prediction": false, "mlp_dropout_p": 0.22, "use_instance_embd": false, "token_embd_dim": 300, "mlp_layer_dim": 200, "dynet_random_seed": null, "use_parent": true, "use_func": true, "use_prep": true, "update_token_embd": true}"""),
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