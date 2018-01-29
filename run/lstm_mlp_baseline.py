import os
from evaluators.classifier_evaluator import ClassifierEvaluator
from models.supersenses.settings import GOLD_ID_GOLD_PREP, GOLD_ID_AUTO_PREP, AUTO_ID_AUTO_PREP, AUTO_ID_GOLD_PREP
from models.supersenses.features.features_test import test_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
from models.supersenses.tuner_domains import PS
from run.dump_vocabs import dump_vocabularies
from vocabulary import Vocabulary
import supersenses
import json
from hyperparameters_tuner import union_settings, override_settings
evaluator = ClassifierEvaluator()

def run(train_records, dev_records, test_records, streusle_loader):
    train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    # pp_vocab = Vocabulary('PREPS')
    # pp_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys) if any([y.supersense_role, y.supersense_func])]))
    #
    # spacy_dep_vocab = Vocabulary('SPACY_DEPS')
    # spacy_dep_vocab.add_words(set([x.spacy_dep for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))
    # spacy_dep_vocab.add_word(None)
    #
    # ud_dep_vocab = Vocabulary('UD_DEPS')
    # ud_dep_vocab.add_words(set([x.ud_dep for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))
    # ud_dep_vocab.add_word(None)
    #
    # ud_pos_vocab = Vocabulary('UD_POS')
    # ud_pos_vocab.add_words(set([x.ud_pos for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))
    # ud_pos_vocab.add_word(None)
    #
    # spacy_pos_vocab = Vocabulary('SPACY_POS')
    # spacy_pos_vocab.add_words(set([x.spacy_pos for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))
    # spacy_pos_vocab.add_word(None)
    #
    # spacy_ner_vocab = Vocabulary('SPACY_NER')
    # spacy_ner_vocab.add_words(set([x.spacy_ner for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))
    # spacy_ner_vocab.add_word(None)
    #
    # token_vocab = Vocabulary('TOKENS')
    # token_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    # ud_lemmas_vocab = Vocabulary('UD_LEMMAS')
    # ud_lemmas_vocab.add_words(set([x.ud_lemma for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    # pss_vocab = Vocabulary('PSS')
    # pss_vocab.add_words(supersenses.PREPOSITION_SUPERSENSES_SET)
    # pss_vocab.add_word(None)
    #
    # spacy_lemmas_vocab = Vocabulary('SPACY_LEMMAS')
    # spacy_lemmas_vocab.add_words(set([x.spacy_lemma for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    # dump_vocabularies([pp_vocab, spacy_dep_vocab, ud_dep_vocab, ud_pos_vocab, ner_vocab, token_vocab, pss_vocab, spacy_pos_vocab])
    # dump_vocabularies([spacy_ner_vocab])
    # dump_vocabularies([spacy_lemmas_vocab])
    # dump_vocabularies([ud_lemmas_vocab])

    test_features()

    tuner = LstmMlpSupersensesModelHyperparametersTuner(
        results_csv_path=os.environ.get('RESULTS_PATH') or '/cs/labs/oabend/aviramstern/results.csv',
        samples=train_samples, # use all after testing
        validation_samples=dev_samples,
        show_progress=True,
        show_epoch_eval=True,
        tuner_domains=override_settings([
            union_settings([
                # GOLD_ID_GOLD_PREP,
                # GOLD_ID_AUTO_PREP
                # AUTO_ID_AUTO_PREP
                AUTO_ID_GOLD_PREP
            ]),
            [PS(name='epochs', values=[1])] # remove after testing
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
    # evaluator = ClassifierEvaluator(predictor=predictor.model)
    # ll_samples = [predictor.sample_to_lowlevel(x) for x in dev_samples]
    # evaluator.evaluate(ll_samples, examples_to_show=5)
    #
    # print('Predictor: loaded')
    #
    # evaluator = ClassifierEvaluator(predictor=loaded_predictor.model)
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
    # # evaluator = ClassifierEvaluator(predictor=lstm_mlp_model.model)
    # # ll_samples = [LstmMlpSupersensesModel.sample_to_lowlevel(x) for x in test_samples]
    # # evaluator.evaluate(ll_samples, examples_to_show=5)
