from evaluators.classifier_evaluator import ClassifierEvaluator
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.tuner_domains import PS
from vocabulary import Vocabulary
import supersenses
import json

evaluator = ClassifierEvaluator()

def streusle_record_to_lstm_model_sample(record):
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
                token=tagged_token.token,
                pos=tagged_token.pos,
                spacy_dep=tagged_token.spacy_dep,
                spacy_head_ind=tagged_token.spacy_head_ind,
                spacy_ner=tagged_token.spacy_ner,
                ud_dep=tagged_token.ud_dep,
                ud_head_ind=tagged_token.ud_head_ind,
            ) for tagged_token in record.tagged_tokens
        ],
        ys=[LstmMlpSupersensesModel.SampleY(
                supersense_role=tagged_token.supersense_role,
                supersense_func=tagged_token.supersense_func
            ) for tagged_token in record.tagged_tokens
        ],
    )

def run(train_records, dev_records, test_records, streusle_loader):
    train_samples = [streusle_record_to_lstm_model_sample(r) for r in train_records]
    dev_samples = [streusle_record_to_lstm_model_sample(r) for r in dev_records]
    test_samples = [streusle_record_to_lstm_model_sample(r) for r in test_records]

    pp_vocab = Vocabulary('Prepositions')
    pp_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys) if any([y.supersense_role, y.supersense_func])]))

    spacy_dep_vocab = Vocabulary('Spacy Dependencies')
    spacy_dep_vocab.add_words(set([x.spacy_dep for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    ud_dep_vocab = Vocabulary('UD Dependencies')
    ud_dep_vocab.add_words(set([x.ud_dep for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    pos_vocab = Vocabulary('POS')
    pos_vocab.add_words(set([x.pos for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    ner_vocab = Vocabulary('NER')
    ner_vocab.add_words(set([x.spacy_ner for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    token_vocab = Vocabulary('Tokens')
    token_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

    pss_vocab = Vocabulary('PrepositionalSupersenses')
    pss_vocab.add_words(supersenses.PREPOSITION_SUPERSENSES_SET)
    pss_vocab.add_word(None)

    # dump_vocabularies([pp_vocab, spacy_dep_vocab, ud_dep_vocab, pos_vocab, ner_vocab, token_vocab, pss_vocab])

    tuner = LstmMlpSupersensesModelHyperparametersTuner(
        results_csv_path='/cs/labs/oabend/aviramstern/results.csv',
        samples=train_samples,
        validation_samples=dev_samples,
        show_progress=True,
        show_epoch_eval=True,
        tuner_domains_override=[
            PS(name='labels_to_predict', values=[
                ('supersense_role', 'supersense_func'),
            ]),
            PS(name='mask_by', values=['pos:IN,PRP$,RB,TO']),
            # PS(name='mask_by', values=['sample-ys']),
            # PS(name='learning_rate', values=[0.1]),
            # PS(name='learning_rate_decay', values=[0.01]),
            # PS(name='mlp_dropout_p', values=[0.1])
            # PS(name='epochs', values=[5])
        ],

        token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
        token_vocab=token_vocab,
        token_onehot_vocab=pp_vocab,
        pos_vocab=pos_vocab,
        spacy_dep_vocab=spacy_dep_vocab,
        ud_dep_vocab=ud_dep_vocab,
        ner_vocab=ner_vocab,
        supersense_vocab=pss_vocab,
    )

    # tuner.tune(n_executions=1)
    tuner.sample_execution(json.loads(
        """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_role", "supersense_func"], "use_token_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "update_pos_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "use_head": true, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "pos_embd_dim": 98, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_ner": false, "deps_from": "ud", "ner_embd_dim": 30, "lstm_dropout_p": 0.1, "use_dep": true}"""
        # """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_role"], "use_token_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "update_pos_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "use_head": true, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "pos_embd_dim": 98, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_ner": false, "deps_from": "ud", "ner_embd_dim": 30, "lstm_dropout_p": 0.1, "use_dep": true}"""
        # """{"use_token_internal": true, "learning_rate_decay": 0.00031622776601683794, "num_lstm_layers": 2, "labels_to_predict": ["supersense_func"], "use_token_onehot": true, "mlp_dropout_p": 0.12, "epochs": 40, "mlp_activation": "relu", "use_token": true, "update_token_embd": false, "update_pos_embd": false, "mlp_layer_dim": 77, "is_bilstm": true, "token_internal_embd_dim": 33, "token_embd_dim": 300, "use_head": true, "learning_rate": 0.31622776601683794, "mlp_layers": 2, "pos_embd_dim": 98, "lstm_h_dim": 64, "use_pos": false, "mask_by": "pos:IN,PRP$,RB,TO", "use_ner": false, "deps_from": "ud", "ner_embd_dim": 30, "lstm_dropout_p": 0.1, "use_dep": true}"""
    ))

    # print('LSTM-MLP evaluation:')
    # lstm_mlp_model = LstmMlpSupersensesModel(
    #     token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
    #     token_vocab=token_vocab,
    #     token_onehot_vocab=pp_vocab,
    #     pos_vocab=pos_vocab,
    #     dep_vocab=dep_vocab,
    #     supersense_vocab=pss_vocab,
    #     hyperparameters=LstmMlpSupersensesModel.HyperParameters(**json.loads(
    #         """ {"use_token_internal": false, "token_internal_embd_dim": 73, "mlp_layers": 2, "epochs": 100, "is_bilstm": true, "labels_to_predict": ["supersense_func"], "use_head": false, "mlp_activation": "relu", "learning_rate_decay": 0.01, "learning_rate": 0.1, "num_lstm_layers": 2, "mlp_dropout_p": 0.12, "pos_embd_dim": 50, "use_token": true, "use_token_onehot": true, "use_pos": false, "update_token_embd": true, "update_pos_embd": false, "mlp_layer_dim": 98, "token_embd_dim": 300, "use_ner": false, "deps_from": "ud", "ner_embd_dim": 30, "lstm_dropout_p": 0.1, "use_dep": true, "lstm_h_dim": 100, "mask_by": "pos:IN,PRP$,RB,TO"}"""
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
