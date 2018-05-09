import os
import random

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from models.supersenses.boknilev_integration import boknilev_record_to_lstm_model_sample_xs
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel



def run():

    train_records, test_records = load_boknilev()
    train_samples_xs, test_samples_xs = [boknilev_record_to_lstm_model_sample_xs(r) for r in train_records], [boknilev_record_to_lstm_model_sample_xs(r) for r in test_records]

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

    predictor = LstmMlpSupersensesModel.load('/cs/labs/oabend/aviramstern/best_results/nn/goldid.autosyn/model')
    for xs in train_samples_xs[:100]:
        ys = predictor.predict(xs)
        print(zip(xs, ys))

if __name__ == '__main__':
    run()