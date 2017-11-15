from collections import namedtuple

import random
import dynet as dy
import numpy as np

from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from utils import update_dict


class LstmMlpSupersensesModel(object):

    Sample = namedtuple('Sample', ['xs', 'ys'])

    class SampleX:
        def __init__(self,
                     token,
                     pos=None,
                     dep=None,
                     head_ind=None):
            self.token = token
            self.pos = pos
            self.dep = dep
            self.head_ind = head_ind

    class SampleY:
        def __init__(self, supersense):
            self.supersense = supersense

    class HyperParameters:
        def __init__(self,
                     use_token,
                     use_pos,
                     use_dep,
                     update_token_embd,
                     update_pos_embd,
                     update_dep_embd,
                     token_embd_dim,
                     pos_embd_dim,
                     dep_embd_dim,
                     mlp_layers,
                     mlp_layer_dim,
                     mlp_activation,
                     lstm_h_dim,
                     num_lstm_layers,
                     is_bilstm,
                     use_head,
                     mlp_dropout_p,
                     epochs,
                     validation_split,
                     learning_rate
                     ):
            self.learning_rate = learning_rate
            self.mlp_activation = mlp_activation
            self.update_token_embd = update_token_embd
            self.update_pos_embd = update_pos_embd
            self.update_dep_embd = update_dep_embd
            self.use_token = use_token
            self.use_pos = use_pos
            self.use_dep = use_dep
            self.token_embd_dim = token_embd_dim
            self.pos_embd_dim = pos_embd_dim
            self.dep_embd_dim = dep_embd_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.use_head = use_head
            self.mlp_dropout_p = mlp_dropout_p
            self.epochs = epochs
            self.validation_split = validation_split


    def __init__(self,
                 token_vocab=None,
                 pos_vocab=None,
                 dep_vocab=None,
                 ss_vocab=None,
                 supersense_vocab=None,
                 token_embd=None,
                 pos_embd=None,
                 dep_embd=None,
                 hyperparameters=None):
        hp = hyperparameters or LstmMlpSupersensesModel.HyperParameters(
            use_token=True,
            use_pos=True,
            use_dep=True,
            token_embd_dim=10,
            pos_embd_dim=10,
            dep_embd_dim=10,
            mlp_layers=2,
            mlp_layer_dim=10,
            lstm_h_dim=40,
            num_lstm_layers=2,
            is_bilstm=True,
            use_head=True,
            mlp_dropout_p=0,
            epochs=30,
            validation_split=0.3
        )

        self.token_vocab = token_vocab
        self.pos_vocab = pos_vocab
        self.dep_vocab = dep_vocab
        self.supersense_vocab = supersense_vocab
        self.token_embd = token_embd
        self.pos_embd = pos_embd
        self.dep_embd = dep_embd
        self.hyperparameters = hp

        self.model = LstmMlpMulticlassModel(
            input_vocabularies={
                'token': token_vocab,
                'pos': pos_vocab,
                'dep': dep_vocab
            },
            input_embeddings={
                'token': token_embd,
                'pos': pos_embd,
                'dep': dep_embd,
            },
            output_vocabulary=ss_vocab,
            hyperparameters=LstmMlpMulticlassModel.HyperParameters(**update_dict(hp.__dict__, {
                'input_fields': list(filter(lambda x: x, [
                    self.hyperparameters.use_token and "token",
                    self.hyperparameters.use_pos and "pos",
                    self.hyperparameters.use_dep and "dep"
                ])),
                'input_embeddings_to_update': {
                    'token': hp.update_token_embd,
                    'pos': hp.update_pos_embd,
                    'dep': hp.update_dep_embd
                },
                'input_embeddings_default_dim': None,
                'input_embedding_dims': {
                    'token': hp.token_embd_dim,
                    'pos': hp.pos_embd_dim,
                    'dep': hp.dep_embd_dim
                }
            }, del_keys=['use_token', 'use_pos', 'use_dep', 'token_embd_dim', 'pos_embd_dim', 'dep_embd_dim',
                         'update_token_embd', 'update_pos_embd', 'update_dep_embd']))
        )

    def _sample_x_to_lowlevel(self, sample_x):
        return LstmMlpMulticlassModel.SampleX(
            fields={
                'token': sample_x.token,
                'pos': sample_x.pos,
                'dep': sample_x.dep
            },
            head_ind=sample_x.head_ind
        )

    def _sample_y_to_lowlevel(self, sample_y):
        return sample_y.supersense

    def _lowlevel_to_sample_y(self, ll_sample_y):
        return LstmMlpSupersensesModel.SampleY(supersense=ll_sample_y)

    def _sample_to_lowlevel(self, sample):
        return LstmMlpMulticlassModel.Sample(
            xs=[self._sample_x_to_lowlevel(x) for x in sample.xs],
            ys=[self._sample_y_to_lowlevel(y) for y in sample.ys],
        )

    def fit(self, samples, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        ll_samples = [self._sample_to_lowlevel(s) for s in samples]
        self.model.fit(ll_samples, show_progress, show_epoch_eval, evaluator)
        return self

    def predict(self, sample_xs, mask=None):
        ll_ys = self.model.predict(sample_xs, mask=mask)
        ys = [self._lowlevel_to_sample_y(ll_s) for ll_s in ll_ys]
        return ys

    @property
    def test_set_evaluation(self):
        return self.model.test_set_evaluation

    @property
    def train_set_evaluation(self):
        return self.model.train_set_evaluation