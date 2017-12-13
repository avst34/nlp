from collections import namedtuple
from pprint import pprint
from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from ptb_poses import assert_pos
from utils import update_dict, clear_nones
import numpy as np

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
        def __init__(self, supersense_role=None, supersense_func=None):
            self.supersense_role = supersense_role
            self.supersense_func = supersense_func

    SUPERSENSE_ROLE = "supersense_role"
    SUPERSENSE_FUNC = "supersense_func"

    class HyperParameters:

        MASK_BY_SAMPLE_YS = 'sample-ys'
        MASK_BY_POS_PREFIX = 'pos:'

        def __init__(self,
                     labels_to_predict,
                     use_token,
                     use_pos,
                     use_dep,
                     use_token_onehot,
                     use_token_internal,
                     update_token_embd,
                     update_pos_embd,
                     token_embd_dim,
                     token_internal_embd_dim,
                     pos_embd_dim,
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
                     learning_rate,
                     learning_rate_decay,
                     mask_by # MASK_BY_SAMPLE_YS or MASK_BY_POS_PREFIX + 'pos1,pos2,...'
                 ):
            self.mask_by = mask_by
            self.labels_to_predict = labels_to_predict
            self.token_internal_embd_dim = token_internal_embd_dim
            self.use_token_internal = use_token_internal
            self.use_token_onehot = use_token_onehot
            self.learning_rate_decay = learning_rate_decay
            self.learning_rate = learning_rate
            self.mlp_activation = mlp_activation
            self.update_token_embd = update_token_embd
            self.update_pos_embd = update_pos_embd
            self.use_token = use_token
            self.use_pos = use_pos
            self.use_dep = use_dep
            self.token_embd_dim = token_embd_dim
            self.pos_embd_dim = pos_embd_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.use_head = use_head
            self.mlp_dropout_p = mlp_dropout_p
            self.epochs = epochs
            self.validation_split = validation_split
            assert(mask_by == LstmMlpSupersensesModel.HyperParameters.MASK_BY_SAMPLE_YS
                   or mask_by.startswith(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX))

            for pos in (self.get_pos_mask() or []):
                assert_pos(pos)

        def is_mask_by_sample_ys(self):
            return self.mask_by == LstmMlpSupersensesModel.HyperParameters.MASK_BY_SAMPLE_YS

        def get_pos_mask(self):
            if not self.mask_by.startswith(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX):
                return None
            opts = self.mask_by[len(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX):].split(',')
            return opts

    def __init__(self,
                 token_vocab,
                 pos_vocab,
                 dep_vocab,
                 token_onehot_vocab,
                 supersense_vocab,
                 token_embd=None,
                 pos_embd=None,
                 hyperparameters=None):
        hp = hyperparameters
        self.token_vocab = token_vocab
        self.pos_vocab = pos_vocab
        self.dep_vocab = dep_vocab
        self.token_onehot_vocab = token_onehot_vocab
        self.supersense_vocab = supersense_vocab
        self.token_embd = token_embd
        self.pos_embd = pos_embd
        self.hyperparameters = hp

        print("LstmMlpSupersensesModel: Building model with the following hyperparameters:")
        pprint(hp.__dict__)

        self.model = LstmMlpMulticlassModel(
            input_vocabularies=clear_nones({
                'token': token_vocab,
                'token_internal': token_vocab,
                'pos': pos_vocab,
                'dep': dep_vocab,
                'token_onehot': token_onehot_vocab
            }),
            input_embeddings=clear_nones({
                'token': token_embd,
                'pos': pos_embd,
                'dep': self._build_vocab_onehot_embd(self.dep_vocab),
                'token_onehot': self._build_vocab_onehot_embd(self.token_onehot_vocab)
            }),
            output_vocabulary=supersense_vocab,
            hyperparameters=LstmMlpMulticlassModel.HyperParameters(**update_dict(hp.__dict__, {
                'lstm_input_fields': list(filter(lambda x: x, [
                    self.hyperparameters.use_token and "token",
                    self.hyperparameters.use_token_internal and "token_internal",
                    self.hyperparameters.use_pos and "pos"
                ])),
                'mlp_input_fields': list(filter(lambda x: x, [
                    self.hyperparameters.use_dep and "dep",
                    self.hyperparameters.use_token_onehot and "token_onehot"
                ])),
                'input_embeddings_to_update': {
                    'token': hp.update_token_embd,
                    'pos': hp.update_pos_embd,
                    'token_onehot': False
                },
                'input_embeddings_default_dim': None,
                'input_embedding_dims': {
                    'token': hp.token_embd_dim,
                    'pos': hp.pos_embd_dim,
                    'dep': self.dep_vocab.size() if self.dep_vocab else 0,
                    'token_onehot': self.token_onehot_vocab.size() if self.token_onehot_vocab else 0,
                    'token_internal': hp.token_internal_embd_dim
                },
                'n_labels_to_predict': len(self.hyperparameters.labels_to_predict)
            }, del_keys=['use_token', 'use_pos', 'use_dep', 'token_embd_dim', 'pos_embd_dim', 'token_internal_embd_dim',
                         'update_token_embd', 'update_pos_embd', 'use_token_onehot', 'use_token_internal', 'labels_to_predict', 'mask_by'])
           )
        )

    def _build_vocab_onehot_embd(self, vocab):
        n_words = vocab.size()
        embeddings = {}
        for word in vocab.all_words():
            word_ind = vocab.get_index(word)
            vec = [0] * n_words
            vec[word_ind] = 1
            embeddings[word] = vec
        return embeddings

    @staticmethod
    def sample_x_to_lowlevel(sample_x):
        return LstmMlpMulticlassModel.SampleX(
            fields={
                'token': sample_x.token,
                'token_onehot': sample_x.token,
                'token_internal': sample_x.token,
                'pos': sample_x.pos,
                'dep': sample_x.dep
            },
            head_ind=sample_x.head_ind
        )

    def sample_y_to_lowlevel(self, sample_y):
        labels = sorted(self.hyperparameters.labels_to_predict)
        ll_y = tuple([getattr(sample_y, label) for label in labels])
        return ll_y

    def lowlevel_to_sample_y(self, ll_sample_y):
        labels = sorted(self.hyperparameters.labels_to_predict)
        return LstmMlpSupersensesModel.SampleY(**{label: ll_sample_y[ind] for ind, label in enumerate(labels)})

    def apply_mask(self, sample_x, sample_y):
        if self.hyperparameters.is_mask_by_sample_ys():
            return sample_y is not None
        else:
            return sample_x.pos in self.hyperparameters.get_pos_mask()

    def get_sample_mask(self, sample):
        return [self.apply_mask(x, y) for x, y in zip(sample.xs, sample.ys)]

    def sample_to_lowlevel(self, sample):
        return LstmMlpMulticlassModel.Sample(
            xs=[self.sample_x_to_lowlevel(x) for x in sample.xs],
            ys=[self.sample_y_to_lowlevel(y) for y in sample.ys],
            mask=self.get_sample_mask(sample)
        )

    def report_masking(self, samples, name='samples'):
        n_sentences = 0
        n_y_labels = 0
        for s in samples:
            mask = self.get_sample_mask(s)
            if any(mask):
                n_sentences += 1
            n_y_labels += len([y for i, y in enumerate(s.ys) if any([y.supersense_func, y.supersense_role]) and mask[i]])
        print("[%s]: %d sentences, %d labels" % (name, n_sentences, n_y_labels))

    def fit(self, samples, validation_samples=None, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        ll_samples = [self.sample_to_lowlevel(s) for s in samples]
        ll_samples = [x for x in ll_samples if any(x.mask)]

        ll_validation_samples = [self.sample_to_lowlevel(s) for s in validation_samples] if validation_samples else None
        ll_validation_samples = [x for x in ll_validation_samples if any(x.mask)] if ll_validation_samples else None

        self.report_masking(samples, 'Training')
        if validation_samples:
            self.report_masking(validation_samples, 'Validation')

        self.model.fit(ll_samples, show_progress=show_progress,
                       show_epoch_eval=show_epoch_eval, evaluator=evaluator,
                       validation_samples=ll_validation_samples)
        return self

    def predict(self, sample_xs, mask=None):
        ll_xs = [self.sample_x_to_lowlevel(x) for x in sample_xs]
        ll_ys = self.model.predict(ll_xs, mask=mask)
        ys = tuple([self.lowlevel_to_sample_y(ll_s) for ll_s in ll_ys])
        return ys

    @property
    def test_set_evaluation(self):
        return self.model.test_set_evaluation

    @property
    def train_set_evaluation(self):
        return self.model.train_set_evaluation