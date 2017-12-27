from collections import namedtuple
from pprint import pprint
from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from models.supersenses import vocabs
from models.supersenses.features.features import build_features
from ptb_poses import assert_pos
from utils import update_dict, clear_nones
import numpy as np

class LstmMlpSupersensesModel(object):

    Sample = namedtuple('Sample', ['xs', 'ys'])

    class SampleX:

        def __init__(self,
                     token,
                     ind,
                     pos=None,
                     spacy_dep=None,
                     spacy_head_ind=None,
                     spacy_ner=None,
                     ud_dep=None,
                     ud_head_ind=None):
            self.token = token
            self.ind = ind
            self.pos = pos
            self.spacy_ner = spacy_ner
            self.spacy_dep = spacy_dep
            self.spacy_head_ind = spacy_head_ind
            self.ud_dep = ud_dep
            self.ud_head_ind = ud_head_ind

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
                     deps_from, # 'spacy' or 'ud'
                     use_spacy_ner,
                     use_prep_onehot,
                     use_token_internal,
                     update_token_embd,
                     token_embd_dim,
                     token_internal_embd_dim,
                     mlp_layers,
                     mlp_layer_dim,
                     mlp_activation,
                     lstm_h_dim,
                     num_lstm_layers,
                     is_bilstm,
                     mlp_dropout_p,
                     lstm_dropout_p,
                     epochs,
                     learning_rate,
                     learning_rate_decay,
                     mask_by # MASK_BY_SAMPLE_YS or MASK_BY_POS_PREFIX + 'pos1,pos2,...'
                     ):
            self.lstm_dropout_p = lstm_dropout_p
            self.mask_by = mask_by
            self.labels_to_predict = labels_to_predict
            self.token_internal_embd_dim = token_internal_embd_dim
            self.use_token_internal = use_token_internal
            self.use_prep_onehot = use_prep_onehot
            self.learning_rate_decay = learning_rate_decay
            self.learning_rate = learning_rate
            self.mlp_activation = mlp_activation
            self.update_token_embd = update_token_embd
            self.use_token = use_token
            self.use_pos = use_pos
            self.use_dep = use_dep
            self.deps_from = deps_from
            self.use_spacy_ner = use_spacy_ner
            self.token_embd_dim = token_embd_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_dropout_p = mlp_dropout_p
            self.epochs = epochs

            assert(mask_by == LstmMlpSupersensesModel.HyperParameters.MASK_BY_SAMPLE_YS
                   or mask_by.startswith(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX))
            for pos in (self.get_pos_mask() or []):
                assert_pos(pos)
            assert(deps_from in ['spacy', 'ud'])

        def is_mask_by_sample_ys(self):
            return self.mask_by == LstmMlpSupersensesModel.HyperParameters.MASK_BY_SAMPLE_YS

        def get_pos_mask(self):
            if not self.mask_by.startswith(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX):
                return None
            opts = self.mask_by[len(LstmMlpSupersensesModel.HyperParameters.MASK_BY_POS_PREFIX):].split(',')
            return opts

        def should_use_ud_dep(self):
            return self.deps_from == 'ud'

    def __init__(self, hyperparameters):
        hp = hyperparameters
        self.hyperparameters = hp

        print("LstmMlpSupersensesModel: Building model with the following hyperparameters:")
        pprint(hp.__dict__)

        self.features = build_features(hp)

        names = lambda features: [f.name for f in features]

        self.model = LstmMlpMulticlassModel(
            input_vocabularies={feat.name: feat.vocab for feat in self.features.list_enum_features()},
            input_embeddings={feat.name: feat.embedding for feat in self.features.list_features_with_embedding(include_auto=False)},
            output_vocabulary=vocabs.PSS,
            hyperparameters=LstmMlpMulticlassModel.HyperParameters(**update_dict(hp.__dict__, {
                'lstm_input_fields': names(self.features.list_lstm_features()),
                'mlp_input_fields': names(self.features.list_mlp_features(include_refs=False)),
                'token_neighbour_types': names(self.features.list_ref_features()),
                'input_embeddings_to_update': {name: True for name in names(self.features.list_updatable_features())},
                'input_embeddings_default_dim': None,
                'input_embedding_dims': {f.name: f.dim for f in self.features.list_features_with_embedding()},
                'n_labels_to_predict': len(self.hyperparameters.labels_to_predict)
            }, del_keys=['use_token', 'use_pos', 'use_dep', 'use_spacy_ner', 'token_embd_dim', 'ner_embd_dim', 'token_internal_embd_dim',
                         'update_token_embd', 'use_prep_onehot', 'use_token_internal', 'labels_to_predict', 'mask_by', 'deps_from'])
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

    def sample_x_to_lowlevel(self, sample_x, sample_xs, x_mask):
        return LstmMlpMulticlassModel.SampleX(
            fields={
                f.name: f.extract(sample_x, sample_xs) for f in self.features.list_enum_features()
                if x_mask or not f.masked_only
            },
            neighbors={
                f.name: f.extract(sample_x, sample_xs) for f in self.features.list_ref_features()
                if x_mask or not f.masked_only
            }
        )

    def sample_y_to_lowlevel(self, sample_y):
        labels = self.hyperparameters.labels_to_predict
        ll_y = tuple([getattr(sample_y, label) for label in labels])
        return ll_y

    def lowlevel_to_sample_y(self, ll_sample_y):
        labels = self.hyperparameters.labels_to_predict
        return LstmMlpSupersensesModel.SampleY(**{label: ll_sample_y[ind] for ind, label in enumerate(labels)})

    def apply_mask(self, sample_x, sample_y):
        if self.hyperparameters.is_mask_by_sample_ys():
            return sample_y is not None
        else:
            return sample_x.pos in self.hyperparameters.get_pos_mask()

    def get_sample_mask(self, sample_xs, sample_ys=None):
        return [self.apply_mask(x, None if sample_ys is None else sample_ys[ind]) for ind, x in enumerate(sample_xs)]

    def sample_to_lowlevel(self, sample):
        mask = self.get_sample_mask(sample.xs, sample.ys)
        return LstmMlpMulticlassModel.Sample(
            xs=[self.sample_x_to_lowlevel(x, sample.xs, x_mask) for x_mask, x in zip(mask, sample.xs)],
            ys=[self.sample_y_to_lowlevel(y) for y in sample.ys],
            mask=mask
        )

    def report_masking(self, samples, name='samples'):
        n_sentences = 0
        n_y_labels = 0
        for s in samples:
            mask = self.get_sample_mask(s.xs, s.ys)
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
        if not mask:
            mask = [True] * len(sample_xs)
        ll_xs = [self.sample_x_to_lowlevel(x, sample_xs, x_mask) for x_mask, x in zip(mask, sample_xs)]
        ll_ys = self.model.predict(ll_xs, mask=mask)
        ys = tuple([self.lowlevel_to_sample_y(ll_s) for ll_s in ll_ys])
        return ys

    @property
    def test_set_evaluation(self):
        return self.model.test_set_evaluation

    @property
    def train_set_evaluation(self):
        return self.model.train_set_evaluation
