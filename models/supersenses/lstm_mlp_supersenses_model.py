import json
from collections import namedtuple
from itertools import chain
from pprint import pprint
from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from models.supersenses import vocabs
from models.supersenses.features.features import build_features
from ptb_poses import assert_pos
from utils import update_dict, clear_nones
import numpy as np

class LstmMlpSupersensesModel:

    class SampleX:

        def __init__(self,
                     token,
                     ind,
                     is_part_of_mwe,
                     ud_upos,
                     ud_xpos,
                     ner,
                     ud_dep,
                     ud_head_ind,
                     lemma,
                     gov_ind, obj_ind, govobj_config,
                     identified_for_pss,
                     lexcat,
                     lemma_word2vec=None,
                     token_word2vec=None):
            self.token_word2vec = token_word2vec
            self.lemma_word2vec = lemma_word2vec
            self.lexcat = lexcat
            self.govobj_config = govobj_config
            self.obj_ind = obj_ind
            self.gov_ind = gov_ind
            self.lemma = lemma
            self.ner = ner
            self.ud_xpos = ud_xpos
            self.ud_upos = ud_upos
            self.token = token
            self.ind = ind
            self.is_part_of_mwe = is_part_of_mwe
            self.ud_dep = ud_dep
            self.ud_head_ind = ud_head_ind
            self.identified_for_pss = identified_for_pss

        def to_dict(self):
            return self.__dict__

        @staticmethod
        def from_dict(d):
            return LstmMlpSupersensesModel.SampleX(**d)

        def __repr__(self):
            return self.token

    class SampleY:

        def __init__(self, supersense_role=None, supersense_func=None):
            self.supersense_role = supersense_role
            self.supersense_func = supersense_func
            assert supersense_role and supersense_func or (not supersense_role and not supersense_func)

        def to_dict(self):
            return self.__dict__

        def is_empty(self):
            return not self.supersense_func and not self.supersense_role

        @staticmethod
        def from_dict(d):
            return LstmMlpSupersensesModel.SampleY(**d)

    class Sample:

        def __init__(self, xs, ys, sample_id):
            self.xs = xs
            self.ys = ys
            self.sample_id = sample_id

        def to_dict(self):
            return {
                'xs': [x.to_dict() for x in self.xs],
                'ys': [y.to_dict() for y in self.ys],
                'sample_id': self.sample_id
            }

        @staticmethod
        def from_dict(d):
            return LstmMlpSupersensesModel.Sample(
                xs=[LstmMlpSupersensesModel.SampleX.from_dict(x) for x in d['xs']],
                ys=[LstmMlpSupersensesModel.SampleY.from_dict(y) for y in d['ys']],
                sample_id=d['sample_id']
            )


    SUPERSENSE_ROLE = "supersense_role"
    SUPERSENSE_FUNC = "supersense_func"

    class HyperParameters:

        def __init__(self,
                     labels_to_predict,
                     use_token,
                     update_token_embd,
                     use_ud_xpos,
                     use_ud_dep,
                     use_ner,
                     use_prep_onehot,
                     use_govobj,
                     use_token_internal,
                     use_lexcat,
                     update_lemmas_embd,
                     token_embd_dim,
                     token_internal_embd_dim,
                     ud_xpos_embd_dim,
                     ud_deps_embd_dim,
                     ner_embd_dim,
                     govobj_config_embd_dim,
                     lexcat_embd_dim,
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
                     mask_mwes,
                     allow_empty_prediction,
                     dynet_random_seed
                     ):
            self.lexcat_embd_dim = lexcat_embd_dim
            self.use_lexcat = use_lexcat
            self.dynet_random_seed = dynet_random_seed
            self.use_govobj = use_govobj
            self.govobj_config_embd_dim = govobj_config_embd_dim
            self.allow_empty_prediction = allow_empty_prediction
            self.use_ner = use_ner
            self.update_lemmas_embd = update_lemmas_embd
            self.ner_embd_dim = ner_embd_dim
            self.ud_deps_embd_dim = ud_deps_embd_dim
            self.ud_xpos_embd_dim = ud_xpos_embd_dim
            self.lstm_dropout_p = lstm_dropout_p
            self.labels_to_predict = labels_to_predict
            self.token_internal_embd_dim = token_internal_embd_dim
            self.use_token_internal = use_token_internal
            self.use_prep_onehot = use_prep_onehot
            self.learning_rate_decay = learning_rate_decay
            self.learning_rate = learning_rate
            self.mlp_activation = mlp_activation
            self.update_token_embd = update_token_embd
            self.use_token = use_token
            self.use_ud_xpos = use_ud_xpos
            self.use_ud_dep = use_ud_dep
            self.use_ner = use_ner
            self.token_embd_dim = token_embd_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_dropout_p = mlp_dropout_p
            self.epochs = epochs
            self.mask_mwes = mask_mwes

        def clone(self, override=None):
            override = override or {}
            params = self.__dict__
            params.update(override)
            return LstmMlpSupersensesModel.HyperParameters(**params)

    def __init__(self, hyperparameters):
        hp = hyperparameters
        self.hyperparameters = hp

        print("LstmMlpSupersensesModel: Building model with the following hyperparameters:")
        pprint(hp.__dict__)

        self.features = build_features(hp)

        names = lambda features: [f.name for f in features]

        self.model = LstmMlpMulticlassModel(
            input_vocabularies={feat.name: feat.vocab for feat in chain(self.features.list_enum_features(), self.features.list_string_features())},
            input_embeddings={feat.name: feat.embedding for feat in self.features.list_features_with_embedding(include_auto=False)},
            output_vocabulary=vocabs.PSS if self.hyperparameters.allow_empty_prediction else vocabs.PSS_WITHOUT_NONE,
            hyperparameters=LstmMlpMulticlassModel.HyperParameters(**update_dict(hp.__dict__, {
                    'lstm_input_fields': names(self.features.list_lstm_features()),
                    'mlp_input_fields': names(self.features.list_mlp_features(include_refs=False)),
                    'token_neighbour_types': names(self.features.list_ref_features()),
                    'input_embeddings_to_allow_partial': names(self.features.list_default_zero_vec_features()),
                    'input_embeddings_to_update': {name: True for name in names(self.features.list_updatable_features())},
                    'input_embeddings_default_dim': None,
                    'input_embedding_dims': {f.name: f.dim for f in self.features.list_features_with_embedding()},
                    'n_labels_to_predict': len(self.hyperparameters.labels_to_predict)
             },
             del_keys=['use_token', 'lemmas_from', 'update_lemmas_embd', 'use_ud_xpos', 'use_govobj', 'use_ud_dep', 'use_ner', 'use_lexcat', 'token_embd_dim', 'ner_embd_dim', 'token_internal_embd_dim',
                       'ud_xpos_embd_dim', 'ud_deps_embd_dim', 'spacy_ner_embd_dim', 'govobj_config_embd_dim', 'lexcat_embd_dim',
                       'update_token_embd', 'use_prep_onehot', 'use_token_internal', 'labels_to_predict', 'mask_by', 'mask_mwes', 'allow_empty_prediction']))
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
                f.name: f.extract(sample_x, sample_xs) for f in chain(self.features.list_enum_features(), self.features.list_string_features())
                if x_mask or not f.masked_only
            },
            neighbors={
                f.name: f.extract(sample_x, sample_xs) for f in self.features.list_ref_features()
                if x_mask or not f.masked_only
            },
            embeddings_override={
                f.name: f.embedding_fallback(sample_x) for f in self.features.list_features_with_embedding_fallback()
            }
        )

    def sample_y_to_lowlevel(self, sample_y):
        labels = self.hyperparameters.labels_to_predict
        ll_y = tuple([getattr(sample_y, label) for label in labels])
        return ll_y

    def lowlevel_to_sample_y(self, ll_sample_y):
        labels = self.hyperparameters.labels_to_predict
        return LstmMlpSupersensesModel.SampleY(**{label: ll_sample_y[ind] for ind, label in enumerate(labels)})

    def apply_mask(self, sample_x):
        return sample_x.identified_for_pss
        # if self.hyperparameters.mask_mwes and sample_x.is_part_of_mwe:
        #     return False
        # if self.hyperparameters.is_mask_by_sample_ys():
        #     return not sample_y.is_empty()
        # elif self.hyperparameters.is_mask_by_auto_id():
        #     return sample_x.autoid_markable
        # else:
        #     return self.get_sample_x_pos(sample_x) in self.hyperparameters.get_pos_mask()

    def get_sample_x_pos(self, sample_x):
        if self.hyperparameters.pos_from == 'ud':
            pos = sample_x.ud_pos
        else:
            assert self.hyperparameters.pos_from == 'spacy'
            pos = sample_x.spacy_pos
        return pos

    def get_sample_mask(self, sample_xs):
        return [self.apply_mask(x) for ind, x in enumerate(sample_xs)]

    def sample_to_lowlevel(self, sample):
        mask = self.get_sample_mask(sample.xs)
        return LstmMlpMulticlassModel.Sample(
            xs=[self.sample_x_to_lowlevel(x, sample.xs, x_mask) for x_mask, x in zip(mask, sample.xs)],
            ys=[self.sample_y_to_lowlevel(y) for y in sample.ys],
            mask=mask
        )

    def report_masking(self, samples, name='samples'):
        n_sentences = 0
        n_y_labels = 0
        for s in samples:
            mask = self.get_sample_mask(s.xs)
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

    def save(self, base_path):
        self.model.save(base_path + '.ll')
        with open(base_path + '.hp', 'w') as f:
            json.dump(vars(self.hyperparameters), f, indent=2)

    @staticmethod
    def load(base_path):
        with open(base_path + '.hp', 'r') as f:
            model = LstmMlpSupersensesModel(hyperparameters=LstmMlpSupersensesModel.HyperParameters(**json.load(f)))
            model.model = LstmMlpMulticlassModel.load(base_path + '.ll')
            return model
