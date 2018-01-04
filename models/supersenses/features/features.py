from collections import namedtuple

from models.supersenses import vocabs, embeddings
from models.supersenses.features.feature import Feature, FeatureType, MountPoint, Features
from models.supersenses.features.features_utils import get_parent, get_grandparent, get_child_of_type, get_children

[LSTM, MLP] = [MountPoint.LSTM, MountPoint.MLP]

def build_features(hyperparameters):
    hp = hyperparameters
    return Features([
        Feature('token-word2vec',   FeatureType.ENUM, vocabs.TOKENS,     embeddings.TOKENS_WORD2VEC,   extractor=lambda tok, sent: tok.token,     mount_point=LSTM, enable=hp.use_token, update=hp.update_token_embd, masked_only=False),
        Feature('token-internal',   FeatureType.ENUM, vocabs.TOKENS,     embeddings.AUTO,              extractor=lambda tok, sent: tok.token,     mount_point=LSTM, enable=hp.use_token_internal, dim=hp.token_internal_embd_dim, update=True, masked_only=False),
        Feature('token.ud-pos',     FeatureType.ENUM, vocabs.UD_POS,     embeddings.UD_POS_ONEHOT,        extractor=lambda tok, sent: tok.ud_pos, mount_point=MLP,  enable=hp.use_pos and hp.pos_from == 'ud'),
        Feature('token.spacy-pos',     FeatureType.ENUM, vocabs.SPACY_POS,     embeddings.SPACY_POS_ONEHOT,        extractor=lambda tok, sent: tok.spacy_pos,   mount_point=MLP,  enable=hp.use_pos and hp.pos_from == 'spacy'),
        Feature('token.ud-dep',     FeatureType.ENUM, vocabs.UD_DEPS,    embeddings.UD_DEPS_ONEHOT,    extractor=lambda tok, sent: tok.ud_dep,    mount_point=MLP,  enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token.spacy-dep',  FeatureType.ENUM, vocabs.SPACY_DEPS, embeddings.SPACY_DEPS_ONEHOT, extractor=lambda tok, sent: tok.spacy_dep, mount_point=MLP,  enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token.spacy-ner',  FeatureType.ENUM, vocabs.SPACY_NER,  embeddings.SPACY_NER_ONEHOT,  extractor=lambda tok, sent: tok.spacy_ner, mount_point=MLP,  enable=hp.use_spacy_ner),

        Feature('prep-onehot', FeatureType.ENUM, vocabs.PREPS, embeddings.PREPS_ONEHOT, extractor=lambda tok, sent: tok.token,  mount_point=MLP, enable=hp.use_prep_onehot, fall_to_none=True),

        Feature('token-ud-parent',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_parent(tok, sent, 'ud_head_ind').ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token-ud-parent.ud-pos',       FeatureType.ENUM, vocabs.UD_POS,       embeddings.UD_POS_ONEHOT,       extractor=lambda tok, sent: get_parent(tok, sent, 'ud_head_ind').ud_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud' and hp.use_pos and hp.pos_from == 'ud'),
        Feature('token-ud-parent.spacy-pos',       FeatureType.ENUM, vocabs.SPACY_POS,       embeddings.SPACY_POS_ONEHOT,       extractor=lambda tok, sent: get_parent(tok, sent, 'ud_head_ind').spacy_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud' and hp.use_pos and hp.pos_from == 'spacy'),
        Feature('token-ud-parent.ud-dep',    FeatureType.ENUM, vocabs.UD_DEPS,   embeddings.UD_DEPS_ONEHOT,   extractor=lambda tok, sent: get_parent(tok, sent, 'ud_head_ind').ud_dep, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token-ud-parent.spacy-ner', FeatureType.ENUM, vocabs.SPACY_NER, embeddings.SPACY_NER_ONEHOT, extractor=lambda tok, sent: get_parent(tok, sent, 'ud_head_ind').spacy_ner, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),

        Feature('token-ud-grandparent',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind').ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token-ud-grandparent.ud-pos',       FeatureType.ENUM, vocabs.UD_POS,       embeddings.UD_POS_ONEHOT,       extractor=lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind').ud_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud' and hp.use_pos and hp.pos_from == 'ud'),
        Feature('token-ud-grandparent.spacy-pos',       FeatureType.ENUM, vocabs.SPACY_POS,       embeddings.SPACY_POS_ONEHOT,       extractor=lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind').spacy_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud' and hp.use_pos and hp.pos_from == 'spacy'),
        Feature('token-ud-grandparent.ud-dep',    FeatureType.ENUM, vocabs.UD_DEPS,   embeddings.UD_DEPS_ONEHOT,   extractor=lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind').ud_dep, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token-ud-grandparent.spacy-ner', FeatureType.ENUM, vocabs.SPACY_NER, embeddings.SPACY_NER_ONEHOT, extractor=lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind').spacy_ner, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),

        Feature('token-spacy-parent',           FeatureType.REF,  None,              None,                         extractor=lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind').ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-parent.ud-pos',       FeatureType.ENUM, vocabs.UD_POS,        embeddings.UD_POS_ONEHOT,        extractor=lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind').ud_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_pos and hp.pos_from == 'ud'),
        Feature('token-spacy-parent.spacy-pos',       FeatureType.ENUM, vocabs.SPACY_POS,        embeddings.SPACY_POS_ONEHOT,        extractor=lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind').spacy_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_pos and hp.pos_from == 'spacy'),
        Feature('token-spacy-parent.spacy-dep', FeatureType.ENUM, vocabs.SPACY_DEPS, embeddings.SPACY_DEPS_ONEHOT, extractor=lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind').spacy_dep, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-parent.spacy-ner', FeatureType.ENUM, vocabs.SPACY_NER,  embeddings.SPACY_NER_ONEHOT,  extractor=lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind').spacy_ner, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),

        Feature('token-spacy-pobj-child',           FeatureType.REF,  None,              None,                         extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', 'spacy_head_ind', 'spacy_dep').ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-pobj-child.ud-pos',       FeatureType.ENUM, vocabs.UD_POS,        embeddings.UD_POS_ONEHOT,        extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', 'spacy_head_ind', 'spacy_dep').ud_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_pos and hp.pos_from == 'ud'),
        Feature('token-spacy-pobj-child.spacy-pos',       FeatureType.ENUM, vocabs.SPACY_POS,        embeddings.SPACY_POS_ONEHOT,        extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', 'spacy_head_ind', 'spacy_dep').spacy_pos, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_pos and hp.pos_from == 'spacy'),
        Feature('token-spacy-pobj-child.spacy-dep', FeatureType.ENUM, vocabs.SPACY_DEPS, embeddings.SPACY_DEPS_ONEHOT, extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', 'spacy_head_ind', 'spacy_dep').spacy_dep, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-pobj-child.spacy-ner', FeatureType.ENUM, vocabs.SPACY_NER,  embeddings.SPACY_NER_ONEHOT,  extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', 'spacy_head_ind', 'spacy_dep').spacy_ner, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),

        Feature('token-spacy-has-children', FeatureType.ENUM,  vocabs.BOOLEAN, embeddings.BOOLEAN_ONEHOT, extractor=lambda tok, sent: str(len(get_children(tok, sent, 'spacy_head_ind')) > 0), mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
    ])