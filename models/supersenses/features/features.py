from collections import namedtuple

from models.supersenses import vocabs, embeddings
from models.supersenses.features.feature import Feature, FeatureType, MountPoint, Features
from models.supersenses.features.features_utils import get_parent, get_grandparent, get_child_of_type, get_children, \
    is_capitalized, get_gov, get_obj

[LSTM, MLP] = [MountPoint.LSTM, MountPoint.MLP]

def build_features(hyperparameters, override=None):
    override = override or {}
    hp = hyperparameters.clone(override)
    return Features([
        Feature('token-word2vec',   FeatureType.STRING, vocabs.TOKENS,     embeddings.TOKENS_WORD2VEC,  embedding_fallback=lambda tok: tok.token_word2vec, default_zero_vec=True, extractor=lambda tok, sent: tok.token,     mount_point=LSTM, enable=hp.use_token, update=hp.update_token_embd, masked_only=False),
        Feature('token.lemma-word2vec',  FeatureType.STRING, vocabs.LEMMAS,  embeddings.LEMMAS_WORD2VEC,  embedding_fallback=lambda tok: tok.lemma_word2vec,  default_zero_vec=True, update=hp.update_lemmas_embd, extractor=lambda tok, sent: tok.lemma, mount_point=LSTM,  enable=True, masked_only=False),
        Feature('token-internal',   FeatureType.STRING, vocabs.TOKENS,     embeddings.AUTO,  extractor=lambda tok, sent: tok.token, embedding_fallback=lambda tok: [0] * hp.token_internal_embd_dim,  default_zero_vec=True,  mount_point=LSTM, enable=hp.use_token_internal, dim=hp.token_internal_embd_dim, update=True, masked_only=False),
        Feature('token.ud_xpos',     FeatureType.ENUM, vocabs.UD_XPOS,     embeddings.AUTO,  dim=hp.ud_xpos_embd_dim,  update=True,        extractor=lambda tok, sent: tok.ud_xpos, mount_point=MLP,  enable=hp.use_ud_xpos),
        Feature('token.dep',     FeatureType.ENUM, vocabs.UD_DEPS,    embeddings.AUTO,   dim=hp.ud_deps_embd_dim,  update=True,    extractor=lambda tok, sent: tok.ud_dep,    mount_point=MLP,  enable=hp.use_ud_dep),
        Feature('token.ner',  FeatureType.ENUM, vocabs.NERS,  embeddings.AUTO,  dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: tok.ner, mount_point=MLP,  enable=hp.use_ner),
        Feature('token.govobj-config',  FeatureType.ENUM, vocabs.GOVOBJ_CONFIGS,  embeddings.AUTO,  dim=hp.govobj_config_embd_dim,  update=True, extractor=lambda tok, sent: tok.govobj_config, mount_point=MLP,  enable=hp.use_govobj),
        Feature('token.lexcat',  FeatureType.ENUM, vocabs.LEXCAT,  embeddings.AUTO,  dim=hp.lexcat_embd_dim,  update=True, extractor=lambda tok, sent: tok.lexcat, mount_point=MLP,  enable=hp.use_lexcat, masked_only=False),

        # Feature('prep-onehot', FeatureType.ENUM, vocabs.PREPS, embeddings.PREPS_ONEHOT, extractor=lambda tok, sent: tok.token,  mount_point=MLP, enable=hp.use_prep_onehot, fall_to_none=True),
        Feature('capitalized-word-follows', FeatureType.ENUM, vocabs.BOOLEAN, embeddings.BOOLEAN, extractor=lambda tok, sent: str(len(sent) > tok.ind + 1 and is_capitalized(sent[tok.ind + 1]) or len(sent) > tok.ind + 2 and is_capitalized(sent[tok.ind + 2])),  mount_point=MLP, masked_only=True, enable=True),

        Feature('token-gov',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_gov(tok, sent).ind, mount_point=MLP, enable=hp.use_govobj),
        Feature('token-gov.ud_xpos',       FeatureType.ENUM, vocabs.UD_XPOS,       embeddings.AUTO,    dim=hp.ud_xpos_embd_dim,  update=True,       extractor=lambda tok, sent: get_gov(tok, sent).ud_xpos, mount_point=MLP, enable=hp.use_govobj and hp.use_ud_xpos),
        Feature('token-gov.dep',    FeatureType.ENUM, vocabs.UD_DEPS,   embeddings.AUTO,   dim=hp.ud_deps_embd_dim,  update=True,   extractor=lambda tok, sent: get_gov(tok, sent).ud_dep, mount_point=MLP, enable=hp.use_govobj and hp.use_ud_dep),
        Feature('token-gov.ner', FeatureType.ENUM, vocabs.NERS, embeddings.AUTO, dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_gov(tok, sent).ner, mount_point=MLP, enable=hp.use_govobj and hp.use_ner),

        Feature('token-obj',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_obj(tok, sent).ind, mount_point=MLP, enable=hp.use_govobj),
        Feature('token-obj.ud_xpos',       FeatureType.ENUM, vocabs.UD_XPOS,       embeddings.AUTO,   dim=hp.ud_xpos_embd_dim,  update=True,       extractor=lambda tok, sent: get_obj(tok, sent).ud_xpos, mount_point=MLP, enable=hp.use_govobj and hp.use_ud_xpos),
        Feature('token-obj.dep',    FeatureType.ENUM, vocabs.UD_DEPS,   embeddings.AUTO,  dim=hp.ud_deps_embd_dim,  update=True,   extractor=lambda tok, sent: get_obj(tok, sent).ud_dep, mount_point=MLP, enable=hp.use_govobj and hp.use_ud_dep),
        Feature('token-obj.ner', FeatureType.ENUM, vocabs.NERS, embeddings.AUTO, dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_obj(tok, sent).ner, mount_point=MLP, enable=hp.use_govobj and hp.use_ner),

        # Feature('token-spacy-pobj-child',           FeatureType.REF,  None,              None,                         extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj').ind, mount_point=MLP, enable=hp.use_ud_dep and hp.deps_from == 'spacy'),
        # Feature('token-spacy-pobj-child.ud_xpos',       FeatureType.ENUM, vocabs.UD_XPOS,        embeddings.AUTO,    dim=hp.ud_xpos_embd_dim,  update=True,        extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj').ud_xpos, mount_point=MLP, enable=hp.use_ud_dep and hp.deps_from == 'spacy' and hp.use_ud_xpos),
        # Feature('token-spacy-pobj-child.dep', FeatureType.ENUM, vocabs.UD_DEPS, embeddings.AUTO, dim=hp.ud_deps_embd_dim,  update=True,  extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj').spacy_dep, mount_point=MLP, enable=hp.use_ud_dep and hp.deps_from == 'spacy'),
        # Feature('token-spacy-pobj-child.ner', FeatureType.ENUM, vocabs.NERS,  embeddings.AUTO,  dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj').spacy_ner, mount_point=MLP, enable=hp.use_ud_dep and hp.deps_from == 'spacy' and hp.use_ner),
        #
        # Feature('token-has-children', FeatureType.ENUM,  vocabs.BOOLEAN, embeddings.BOOLEAN, extractor=lambda tok, sent: str(len(get_children(tok, sent)) > 0), mount_point=MLP, enable=hp.use_ud_dep and hp.deps_from == 'spacy'),
    ])