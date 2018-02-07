from collections import namedtuple

from models.supersenses import vocabs, embeddings
from models.supersenses.features.feature import Feature, FeatureType, MountPoint, Features
from models.supersenses.features.features_utils import get_parent, get_grandparent, get_child_of_type, get_children, is_capitalized

[LSTM, MLP] = [MountPoint.LSTM, MountPoint.MLP]

def build_features(hyperparameters, override=None):
    override = override or {}
    hp = hyperparameters.clone(override)
    return Features([
        Feature('token-word2vec',   FeatureType.ENUM, vocabs.TOKENS,     embeddings.TOKENS_WORD2VEC,  default_zero_vec=True,   extractor=lambda tok, sent: tok.token,     mount_point=LSTM, enable=hp.use_token, update=hp.update_token_embd, masked_only=False),
        Feature('token.lemma-word2vec',  FeatureType.ENUM, vocabs.LEMMAS(hp.lemmas_from),  embeddings.LEMMAS_WORD2VEC(hp.lemmas_from),  default_zero_vec=True, update=hp.update_lemmas_embd, extractor=lambda tok, sent: tok.lemma(hp.lemmas_from), mount_point=LSTM,  enable=True, masked_only=False),
        Feature('token-internal',   FeatureType.ENUM, vocabs.TOKENS,     embeddings.AUTO,  extractor=lambda tok, sent: tok.token,     mount_point=LSTM, enable=hp.use_token_internal, dim=hp.token_internal_embd_dim, update=True, masked_only=False),
        Feature('token.pos',     FeatureType.ENUM, vocabs.POS(hp.pos_from),     embeddings.AUTO,  dim=hp.pos_embd_dim,  update=True,        extractor=lambda tok, sent: tok.pos(hp.pos_from), mount_point=MLP,  enable=hp.use_pos),
        Feature('token.dep',     FeatureType.ENUM, vocabs.DEPS(hp.deps_from),    embeddings.AUTO,   dim=hp.deps_embd_dim,  update=True,    extractor=lambda tok, sent: tok.dep(hp.deps_from),    mount_point=MLP,  enable=hp.use_dep),
        Feature('token.ner',  FeatureType.ENUM, vocabs.NER(hp.ners_from),  embeddings.AUTO,  dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: tok.ner(hp.ners_from), mount_point=MLP,  enable=hp.use_ner),

        # Feature('prep-onehot', FeatureType.ENUM, vocabs.PREPS, embeddings.PREPS_ONEHOT, extractor=lambda tok, sent: tok.token,  mount_point=MLP, enable=hp.use_prep_onehot, fall_to_none=True),
        Feature('capitalized-word-follows', FeatureType.ENUM, vocabs.BOOLEAN, embeddings.BOOLEAN, extractor=lambda tok, sent: str(len(sent) > tok.ind + 1 and is_capitalized(sent[tok.ind + 1]) or len(sent) > tok.ind + 2 and is_capitalized(sent[tok.ind + 2])),  mount_point=MLP, masked_only=True, enable=True),

        Feature('token-parent',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_parent(tok, sent, hp.deps_from).ind, mount_point=MLP, enable=hp.use_dep),
        Feature('token-parent.pos',       FeatureType.ENUM, vocabs.POS(hp.pos_from),       embeddings.AUTO,    dim=hp.pos_embd_dim,  update=True,       extractor=lambda tok, sent: get_parent(tok, sent, hp.deps_from).pos(hp.pos_from), mount_point=MLP, enable=hp.use_dep and hp.use_pos),
        Feature('token-parent.dep',    FeatureType.ENUM, vocabs.DEPS(hp.deps_from),   embeddings.AUTO,   dim=hp.deps_embd_dim,  update=True,   extractor=lambda tok, sent: get_parent(tok, sent, hp.deps_from).dep(hp.deps_from), mount_point=MLP, enable=hp.use_dep),
        Feature('token-parent.ner', FeatureType.ENUM, vocabs.NER(hp.ners_from), embeddings.AUTO, dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_parent(tok, sent, hp.deps_from).ner(hp.ners_from), mount_point=MLP, enable=hp.use_dep and hp.use_ner),

        Feature('token-grandparent',           FeatureType.REF,  None,             None,                        extractor=lambda tok, sent: get_grandparent(tok, sent, hp.deps_from).ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud'),
        Feature('token-grandparent.pos',       FeatureType.ENUM, vocabs.POS(hp.pos_from),       embeddings.AUTO,   dim=hp.pos_embd_dim,  update=True,       extractor=lambda tok, sent: get_grandparent(tok, sent, hp.deps_from).pos(hp.pos_from), mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'ud' and hp.use_pos),
        Feature('token-grandparent.dep',    FeatureType.ENUM, vocabs.DEPS(hp.deps_from),   embeddings.AUTO,  dim=hp.deps_embd_dim,  update=True,   extractor=lambda tok, sent: get_grandparent(tok, sent, hp.deps_from).dep(hp.deps_from), mount_point=MLP, enable=hp.use_dep),
        Feature('token-grandparent.ner', FeatureType.ENUM, vocabs.NER(hp.ners_from), embeddings.AUTO, dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_grandparent(tok, sent, hp.deps_from).ner(hp.ners_from), mount_point=MLP, enable=hp.use_dep and hp.use_ner),

        Feature('token-spacy-pobj-child',           FeatureType.REF,  None,              None,                         extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', hp.deps_from).ind, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-pobj-child.pos',       FeatureType.ENUM, vocabs.POS(hp.pos_from),        embeddings.AUTO,    dim=hp.pos_embd_dim,  update=True,        extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', hp.deps_from).pos(hp.pos_from), mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_pos),
        Feature('token-spacy-pobj-child.dep', FeatureType.ENUM, vocabs.DEPS(hp.deps_from), embeddings.AUTO, dim=hp.deps_embd_dim,  update=True,  extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', hp.deps_from).spacy_dep, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
        Feature('token-spacy-pobj-child.ner', FeatureType.ENUM, vocabs.NER(hp.ners_from),  embeddings.AUTO,  dim=hp.ner_embd_dim,  update=True, extractor=lambda tok, sent: get_child_of_type(tok, sent, 'pobj', hp.deps_from).spacy_ner, mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy' and hp.use_ner),

        Feature('token-spacy-has-children', FeatureType.ENUM,  vocabs.BOOLEAN, embeddings.BOOLEAN, extractor=lambda tok, sent: str(len(get_children(tok, sent, hp.deps_from)) > 0), mount_point=MLP, enable=hp.use_dep and hp.deps_from == 'spacy'),
    ])