from collections import namedtuple

from models.supersenses import vocabs, embeddings
from models.supersenses.features.features_utils import get_parent, get_grandparent


class FeatureType:
    ENUM = 'ENUM'
    REF = 'REF'

Feature = namedtuple('Feature', ['name', 'type', 'vocab', 'embedding', 'extractor'])

FEATURES = [
    Feature('token-word2vec',          FeatureType.ENUM, vocabs.TOKENS,    embeddings.TOKENS_WORD2VEC, lambda tok, sent: tok.token),
    Feature('token-internal',          FeatureType.ENUM, vocabs.TOKENS,    embeddings.AUTO,            lambda tok, sent: tok.token),
    Feature('token-ud-parent',         FeatureType.REF,  None,             None,                       lambda tok, sent: get_parent(tok, sent, 'ud_head_ind')),
    Feature('token-ud-grandparent',    FeatureType.REF,  None,             None,                       lambda tok, sent: get_grandparent(tok, sent, 'ud_head_ind')),
    Feature('token-spacy-parent',      FeatureType.REF,  None,             None,                       lambda tok, sent: get_parent(tok, sent, 'spacy_head_ind')),
    Feature('token-spacy-grandparent', FeatureType.REF,  None,             None,                       lambda tok, sent: get_grandparent(tok, sent, 'spacy_head_ind')),

    Feature('ud-governor',         FeatureType.REF,  None,             lambda tok, sent: get_ud_governor(tok, sent)),
    Feature('ud-governor.pos',     FeatureType.ENUM, vocabs.POS,       lambda tok, sent: get_ud_governor(tok, sent).pos),
    Feature('ud-governor.ner',     FeatureType.REF,  vocabs.SPACY_NER, lambda tok, sent: get_ud_governor(tok, sent).spacy_ner),
    Feature('ud-head-noun',        FeatureType.REF,  None,             lambda tok, sent: get_ud_head_noun(tok, sent)),
    Feature('ud-head-noun.pos',    FeatureType.ENUM, vocabs.POS,       lambda tok, sent: get_ud_head_noun(tok, sent).pos),
    Feature('ud-head-noun.ner',    FeatureType.REF,  vocabs.SPACY_NER, lambda tok, sent: get_ud_head_noun(tok, sent).spacy_ner),
    Feature('spacy-governor',      FeatureType.REF,  None,             lambda tok, sent: get_ud_governor(tok, sent)),
    Feature('spacy-governor.pos',  FeatureType.ENUM, vocabs.POS,       lambda tok, sent: get_ud_governor(tok, sent).pos),
    Feature('spacy-governor.ner',  FeatureType.REF,  vocabs.SPACY_NER, lambda tok, sent: get_ud_governor(tok, sent).spacy_ner),
    Feature('spacy-head-noun',     FeatureType.REF,  None,             lambda tok, sent: get_ud_head_noun(tok, sent)),
    Feature('spacy-head-noun.pos', FeatureType.ENUM, vocabs.POS,       lambda tok, sent: get_ud_head_noun(tok, sent).pos),
    Feature('spacy-head-noun.ner', FeatureType.REF,  vocabs.SPACY_NER, lambda tok, sent: get_ud_head_noun(tok, sent).spacy_ner),
]