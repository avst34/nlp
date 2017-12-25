from collections import namedtuple

from models.supersenses import vocabs


class Types:
    ENUM = 'ENUM'
    REF = 'REF'

Feature = namedtuple('Feature', ['name', 'type', 'vocab', 'extractor'])

FEATURES = [
    Feature('ud-governor', Types.REF, None, lambda token, sent: get_ud_governor(token, sent)),
    Feature('ud-governor.pos', Types.ENUM, vocabs.POS, lambda token, sent: get_ud_governor(token, sent).pos),
    Feature('ud-governor.ner', Types.REF, vocabs.SPACY_NER, lambda token, sent: get_ud_governor(token, sent).spacy_ner),
    Feature('ud-head-noun', Types.REF, None, lambda token, sent: get_ud_head_noun(token, sent)),
    Feature('ud-head-noun.pos', Types.ENUM, vocabs.POS, lambda token, sent: get_ud_head_noun(token, sent).pos),
    Feature('ud-head-noun.ner', Types.REF, vocabs.SPACY_NER, lambda token, sent: get_ud_head_noun(token, sent).spacy_ner),
    Feature('spacy-governor', Types.REF, None, lambda token, sent: get_ud_governor(token, sent)),
    Feature('spacy-governor.pos', Types.ENUM, vocabs.POS, lambda token, sent: get_ud_governor(token, sent).pos),
    Feature('spacy-governor.ner', Types.REF, vocabs.SPACY_NER, lambda token, sent: get_ud_governor(token, sent).spacy_ner),
    Feature('spacy-head-noun', Types.REF, None, lambda token, sent: get_ud_head_noun(token, sent)),
    Feature('spacy-head-noun.pos', Types.ENUM, vocabs.POS, lambda token, sent: get_ud_head_noun(token, sent).pos),
    Feature('spacy-head-noun.ner', Types.REF, vocabs.SPACY_NER, lambda token, sent: get_ud_head_noun(token, sent).spacy_ner),
]