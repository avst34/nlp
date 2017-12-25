from .preps_onehot import PREPS_ONEHOT
from .spacy_deps_onehot import SPACY_DEPS_ONEHOT
from .spacy_ner_onehot import SPACY_NER_ONEHOT
from .tokens_word2vec import TOKENS_WORD2VEC
from .ud_deps_onehot import UD_DEPS_ONEHOT

AUTO = 'AUTO' # A randomally initialized embeddings that will be leared during training (vector size should be provided extenally)