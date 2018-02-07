def _get_embd(type, name):
    return globals()[type.upper() + "_" + name]

from .tokens_word2vec import TOKENS_WORD2VEC

from .ud_lemmas_word2vec import UD_LEMMAS_WORD2VEC
from .spacy_lemmas_word2vec import SPACY_LEMMAS_WORD2VEC

def LEMMAS_WORD2VEC(type):
    return _get_embd(type, 'LEMMAS_WORD2VEC')

from .boolean import BOOLEAN

AUTO = 'AUTO' # A randomally initialized embeddings that will be leared during training (vector size should be provided extenally)