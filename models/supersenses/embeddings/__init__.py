from .boolean import BOOLEAN
from .lemmas_word2vec import LEMMAS_WORD2VEC
from .tokens_word2vec import TOKENS_WORD2VEC

AUTO = 'AUTO' # A randomally initialized embeddings that will be leared during training (vector size should be provided extenally)
INSTANCE = 'INSTANCE' # A per-instance embedding provided externally