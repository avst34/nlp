from datasets.streusle_v3 import StreusleLoader

TOKENS_WORD2VEC = StreusleLoader().get_tokens_word2vec_model().as_dict()
