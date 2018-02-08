from datasets.streusle_v4 import StreusleLoader

CORENLP_LEMMAS_WORD2VEC = StreusleLoader().get_corenlp_lemmas_word2vec_model().as_dict()
