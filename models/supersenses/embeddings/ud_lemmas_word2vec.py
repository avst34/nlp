from datasets.streusle_v4 import StreusleLoader

UD_LEMMAS_WORD2VEC = StreusleLoader().get_ud_lemmas_word2vec_model().as_dict()
