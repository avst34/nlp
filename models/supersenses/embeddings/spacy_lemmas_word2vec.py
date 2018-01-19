from datasets.streusle_v4 import StreusleLoader

SPACY_LEMMAS_WORD2VEC = StreusleLoader().get_spacy_lemmas_word2vec_model().as_dict()
