import os
import pickle

with open(os.path.basename(__file__) + '/lemmas_word2vec.pickle', 'rb') as f:
    LEMMAS_WORD2VEC = pickle.load(f)
