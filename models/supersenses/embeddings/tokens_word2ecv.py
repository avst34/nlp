import os
import pickle

with open(os.path.basename(__file__) + '/tokens_word2vec.pickle', 'rb') as f:
    TOKENS_WORD2VEC = pickle.load(f)
