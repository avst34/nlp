import os
import pickle

with open(os.path.dirname(__file__) + '/syntax_vectors_en_100.pickle', 'rb') as f:
    SYNTAX_WORD_VECTORS = pickle.load(f)
