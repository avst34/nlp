import os
import pickle

from allennlp.commands.elmo import ElmoEmbedder
import hashlib
elmo = ElmoEmbedder()



def run_elmo(tokens, cache_dir=os.path.dirname(__file__) + '/elmo_cache'):
    print('run_elmo', tokens)
    hash = hashlib.md5(' '.join(tokens).encode('utf8')).digest().hex()
    fname = cache_dir +'/' + hash + '.elmo'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        embeddings = elmo.embed_sentence(tokens)
        embeddings = [(embeddings[0][i], embeddings[1][i], embeddings[2][i]) for i in range(len(tokens))]
        with open(fname, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings
