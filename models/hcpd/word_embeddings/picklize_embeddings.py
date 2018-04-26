import pickle

embeddings = {}

with open('vectors.english.100.txt') as vf:
    lines = vf.readlines()
    for line in lines:
        toks = line.split()
        embeddings[toks[0]] = [float(x) for x in toks[1:]]

with open('syntax_vectors_en_100.pickle', 'wb') as vf:
    pickle.dump(embeddings, vf)