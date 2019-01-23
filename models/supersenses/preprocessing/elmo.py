from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()

def run_elmo(tokens):
    embeddings = elmo.embed_sentence(tokens)
    return [(embeddings[0][i], embeddings[1][i], embeddings[2][i]) for i in range(len(tokens))]