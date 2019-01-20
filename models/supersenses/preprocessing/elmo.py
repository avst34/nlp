from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()

def run_elmo(tokens):
    return zip(elmo.embed_sentence(tokens))