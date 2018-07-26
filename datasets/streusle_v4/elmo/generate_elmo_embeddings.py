from datasets.streusle_v4 import StreusleLoader

records = StreusleLoader().load()

def gen_sents_file():
    sents = [' '.join([t.token for t in rec.tagged_tokens]) + '\n' for rec in records]
    with open('/tmp/streusle_sents.txt', 'w') as f:
        for sent in sents:
            f.write(sent)

