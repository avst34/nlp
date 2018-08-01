from datasets.streusle_v4 import StreusleLoader

# records = StreusleLoader().load()

def gen_sents_file():
    sents = [' '.join([t.token for t in rec.tagged_tokens]) + '\n' for rec in records]
    with open('/tmp/streusle_sents.txt', 'w') as f:
        for sent in sents:
            f.write(sent)

records_with_elmo = StreusleLoader(load_elmo=True).load()
StreusleLoader(load_elmo=True).load_train()
StreusleLoader(load_elmo=True).load_dev()
StreusleLoader(load_elmo=True).load_test()
for rec in records_with_elmo[:2]:
    print(rec.tagged_tokens[0].elmo)