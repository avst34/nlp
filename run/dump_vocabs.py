def format_list(l):
    return '[\n' + ',\n'.join(repr(s) for s in l) + ']\n'

def dump_vocabularies(vocabs):
    for vocab in vocabs:
        with open('models/supersenses/vocabs/' + vocab.name.lower() + '.py', 'w', encoding='utf-8') as out_f:
            out_f.write("""from vocabulary import Vocabulary
            
%s = Vocabulary('%s', %s)
""" % (vocab.name, vocab.name, format_list(vocab.as_list())))
