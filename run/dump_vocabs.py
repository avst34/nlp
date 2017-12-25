def dump_vocabularies(vocabs):
    for vocab in vocabs:
        with open('models/supersenses/vocabs/' + vocab.name + '.py', 'w') as out_f:
            out_f.write("""from vocabulary import Vocabulary
            
%s = Vocabulary('%s', %s)
""" % (vocab.name, vocab.name, repr(vocab.as_list())
       .replace(', ', ',\n').replace('[', '[\n').replace(']', ',\n]'))
                        )
