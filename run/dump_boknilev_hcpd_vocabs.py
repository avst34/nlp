import os
from vocabulary import Vocabulary
from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples

SRC_BASE = os.path.dirname(__file__) + '/..'


def format_list(l):
    return '[\n' + ',\n'.join(repr(s) for s in l) + ']\n'

def dump_vocabs(vocabs):
    for vocab in vocabs:
        with open(SRC_BASE + '/models/hcpd/vocabs/' + vocab.name.lower() + '.py', 'w', encoding='utf-8') as out_f:
            out_f.write("""from vocabulary import Vocabulary
            
%s = Vocabulary('%s', %s)
""" % (vocab.name, vocab.name, format_list(vocab.as_list())))


def build_vocabs():
    train, dev, test = load_boknilev()
    samples = [s for r in train + dev + test for s in boknilev_record_to_hcpd_samples(r)]

    gold_pos_vocab = Vocabulary('GOLD_POS')
    gold_pos_vocab.add_words(set([hc.next_pos for s in samples for hc in s.x.head_cands]))
    gold_pos_vocab.add_word(None)

    words_vocab = Vocabulary('WORDS')
    words_vocab.add_words(set([hc.word for s in samples for hc in s.x.head_cands]))
    words_vocab.add_words(set([s.x.pp.word for s in samples]))
    words_vocab.add_words(set([s.x.child.word for s in samples]))
    words_vocab.add_word(None)

    return [gold_pos_vocab, words_vocab]

if __name__ == '__main__':
    vocabs = build_vocabs()
    dump_vocabs(vocabs)
