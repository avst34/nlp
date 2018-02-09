import os

from datasets.streusle_v4 import StreusleLoader, supersenses, Word2VecModel
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
from vocabulary import Vocabulary


def format_list(l):
    return '[\n' + ',\n'.join(repr(s) for s in l) + ']\n'

def dump_vocabs(vocabs):
    for vocab in vocabs:
        with open('models/supersenses/vocabs/' + vocab.name.lower() + '.py', 'w', encoding='utf-8') as out_f:
            out_f.write("""from vocabulary import Vocabulary
            
%s = Vocabulary('%s', %s)
""" % (vocab.name, vocab.name, format_list(vocab.as_list())))

def build_vocabs():
    tasks = ['.'.join([id, syn]) for id in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]

    loader = StreusleLoader()
    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/streusle_4alpha'
    all_files = [STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json' for task in tasks]
    records = sum([loader.load(f, input_format='json') for f in all_files], [])
    samples = [streusle_record_to_lstm_model_sample(r) for r in records]

    pp_vocab = Vocabulary('PREPS')
    pp_vocab.add_words(set([x.token for s in samples for x, y in zip(s.xs, s.ys) if any([y.supersense_role, y.supersense_func])]))

    ner_vocab = Vocabulary('NERS')
    ner_vocab.add_words(set([x.ner for s in samples for x, y in zip(s.xs, s.ys)]))
    ner_vocab.add_word(None)

    lemmas_vocab = Vocabulary('LEMMAS')
    lemmas_vocab.add_words(set([x.corenlp_lemma for s in samples for x, y in zip(s.xs, s.ys)]))

    ud_dep_vocab = Vocabulary('UD_DEPS')
    ud_dep_vocab.add_words(set([x.ud_dep for s in samples for x, y in zip(s.xs, s.ys)]))
    ud_dep_vocab.add_word(None)

    ud_xpos_vocab = Vocabulary('UD_XPOS')
    ud_xpos_vocab.add_words(set([x.ud_xpos for s in samples for x, y in zip(s.xs, s.ys)]))
    ud_xpos_vocab.add_word(None)

    token_vocab = Vocabulary('TOKENS')
    token_vocab.add_words(set([x.token for s in samples for x, y in zip(s.xs, s.ys)]))

    govobj_config_vocab = Vocabulary('GOVOBJ_CONFIGS')
    govobj_config_vocab.add_words(set([x.govobj_config for s in samples for x, y in zip(s.xs, s.ys)]))

    pss_vocab = Vocabulary('PSS')
    pss_vocab.add_words(supersenses.PREPOSITION_SUPERSENSES_SET)
    pss_vocab.add_word(None)

    return [pp_vocab, ner_vocab, lemmas_vocab, ud_dep_vocab, ud_xpos_vocab, token_vocab, pss_vocab, govobj_config_vocab]

def dump_w2v(vocabs):
    wvm = Word2VecModel.load_google_model()
    for name in ['TOKENS', 'LEMMAS']:
        vocab = [v for v in vocabs if v.name == name][0]
        with open('models/supersenses/embeddings/' + vocab.name.lower() + '_word2vec.pickle', 'wb') as f:
            wvm.dump(vocab.all_words(), f, skip_missing=True)

if __name__ == '__main__':
    vocabs = build_vocabs()
    dump_vocabs(vocabs)
    dump_w2v(vocabs)
