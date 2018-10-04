import os

from datasets.streusle_v4 import StreusleLoader, supersense_repo, Word2VecModel
from vocabulary import Vocabulary

SRC_BASE = os.path.dirname(__file__)


def format_list(l):
    return '[\n' + ',\n'.join(repr(s) for s in l) + ']\n'

def dump_vocabs(vocabs):
    for vocab in vocabs:
        with open(SRC_BASE + '/vocabs/' + vocab.name.lower() + '.py', 'w', encoding='utf-8') as out_f:
            out_f.write("""from vocabulary import Vocabulary
            
%s = Vocabulary('%s', %s)
""" % (vocab.name, vocab.name, format_list(vocab.as_list())))


def dump_w2v(vocabs):
    wvm = Word2VecModel.load_google_model()
    for name in ['TOKENS', 'LEMMAS']:
        vocab = [v for v in vocabs if v.name == name][0]
        with open(SRC_BASE + '/embeddings/' + vocab.name.lower() + '_word2vec.pickle', 'wb') as f:
            wvm.dump(vocab.all_words(), f, skip_missing=True)


def build_vocabs():
    tasks = ['.'.join([id, syn]) for id in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    stypes = ['train', 'dev', 'test']

    loader = StreusleLoader()
    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'
    all_files = [STREUSLE_BASE + '/' + stype + '/streusle.ud_' + stype + '.' + task + '.json' for task in tasks for stype in stypes]
    records = sum([loader.load(f, input_format='json') for f in all_files], [])

    pp_vocab = Vocabulary('PREPS')
    pp_vocab.add_words(set([prep for r in records for ttok in r.tagged_tokens for prep in (ttok.prep_toks or []) if ttok.supersense_role]))

    gov_vocab = Vocabulary('GOV')
    gov_vocab.add_words(set([r.tagged_tokens[ttok.gov_ind].token for r in records for ttok in r.tagged_tokens  if ttok.supersense_role and ttok.gov_ind is not None]))

    obj_vocab = Vocabulary('OBJ')
    obj_vocab.add_words(set([r.tagged_tokens[ttok.obj_ind].token for r in records for ttok in r.tagged_tokens  if ttok.supersense_role and ttok.obj_ind is not None]))

    pss_vocab = Vocabulary('PSS')
    pss_vocab.add_words(supersense_repo.PREPOSITION_SUPERSENSES_SET)
    pss_vocab.add_word(None)

    return [pp_vocab, gov_vocab, obj_vocab]

if __name__ == '__main__':
    vocabs = build_vocabs()
    dump_vocabs(vocabs)
    # dump_w2v(vocabs)
