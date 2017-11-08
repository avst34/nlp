import os
import json
from pprint import pprint
from collections import namedtuple
import supersenses
from vocabulary import Vocabulary, VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'streusle-3.0')

ENHANCEMENTS = namedtuple('SEnhancements', ['WORD2VEC_PATH', 'WORD2VEC_MISSING_PATH'])(
    WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'word2vec.pickle'),
    WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'word2vec_missing.json')
)

w2v = Word2VecModel({})
if os.path.exists(ENHANCEMENTS.WORD2VEC_PATH):
    with open(ENHANCEMENTS.WORD2VEC_PATH, 'rb')as f:
        w2v = Word2VecModel.load(f)

class TaggedToken(namedtuple('TokenData_', ['token', 'token_word2vec', 'pos', 'supersense'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.supersense_type = supersenses.get_supersense_type(self.supersense) if self.supersense else None

class StreusleRecord(namedtuple('StreusleRecord_', ['id', 'sentence', 'data'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tagged_tokens = [
            TaggedToken(
                token=tok_data[0],
                token_word2vec=w2v.get(tok_data[0]),
                pos=tok_data[1],
                supersense=supersenses.filter_non_supersense(self.data['labels'].get(str(i + 1), [None, None])[1])
            ) for i, tok_data in enumerate(self.data['words'])
        ]

class StreusleLoader(object):

    def __init__(self):
        self._f = open(os.path.join(STREUSLE_DIR, 'streusle.sst'))
        pass

    def load(self, limit=None):
        records = []
        while True:
            if limit == len(records):
                break
            line = self._f.readline()
            if line == '':
                break
            line = line.split('\t')
            records.append(StreusleRecord(id=line[0], sentence=line[1], data=json.loads(line[2])))
        return records

    def get_tokens_word2vec_model(self):
        return w2v


wordVocabularyBuilder = VocabularyBuilder(lambda record: [x.token for x in record.tagged_tokens])
posVocabularyBuilder = VocabularyBuilder(lambda record: [x.pos for x in record.tagged_tokens if x.pos])
ssVocabularyBuilder = VocabularyBuilder(lambda record: [x.supersense for x in record.tagged_tokens if x.supersense])

