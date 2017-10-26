import os
import json
from pprint import pprint
from collections import namedtuple
import supersenses
from vocabulary import Vocabulary, VocabularyBuilder

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'streusle-3.0')

class TaggedToken(namedtuple('TokenData_', ['token', 'pos', 'supersense'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.supersense_type = supersenses.get_supersense_type(self.supersense) if self.supersense else None

class StreusleRecord(namedtuple('StreusleRecord_', ['id', 'sentence', 'data'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tagged_tokens = [
            TaggedToken(
                token=tok_data[0],
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

wordVocabularyBuilder = VocabularyBuilder(lambda record: [x.token for x in record.tagged_tokens])
posVocabularyBuilder = VocabularyBuilder(lambda record: [x.pos for x in record.tagged_tokens if x.pos])
ssVocabularyBuilder = VocabularyBuilder(lambda record: [x.supersense for x in record.tagged_tokens if x.supersense])

