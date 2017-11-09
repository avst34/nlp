import os
import json
from pprint import pprint
from collections import namedtuple
import supersenses
from vocabulary import Vocabulary, VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'streusle-3.0')

ENHANCEMENTS = namedtuple('SEnhancements', ['WORD2VEC_PATH', 'WORD2VEC_MISSING_PATH', 'SPACY_DEP_TREES'])(
    WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'word2vec.pickle'),
    WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'word2vec_missing.json'),
    SPACY_DEP_TREES=os.path.join(STREUSLE_DIR, 'spacy_dep_trees.json')
)

W2V = Word2VecModel({})
if os.path.exists(ENHANCEMENTS.WORD2VEC_PATH):
    with open(ENHANCEMENTS.WORD2VEC_PATH, 'rb')as f:
        W2V = Word2VecModel.load(f)

SPACY_DEP_TREES = {}
TreeNode = namedtuple('TreeNode', ['head_ind', 'dep'])
if os.path.exists(ENHANCEMENTS.SPACY_DEP_TREES):
    with open(ENHANCEMENTS.SPACY_DEP_TREES, 'r')as f:
        SPACY_DEP_TREES = {
            rec_id: [TreeNode(head_ind=node[0], dep=node[1]) for node in tree_nodes]
            for rec_id, tree_nodes in json.load(f).items()
        }

class TaggedToken(namedtuple('TokenData_', ['token', 'token_word2vec', 'pos', 'supersense', 'head_ind', 'dep'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.supersense_type = supersenses.get_supersense_type(self.supersense) if self.supersense else None

class StreusleRecord(namedtuple('StreusleRecord_', ['id', 'sentence', 'data', 'spacy_dep_tree'])):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tagged_tokens = [
            TaggedToken(
                token=tok_data[0],
                token_word2vec=W2V.get(tok_data[0]),
                pos=tok_data[1],
                head_ind=self.spacy_dep_tree[i].head_ind if self.spacy_dep_tree else None,
                dep=self.spacy_dep_tree[i].dep if self.spacy_dep_tree else None,
                supersense=supersenses.filter_non_supersense(self.data['labels'].get(str(i + 1), [None, None])[1]),
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
            records.append(StreusleRecord(id=line[0], sentence=line[1], data=json.loads(line[2]), spacy_dep_tree=SPACY_DEP_TREES.get(line[0])))
        return records

    def get_tokens_word2vec_model(self):
        return W2V


wordVocabularyBuilder = VocabularyBuilder(lambda record: [x.token for x in record.tagged_tokens])
posVocabularyBuilder = VocabularyBuilder(lambda record: [x.pos for x in record.tagged_tokens if x.pos])
ssVocabularyBuilder = VocabularyBuilder(lambda record: [x.supersense for x in record.tagged_tokens if x.supersense])
depsVocabularyBuilder = VocabularyBuilder(lambda record: [x.dep for x in record.tagged_tokens if x.dep])

