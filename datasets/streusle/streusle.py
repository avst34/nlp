import os
import csv
import json
from pprint import pprint
from collections import namedtuple
import supersenses
from vocabulary import Vocabulary, VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'streusle-3.0-v2')

ENHANCEMENTS = namedtuple('SEnhancements', ['WORD2VEC_PATH', 'WORD2VEC_MISSING_PATH', 'SPACY_DEP_TREES', 'DEV_SET_SENTIDS', 'DEV_SET_SENTIDS_UD_SPLIT'])(
    WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'word2vec.pickle'),
    WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'word2vec_missing.json'),
    SPACY_DEP_TREES=os.path.join(STREUSLE_DIR, 'spacy_dep_trees.json'),
    DEV_SET_SENTIDS=os.path.join(STREUSLE_DIR, 'splits/psst-dev.sentids'),
    DEV_SET_SENTIDS_UD_SPLIT=os.path.join(STREUSLE_DIR, 'splits/psst-dev-ud-split.sentids')
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

class TaggedToken(namedtuple('TokenData_', ['token', 'token_word2vec', 'pos', 'supersense_role', 'supersense_func', 'head_ind', 'dep'])):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if (self.supersense_role is not None) != (self.supersense_role is not None):
            raise Exception("TaggedToken initialized with only one supersense")

        self.supersense_role_type = supersenses.get_supersense_type(self.supersense_role) if self.supersense_role else None
        self.supersense_func_type = supersenses.get_supersense_type(self.supersense_func) if self.supersense_func else None
        combined_supersense = None
        if self.supersense_role and self.supersense_func:
            combined_supersense = self.supersense_role + '|' + self.supersense_func
        self.combined_supersense = combined_supersense

class StreusleRecord:

    def __init__(self,
                 id,
                 sentence,
                 data,
                 spacy_dep_tree,
                 only_supersenses=None):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.data = data
        self.spacy_dep_tree = spacy_dep_tree
        self.ignored_supersenses = []

        if not only_supersenses:
            only_supersenses = supersenses.SUPERSENSES_SET

        def filter_supersense(ss):
            if ss in only_supersenses:
                return ss
            if ss is not None:
                self.ignored_supersenses.append(ss)
            return None

        def extract_supersense_pair(label):
            if label is not None:
                _label = label.split(';')[0]
            else:
                _label = None

            if _label is None or any([t in _label.lower() for t in ['`', '_', '?', 'mwe']]):
                pair = [None, None]
            else:
                pair = [filter_supersense(x.strip()) for x in _label.split('|')]
            if len(pair) == 1:
                pair = [pair[0], pair[0]]
            if None in pair:
                pair = [None, None]
                if label is not None:
                    self.ignored_supersenses.append(label)
            return pair

        self.tagged_tokens = [
            TaggedToken(
                token=tok_data[0],
                token_word2vec=W2V.get(tok_data[0]),
                pos=tok_data[1],
                head_ind=self.spacy_dep_tree[i].head_ind if self.spacy_dep_tree else None,
                dep=self.spacy_dep_tree[i].dep if self.spacy_dep_tree else None,
                # supersense=filter_supersense(self.data['labels'].get(str(i + 1), [None, None])[1]),
                supersense_role=extract_supersense_pair(self.data['labels'].get(str(i + 1), [None, None])[1])[0],
                supersense_func=extract_supersense_pair(self.data['labels'].get(str(i + 1), [None, None])[1])[1]
            ) for i, tok_data in enumerate(self.data['words'])
        ]
        self.pss_tokens = [x for x in self.tagged_tokens if x.supersense_func in supersenses.PREPOSITION_SUPERSENSES_SET or x.supersense_role in supersenses.PREPOSITION_SUPERSENSES_SET]

class StreusleLoader(object):

    def __init__(self):
        pass

    def load(self, only_with_supersenses=supersenses.PREPOSITION_SUPERSENSES_SET):
        streusle_file = os.path.join(STREUSLE_DIR, 'streusle.sst')
        print('Loading streusle data from ' + streusle_file)
        with open(streusle_file) as f:
            records = []
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.split('\t')
                record = StreusleRecord(id=line[0],
                                        sentence=line[1],
                                        data=json.loads(line[2]),
                                        spacy_dep_tree=SPACY_DEP_TREES.get(line[0]),
                                        only_supersenses=only_with_supersenses)
                if only_with_supersenses:
                    if not any([token.combined_supersense for token in record.tagged_tokens]):
                        continue
                records.append(record)
            test_sentids = self._load_test_sentids()
            dev_sentids = self._load_dev_sentids()
            test_records = [r for r in records if r.id in test_sentids]
            dev_records = [r for r in records if r.id in dev_sentids]
            train_records = [r for r in records if r.id not in test_sentids and r.id not in dev_sentids]
            return train_records, dev_records, test_records

    def _load_test_sentids(self):
        with open(os.path.join(STREUSLE_DIR, 'splits', 'psst-test-ud-split.sentids'), 'r') as f:
            return set([x.strip() for x in f.readlines()])

    def _load_dev_sentids(self):
        with open(os.path.join(ENHANCEMENTS.DEV_SET_SENTIDS_UD_SPLIT), 'r') as f:
            return set([x.strip() for x in f.readlines()])

    @staticmethod
    def get_dist(records, all_supersenses=supersenses.PREPOSITION_SUPERSENSES_SET):
        return {
            pss: len([tok for rec in records for tok in rec.pss_tokens if tok.supersense == pss])
            for pss in all_supersenses
        }

    def dump_split_dist(self, out_csv_path, all_supersenses=supersenses.PREPOSITION_SUPERSENSES_SET):
        train, dev, test = self.load()
        train_dist = self.get_dist(train)
        dev_dist = self.get_dist(dev)
        test_dist = self.get_dist(test)
        with open(out_csv_path, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['supersense', 'train', 'dev', 'test'])
            for pss in sorted(all_supersenses):
                csvwriter.writerow([pss] + ["%2.4f%% (%d)" % (dist.get(pss, 0)/sum(dist.values())*100, dist.get(pss, 0)) for dist in [train_dist, dev_dist, test_dist]])

    def get_tokens_word2vec_model(self):
        return W2V


wordVocabularyBuilder = VocabularyBuilder(lambda record: [x.token for x in record.tagged_tokens])
posVocabularyBuilder = VocabularyBuilder(lambda record: [x.pos for x in record.tagged_tokens if x.pos])
ssVocabularyBuilder = VocabularyBuilder(lambda record: [x.supersense for x in record.tagged_tokens if x.supersense])
depsVocabularyBuilder = VocabularyBuilder(lambda record: [x.dep for x in record.tagged_tokens if x.dep])

