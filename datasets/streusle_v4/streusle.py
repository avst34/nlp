import os
import sys
import csv
import json
from pprint import pprint
from collections import namedtuple
import supersenses
from vocabulary import Vocabulary, VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'streusle_4alpha')
sys.path.append(STREUSLE_DIR)
print(STREUSLE_DIR)
from .streusle_4alpha import conllulex2json

ENHANCEMENTS = namedtuple('SEnhancements', ['WORD2VEC_PATH', 'WORD2VEC_MISSING_PATH', 'SPACY_DEP_TREES', 'SPACY_NERS', 'SPACY_POS', 'DEV_SET_SENTIDS', 'DEV_SET_SENTIDS_UD_SPLIT', 'UD_DEP_TREES'])(
    WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'word2vec.pickle'),
    WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'word2vec_missing.json'),
    SPACY_DEP_TREES=os.path.join(STREUSLE_DIR, 'spacy_dep_trees.json'),
    SPACY_NERS=os.path.join(STREUSLE_DIR, 'spacy_ners.json'),
    SPACY_POS=os.path.join(STREUSLE_DIR, 'spacy_pos.json'),
    UD_DEP_TREES=os.path.join(STREUSLE_DIR, 'ud_dep_trees.json'),
    DEV_SET_SENTIDS=os.path.join(STREUSLE_DIR, 'splits/psst-dev.sentids'),
    DEV_SET_SENTIDS_UD_SPLIT=os.path.join(STREUSLE_DIR, 'splits/psst-dev-ud-split.sentids')
)

W2V = Word2VecModel({})
if os.path.exists(ENHANCEMENTS.WORD2VEC_PATH):
    with open(ENHANCEMENTS.WORD2VEC_PATH, 'rb')as f:
        W2V = Word2VecModel.load(f)

TreeNode = namedtuple('TreeNode', ['head_ind', 'dep'])
def load_dep_tree(tree_json_path):
    if os.path.exists(tree_json_path):
        with open(tree_json_path, 'r') as f:
            return {
                rec_id: [TreeNode(head_ind=node[0], dep=node[1]) for node in tree_nodes]
                for rec_id, tree_nodes in json.load(f).items()
            }
    else:
        return {}

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            obj = json.load(f)
    else:
        obj = default
    return obj


SPACY_DEP_TREES = load_dep_tree(ENHANCEMENTS.SPACY_DEP_TREES)
UD_DEP_TREES = load_dep_tree(ENHANCEMENTS.UD_DEP_TREES)
SPACY_NERS = load_json(ENHANCEMENTS.SPACY_NERS, {})
SPACY_POS = load_json(ENHANCEMENTS.SPACY_POS, {})

class TaggedToken:
    def __init__(self, token, ind, token_word2vec, supersense_role, supersense_func, spacy_head_ind, spacy_dep, ud_head_ind, ud_dep, part_of_wmwe, part_of_smwe, is_first_mwe_token, spacy_ner, ud_upos, ud_xpos, spacy_pos, ud_lemma):
        self.token = token
        self.ind = ind
        self.token_word2vec = token_word2vec
        self.supersense_role = supersense_role
        self.supersense_func = supersense_func
        self.spacy_head_ind = spacy_head_ind
        self.spacy_dep = spacy_dep
        self.ud_head_ind = ud_head_ind
        self.ud_dep = ud_dep
        self.part_of_wmwe = part_of_wmwe
        self.part_of_smwe = part_of_smwe
        self.is_first_mwe_token = is_first_mwe_token
        self.spacy_ner = spacy_ner
        self.ud_upos = ud_upos
        self.ud_xpos = ud_xpos
        self.spacy_pos = spacy_pos
        self.ud_lemma = ud_lemma

        if (self.supersense_role is not None) != (self.supersense_role is not None):
            raise Exception("TaggedToken initialized with only one supersense")

        if self.is_part_of_mwe and not self.is_first_mwe_token:
            self.supersense_func = None
            self.supersense_role = None

        self.supersense_role_type = supersenses.get_supersense_type(self.supersense_role) if self.supersense_role else None
        self.supersense_func_type = supersenses.get_supersense_type(self.supersense_func) if self.supersense_func else None

        supersense_combined = None
        if self.supersense_role and self.supersense_func:
            supersense_combined = self.supersense_role + '|' + self.supersense_func
        self.supersense_combined = supersense_combined

        self.is_part_of_mwe = self.part_of_smwe or self.part_of_wmwe

class StreusleRecord:

    def __init__(self,
                 id,
                 sentence,
                 data,
                 spacy_dep_tree,
                 spacy_ners,
                 spacy_pos,
                 ud_dep_tree,
                 only_supersenses=None):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.data = data
        self.spacy_dep_tree = spacy_dep_tree
        self.ud_dep_tree = ud_dep_tree
        self.ignored_supersenses = []

        if not only_supersenses:
            only_supersenses = supersenses.SUPERSENSES_SET

        def filter_supersense(ss):
            if ss in only_supersenses:
                return ss
            if ss is not None:
                self.ignored_supersenses.append(ss)
            return None

        def extract_supersense_pair(ss1, ss2):
            def process(ss):
                if ss in ['_', '??', '`$']:
                    return None
                return filter_supersense(ss.split('.')[1] if ss != '_' else None)
            ss1 = process(ss1)
            ss2 = process(ss2)
            if ss1 and not ss2:
                ss2 = ss1
            assert all([ss1, ss2]) or not any([ss1, ss2])
            return [ss1, ss2]

        self.tagged_tokens = [
            TaggedToken(
                token=tok_data['word'],
                ind=i,
                token_word2vec=W2V.get(tok_data['word']),
                ud_upos=tok_data['upos'],
                ud_xpos=tok_data['xpos'],
                spacy_pos=spacy_pos[i] if spacy_pos else None,
                ud_lemma=tok_data['lemma'],
                spacy_head_ind=self.spacy_dep_tree[i].head_ind if self.spacy_dep_tree else None,
                spacy_dep=self.spacy_dep_tree[i].dep if self.spacy_dep_tree else None,
                ud_head_ind=self.ud_dep_tree[i].head_ind if self.ud_dep_tree else None,
                ud_dep=self.ud_dep_tree[i].dep if self.ud_dep_tree else None,
                spacy_ner=spacy_ners[i] if spacy_ners else None,
                supersense_role=extract_supersense_pair(tok_data['ss'], tok_data['ss2'])[0],
                supersense_func=extract_supersense_pair(tok_data['ss'], tok_data['ss2'])[1],
                part_of_smwe=self.data['smwes'].get(i+1) is not None,
                part_of_wmwe=self.data['wmwes'].get(i+1) is not None,
                is_first_mwe_token=(self.data['smwes'].get(i + 1) or self.data['wmwes'].get(i+1) or {'id': None})['id'] == tok_data['id']
            ) for i, tok_data in enumerate(self.data['toks'])
        ]
        self.pss_tokens = [x for x in self.tagged_tokens if x.supersense_func in supersenses.PREPOSITION_SUPERSENSES_SET or x.supersense_role in supersenses.PREPOSITION_SUPERSENSES_SET]


class StreusleLoader(object):

    def __init__(self):
        pass

    def load(self, only_with_supersenses=supersenses.PREPOSITION_SUPERSENSES_SET):
        streusle_file = os.path.join(STREUSLE_DIR, 'streusle.conllulex')
        print('Loading streusle data from ' + streusle_file)
        records = []
        with open(streusle_file, 'r') as f:
            sents = list(conllulex2json.load_sents(f))
            for sent in sents:
                record = StreusleRecord(id=sent['streusle_sent_id'],
                                        sentence=sent['text'],
                                        data=sent,
                                        ud_dep_tree=UD_DEP_TREES.get(sent['streusle_sent_id']),
                                        spacy_dep_tree=SPACY_DEP_TREES.get(sent['streusle_sent_id']),
                                        spacy_ners=SPACY_NERS.get(sent['streusle_sent_id']),
                                        spacy_pos=SPACY_POS.get(sent['streusle_sent_id']),
                                        only_supersenses=only_with_supersenses)
                assert not SPACY_DEP_TREES or SPACY_DEP_TREES.get(sent['streusle_sent_id'])
                assert not SPACY_NERS or SPACY_NERS.get(sent['streusle_sent_id'])
                # assert not SPACY_POS or SPACY_POS.get(sent['streusle_sent_id'])
                assert not UD_DEP_TREES or UD_DEP_TREES.get(sent['streusle_sent_id'])
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

