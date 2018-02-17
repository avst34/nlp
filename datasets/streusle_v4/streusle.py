import os
import copy
import sys
import csv
import json
from itertools import chain
from pprint import pprint
from collections import namedtuple, defaultdict
import supersense_repo
from vocabulary import Vocabulary, VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'release')
sys.path.append(STREUSLE_DIR)
print(STREUSLE_DIR)
from .release import conllulex2json

ENHANCEMENTS = namedtuple('SEnhancements', ['WORD2VEC_PATH', 'WORD2VEC_MISSING_PATH', 'UD_LEMMAS_WORD2VEC_PATH', 'UD_LEMMAS_WORD2VEC_MISSING_PATH', 'SPACY_DEP_TREES', 'CORENLP_DEP_TREES', 'SPACY_NERS', 'CORENLP_NERS', 'SPACY_POS', 'CORENLP_POS', 'TRAIN_SET_SENTIDS_UD_SPLIT', 'DEV_SET_SENTIDS_UD_SPLIT', 'TEST_SET_SENTIDS_UD_SPLIT', 'UD_DEP_TREES', 'SPACY_LEMMAS', 'CORENLP_LEMMAS', 'SPACY_LEMMAS_WORD2VEC_PATH', 'CORENLP_LEMMAS_WORD2VEC_PATH', 'SPACY_LEMMAS_WORD2VEC_MISSING_PATH', 'CORENLP_LEMMAS_WORD2VEC_MISSING_PATH', 'STANFORD_CORE_NLP_OUTPUT', 'HEURISTIC_GOVOBJ'])(
    WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'word2vec.pickle'),
    WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'word2vec_missing.json'),
    UD_LEMMAS_WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'ud_lemmas_word2vec.pickle'),
    UD_LEMMAS_WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'ud_lemmas_word2vec_missing.json'),
    SPACY_DEP_TREES=os.path.join(STREUSLE_DIR, 'spacy_dep_trees.json'),
    SPACY_NERS=os.path.join(STREUSLE_DIR, 'spacy_ners.json'),
    SPACY_POS=os.path.join(STREUSLE_DIR, 'spacy_pos.json'),
    SPACY_LEMMAS=os.path.join(STREUSLE_DIR, 'spacy_lemmas.json'),
    SPACY_LEMMAS_WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'spacy_lemmas_word2vec.pickle'),
    SPACY_LEMMAS_WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'spacy_lemmas_word2vec_missing.json'),
    CORENLP_DEP_TREES=os.path.join(STREUSLE_DIR, 'corenlp_dep_trees.json'),
    CORENLP_NERS=os.path.join(STREUSLE_DIR, 'corenlp_ners.json'),
    CORENLP_POS=os.path.join(STREUSLE_DIR, 'corenlp_pos.json'),
    CORENLP_LEMMAS=os.path.join(STREUSLE_DIR, 'corenlp_lemmas.json'),
    CORENLP_LEMMAS_WORD2VEC_PATH=os.path.join(STREUSLE_DIR, 'corenlp_lemmas_word2vec.pickle'),
    CORENLP_LEMMAS_WORD2VEC_MISSING_PATH=os.path.join(STREUSLE_DIR, 'corenlp_lemmas_word2vec_missing.json'),
    STANFORD_CORE_NLP_OUTPUT=os.path.join(STREUSLE_DIR, 'streusle.corenlp'),
    UD_DEP_TREES=os.path.join(STREUSLE_DIR, 'ud_dep_trees.json'),
    TRAIN_SET_SENTIDS_UD_SPLIT=os.path.join(STREUSLE_DIR, 'ud_train_sent_ids.txt'),
    DEV_SET_SENTIDS_UD_SPLIT=os.path.join(STREUSLE_DIR, 'ud_dev_sent_ids.txt'),
    TEST_SET_SENTIDS_UD_SPLIT=os.path.join(STREUSLE_DIR, 'ud_test_sent_ids.txt'),
    HEURISTIC_GOVOBJ=os.path.join(STREUSLE_DIR, 'heuristic_govobj.json')
)

def sentid_to_streusle_id(sent_id):
    # reviews-046906-0001
    _, id1, id2 = sent_id.split('-')
    while id2.startswith('0'):
        id2 = id2[1:]
    return "ewtb.r." + id1 + "." + id2

def load_word2vec(path):
    w2v = Word2VecModel({})
    if os.path.exists(path):
        with open(path, 'rb')as f:
            w2v = Word2VecModel.load(f)
    return w2v

W2V = load_word2vec(ENHANCEMENTS.WORD2VEC_PATH)
UD_LEMMAS_W2V = load_word2vec(ENHANCEMENTS.UD_LEMMAS_WORD2VEC_PATH)
SPACY_LEMMAS_W2V = load_word2vec(ENHANCEMENTS.SPACY_LEMMAS_WORD2VEC_PATH)
CORENLP_LEMMAS_W2V = load_word2vec(ENHANCEMENTS.CORENLP_LEMMAS_WORD2VEC_PATH)

TreeNode = namedtuple('TreeNode', ['id', 'head_id', 'dep'])
def load_dep_tree(tree_json_path):
    if os.path.exists(tree_json_path):
        with open(tree_json_path, 'r') as f:
            return {
                rec_id: {node[0]: TreeNode(id=node[0], head_id=node[1], dep=node[2]) for node in tree_nodes.values()}
                for rec_id, tree_nodes in json.load(f).items()
            }
    else:
        return {}

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            obj = json.load(f)
        for k, v in obj.items():
            obj[k] = {int(k1): v1 for k1, v1 in v.items()}
    else:
        obj = default
    return obj


SPACY_DEP_TREES = load_dep_tree(ENHANCEMENTS.SPACY_DEP_TREES)
UD_DEP_TREES = load_dep_tree(ENHANCEMENTS.UD_DEP_TREES)
CORENLP_DEP_TREES = load_dep_tree(ENHANCEMENTS.CORENLP_DEP_TREES)
SPACY_NERS = load_json(ENHANCEMENTS.SPACY_NERS, {})
CORENLP_NERS = load_json(ENHANCEMENTS.CORENLP_NERS, {})
SPACY_LEMMAS = load_json(ENHANCEMENTS.SPACY_LEMMAS, {})
CORENLP_LEMMAS = load_json(ENHANCEMENTS.CORENLP_LEMMAS, {})
SPACY_POS = load_json(ENHANCEMENTS.SPACY_POS, {})
CORENLP_POS = load_json(ENHANCEMENTS.CORENLP_POS, {})

class TaggedToken:
    def __init__(self, token,
                 ind,
                 token_word2vec,
                 supersense_role,
                 supersense_func,
                 ud_head_ind, ud_dep,
                 is_part_of_wmwe,
                 is_part_of_smwe,
                 is_first_mwe_token,
                 we_toknums,
                 ner,
                 ud_upos, ud_xpos,
                 lemma,
                 ud_id,
                 gov_ind, obj_ind, govobj_config, lexcat, _raw_ss_ss2):
        self.lexcat = lexcat
        self.lemma = lemma
        self.ner = ner
        self.govobj_config = govobj_config
        self.obj_ind = obj_ind
        self.gov_ind = gov_ind
        self.we_toknums = we_toknums
        self.ud_id = ud_id
        self.token = token
        self.ind = ind
        self.token_word2vec = token_word2vec
        self.supersense_role = supersense_role
        self.supersense_func = supersense_func
        self.ud_head_ind = ud_head_ind
        self.ud_dep = ud_dep
        self.is_part_of_wmwe = is_part_of_wmwe
        self.is_part_of_smwe = is_part_of_smwe
        self.is_first_mwe_token = is_first_mwe_token
        self.ud_upos = ud_upos
        self.ud_xpos = ud_xpos
        self._raw_ss_ss2 = _raw_ss_ss2

        if (self.supersense_role is not None) != (self.supersense_role is not None):
            raise Exception("TaggedToken initialized with only one supersense")

        self.is_part_of_mwe = self.is_part_of_smwe or self.is_part_of_wmwe

        if self.is_part_of_mwe and not self.is_first_mwe_token:
            self.supersense_func = None
            self.supersense_role = None

        self.supersense_role_type = supersense_repo.get_supersense_type(self.supersense_role) if self.supersense_role else None
        self.supersense_func_type = supersense_repo.get_supersense_type(self.supersense_func) if self.supersense_func else None

        supersense_combined = None
        if self.supersense_role and self.supersense_func:
            supersense_combined = self.supersense_role + '|' + self.supersense_func
        self.supersense_combined = supersense_combined

        self.identified_for_pss = lexcat in ['P', 'PP', 'INF.P', 'POSS', 'PRON.POSS'] and not any([s in _raw_ss_ss2 for s in ['??', '`$']])

        assert not self.is_part_of_mwe or not(not self.is_first_mwe_token and self.supersense_combined)

ignored_supersenses = []

class StreusleRecord:

    def __init__(self,
                 id,
                 sentence,
                 data,
                 only_supersenses=None):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.data = data

        if not only_supersenses:
            only_supersenses = supersense_repo.SUPERSENSES_SET

        def filter_supersense(ss):
            if ss:
                if "." not in ss:
                    ignored_supersenses.append(ss)
                else:
                    type, ss = ss.split('.')
                    if ss in only_supersenses:
                        return ss
                    else:
                        assert type != 'p'
                        ignored_supersenses.append(ss)
            return None

        def extract_supersense_pair(ss1, ss2):
            def process(ss):
                return filter_supersense(ss)
            ss1 = process(ss1)
            ss2 = process(ss2)
            if ss1 and not ss2:
                ss2 = ss1
            assert all([ss1, ss2]) or not any([ss1, ss2])
            return [ss1, ss2]

        wes = sum([list(data['swes'].values()), list(data['smwes'].values())], [])
        tok_we = {we['toknums'][0]: we for we in wes}
        we_toknums = {we['toknums'][0]: we['toknums'] for we in wes}
        smwes_toknums = sum([we['toknums'] for we in self.data['smwes'].values()], [])
        wmwes_toknums = sum([we['toknums'] for we in self.data['wmwes'].values()], [])
        tok_ss = defaultdict(lambda: (None, None))
        for we in wes:
            pair = extract_supersense_pair(we.get('ss'), we.get('ss2'))
            cur = tok_ss[we['toknums'][0]]
            tok_ss[we['toknums'][0]] = (cur[0] or pair[0], cur[1] or pair[1])

        first_wes_ids = [we['toknums'][0] for we in wes]

        id_to_ind = {tok['#']: ind for ind, tok in enumerate(self.data['toks'])}

        self.tagged_tokens = [
            TaggedToken(
                ud_id=tok_data['#'],
                token=tok_data['word'],
                ind=i,
                token_word2vec=W2V.get(tok_data['word']),
                ud_upos=tok_data['upos'],
                ud_xpos=tok_data['xpos'],
                lemma=tok_data['lemma'],
                ud_head_ind=id_to_ind.get(tok_data['head']),
                ud_dep=tok_data['deprel'],
                ner=tok_data.get('ner'),
                supersense_role=tok_ss[tok_data['#']][0],
                supersense_func=tok_ss[tok_data['#']][1],
                is_part_of_smwe=tok_data['#'] in smwes_toknums,
                is_part_of_wmwe=tok_data['#'] in wmwes_toknums,
                we_toknums=we_toknums.get(tok_data['#']),
                is_first_mwe_token=tok_data['#'] in first_wes_ids,
                gov_ind=id_to_ind.get(tok_we.get(tok_data['#'], {}).get('heuristic_relation', {}).get('gov')),
                obj_ind=id_to_ind.get(tok_we.get(tok_data['#'], {}).get('heuristic_relation', {}).get('obj')),
                govobj_config=tok_we.get(tok_data['#'], {}).get('heuristic_relation', {}).get('config'),
                lexcat=tok_we.get(tok_data['#'], {}).get('lexcat'),
                _raw_ss_ss2=''.join([tok_we.get(tok_data['#'], {}).get(ss) or '' for ss in ['ss', 'ss2']])
            ) for i, tok_data in enumerate(self.data['toks'])
        ]
        self.pss_tokens = [x for x in self.tagged_tokens if x.supersense_func in supersense_repo.PREPOSITION_SUPERSENSES_SET or x.supersense_role in supersense_repo.PREPOSITION_SUPERSENSES_SET]
        assert {t.ud_id for t in self.pss_tokens} == {we['toknums'][0] for we in wes if (we.get('ss') or '').startswith('p.')}

    def get_tok_by_ud_id(self, ud_id):
        return [t for t in self.tagged_tokens if t.ud_id == ud_id][0]

    def build_data_with_supersenses(self, supersenses, ident):
        # supersenses - [(role, func), (role, func), ...]
        assert ident in ['autoid', 'goldid']
        assert len(self.tagged_tokens) == len(supersenses)
        format_supersense = lambda ss: 'p.' + ss if ss else None
        data = copy.deepcopy(self.data)

        for we_type in ['swes', 'smwes']:
            for k, we in data[we_type].items():
                we['we_id'] = k
                we['we_type'] = we_type

        orig = {
            'swes': data['swes'],
            'wmwes': data['wmwes'],
            'smwes': data['smwes']
        }
        data['swes'] = {}
        data['wmwes'] = {}
        data['smwes'] = {}
        for token, (role, func) in zip(self.tagged_tokens, supersenses):
            found_we = None
            wes = chain(orig['swes'].values(), orig['smwes'].values())
            for we in wes:
                if we['toknums'][0] == token.ud_id:
                    found_we = we
            if found_we:
                new_we = {
                    'toknums': found_we['toknums'],
                    'lexcat': found_we.get('lexcat'),
                    'ss': format_supersense(role),
                    'ss2': format_supersense(func)
                }
                data[found_we['we_type']][found_we['we_id']] = new_we
            elif role or func:
                raise Exception("Couldn't find a match for system supersense in data (" + ident + ")")

        for we in chain(data['swes'].values(), data['smwes'].values()):
            assert not self.get_tok_by_ud_id(we['toknums'][0]).identified_for_pss or we['ss'].startswith('p.') and we['ss2'].startswith('p.')

        return data

class StreusleLoader(object):

    def __init__(self):
        pass

    def load(self, conllulex_path, only_with_supersenses=supersense_repo.PREPOSITION_SUPERSENSES_SET, input_format='conllulex'):
        assert input_format in ['conllulex', 'json']
        print('Loading streusle data from ' + conllulex_path)
        records = []
        with open(conllulex_path, 'r', encoding='utf8') as f:
            if input_format == 'conllulex':
                sents = list(conllulex2json.load_sents(f))
            else:
                sents = json.load(f)
            for sent in sents:
                record = StreusleRecord(id=sent['streusle_sent_id'],
                                        sentence=sent['text'],
                                        data=sent,
                                        only_supersenses=only_with_supersenses)
                records.append(record)

        return records

    @staticmethod
    def get_dist(records, all_supersenses=supersense_repo.PREPOSITION_SUPERSENSES_SET):
        return {
            pss: len([tok for rec in records for tok in rec.pss_tokens if tok.supersense == pss])
            for pss in all_supersenses
        }

    # def get_tokens_word2vec_model(self):
    #     return W2V
    #
    # def get_ud_lemmas_word2vec_model(self):
    #     return UD_LEMMAS_W2V
    #
    # def get_spacy_lemmas_word2vec_model(self):
    #     return SPACY_LEMMAS_W2V
    #
    # def get_corenlp_lemmas_word2vec_model(self):
    #     return CORENLP_LEMMAS_W2V


wordVocabularyBuilder = VocabularyBuilder(lambda record: [x.token for x in record.tagged_tokens])
posVocabularyBuilder = VocabularyBuilder(lambda record: [x.pos for x in record.tagged_tokens if x.pos])
ssVocabularyBuilder = VocabularyBuilder(lambda record: [x.supersense for x in record.tagged_tokens if x.supersense])
depsVocabularyBuilder = VocabularyBuilder(lambda record: [x.dep for x in record.tagged_tokens if x.dep])
