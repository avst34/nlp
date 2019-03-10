import copy
import json
import os
import sys
from collections import namedtuple, defaultdict
from functools import reduce
from itertools import chain

import h5py

import supersense_repo
from models.supersenses.preprocessing.elmo import run_elmo
from vocabulary import VocabularyBuilder
from word2vec import Word2VecModel

STREUSLE_DIR = os.path.join(os.path.dirname(__file__), 'release')
ELMO_FILE = os.path.join(os.path.dirname(__file__), 'elmo/elmo_layers.hdf5')
sys.path.append(STREUSLE_DIR)

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
                 noun_ss,
                 verb_ss,
                 ud_head_ind,
                 ud_grandparent_ind_override,
                 ud_dep,
                 is_part_of_wmwe,
                 is_part_of_smwe,
                 first_we_token_id,
                 is_first_mwe_token,
                 we_toknums,
                 ner,
                 ud_upos, ud_xpos,
                 lemma,
                 ud_id,
                 prep_toks,
                 gov_ind, obj_ind, govobj_config, lexcat, _raw_ss_ss2, elmo=None, hidden=False):
        self.first_we_token_id = first_we_token_id
        self.ud_grandparent_ind_override = ud_grandparent_ind_override
        self.hidden = hidden
        self.elmo = elmo
        self.verb_ss = verb_ss
        self.noun_ss = noun_ss
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
        self.prep_toks = prep_toks

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
        assert not self.is_first_mwe_token or self.first_we_token_id == self.ud_id
        assert not self.is_part_of_mwe or self.first_we_token_id is not None

ignored_supersenses = []

class StreusleRecord:

    def __init__(self,
                 id,
                 sentence,
                 data,
                 only_supersenses=None,
                 load_elmo=False):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.data = data
        self.load_elmo = load_elmo

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
            cur = tok_ss[int(we['toknums'][0])]
            tok_ss[int(we['toknums'][0])] = (cur[0] or pair[0], cur[1] or pair[1])

        first_wes_ids = [we['toknums'][0] for we in wes]
        first_tok_ids = {int(toknum): int(we['toknums'][0]) for we in wes for toknum in we['toknums']}
        id_to_ind = {int(tok['#']): ind for ind, tok in enumerate(self.data['toks'])}

        if self.load_elmo:
            _elmo_embeddings = run_elmo([tok_data['word'] for tok_data in self.data['toks'] if not tok_data.get('hidden')])
            elmo_embeddings = []
            eind = 0
            for tok in self.data['toks']:
                if tok.get('hidden'):
                    elmo_embeddings.append(None)
                else:
                    elmo_embeddings.append(_elmo_embeddings[eind])
                    eind += 1
            assert eind == len(_elmo_embeddings), (eind, len(_elmo_embeddings))
        else:
            elmo_embeddings = [None for _ in self.data['toks']]

        assert len(elmo_embeddings) == len(self.data['toks'])

        self.tagged_tokens = [
            TaggedToken(
                ud_id=int(tok_data['#']),
                token=tok_data['word'],
                ind=i,
                token_word2vec=W2V.get(tok_data['word']),
                ud_upos=tok_data['upos'],
                ud_xpos=tok_data['xpos'],
                lemma=tok_data['lemma'],
                ud_head_ind=id_to_ind.get(tok_data['head']),
                ud_grandparent_ind_override=id_to_ind.get(tok_data.get('grandparent_override')),
                ud_dep=tok_data['deprel'],
                ner=tok_data.get('ner'),
                supersense_role=tok_ss[int(tok_data['#'])][0],
                supersense_func=tok_ss[int(tok_data['#'])][1],
                noun_ss=None,
                verb_ss=None,
                is_part_of_smwe=int(tok_data['#']) in smwes_toknums,
                is_part_of_wmwe=int(tok_data['#']) in wmwes_toknums,
                first_we_token_id=first_tok_ids[int(tok_data['#'])] if int(tok_data['#']) in first_tok_ids else None,
                we_toknums=we_toknums.get(int(tok_data['#'])),
                is_first_mwe_token=int(tok_data['#']) in first_wes_ids,
                gov_ind=id_to_ind.get(tok_we.get(int(tok_data['#']), {}).get('heuristic_relation', {}).get('gov')),
                obj_ind=id_to_ind.get(tok_we.get(int(tok_data['#']), {}).get('heuristic_relation', {}).get('obj')),
                govobj_config=tok_we.get(int(tok_data['#']), {}).get('heuristic_relation', {}).get('config'),
                lexcat=tok_we.get(int(tok_data['#']), {}).get('lexcat'),
                _raw_ss_ss2=''.join([tok_we.get(int(tok_data['#']), {}).get(ss) or '' for ss in ['ss', 'ss2']]),
                prep_toks=[self.data['toks'][id_to_ind[tokid]]['word'] for tokid in we_toknums.get(int(tok_data['#']), [])],
                elmo=elmo_embeddings[i],
                hidden=tok_data.get('hidden')
            ) for i, tok_data in enumerate(self.data['toks'])
        ]
        self.pss_tokens = [x for x in self.tagged_tokens if x.supersense_func in supersense_repo.PREPOSITION_SUPERSENSES_SET or x.supersense_role in supersense_repo.PREPOSITION_SUPERSENSES_SET]
        assert {t.ud_id for t in self.pss_tokens} == {we['toknums'][0] for we in wes if (we.get('ss') or '').startswith('p.')}

    def tokens(self):
        return [tok.token for tok in self.tagged_tokens]

    def get_tok_by_ud_id(self, ud_id):
        return [t for t in self.tagged_tokens if t.ud_id == ud_id][0]

    def build_data_with_supersenses(self, supersenses, ident, supersenses_dists=None):
        # supersenses - [(role, func), (role, func), ...]
        # supersenses dists - [(role dist, func dist), (role dist, func dist), ...]
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
        for token, (role, func), (role_dist, func_dist) in zip(self.tagged_tokens, supersenses, supersenses_dists or [(None, None)] * len(supersenses)):
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
                    'ss2': format_supersense(func),
                    'ss_dist': {format_supersense(r): p for r, p in role_dist.items()} if role_dist else None,
                    'ss2_dist': {format_supersense(f): p for f, p in func_dist.items()} if func_dist else None,
                }
                data[found_we['we_type']][found_we['we_id']] = new_we
            elif role or func:
                raise Exception("Couldn't find a match for system supersense in data (" + ident + ")")

        is_pss = lambda pss: pss is not None and pss.startswith('p.')

        for we in chain(data['swes'].values(), data['smwes'].values()):
            assert not self.get_tok_by_ud_id(we['toknums'][0]).identified_for_pss or is_pss(we['ss']) or is_pss(we['ss2'])

        return data

class StreusleLoader(object):

    def __init__(self, load_elmo=False, syntax='gold', ident='gold', task_name=None):
        self.load_elmo = load_elmo
        self.syntax = syntax
        self.ident = ident
        if task_name:
            self.syntax = None
            self.ident = None
            if 'goldid' in task_name:
                self.ident = 'gold'
            if 'autoid' in task_name:
                self.ident = 'auto'
            if 'goldsyn' in task_name:
                self.syntax = 'gold'
            if 'autosyn' in task_name:
                self.syntax = 'auto'

        assert syntax in ['gold', 'auto']
        assert ident in ['gold', 'auto']

    def load(self, conllulex_path=STREUSLE_DIR + '/streusle.conllulex', only_with_supersenses=supersense_repo.PREPOSITION_SUPERSENSES_SET, input_format='conllulex'):
        assert input_format in ['conllulex', 'json']
        print('Loading streusle data from ' + conllulex_path)
        records = []
        with open(conllulex_path, 'r', encoding='utf8') as f:
            if input_format == 'conllulex':
                sents = list(conllulex2json.load_sents(f))
            else:
                sents = json.load(f)
            for sent_ind, sent in enumerate(sents):
                # print("Loading %d/%d" % (sent_ind, len(sents)))
                # sent_txt = ' '.join(tok['word'] for tok in sent['toks'])
                record = StreusleRecord(id=sent['streusle_sent_id'],
                                        sentence=sent['text'],
                                        data=sent,
                                        only_supersenses=only_with_supersenses,
                                        load_elmo=self.load_elmo
                                        )
                records.append(record)

        return records

    def load_train(self):
        return self.load(STREUSLE_DIR + '/train/streusle.ud_train.' + self.ident + 'id.' + self.syntax + 'syn.json', input_format='json')

    def load_dev(self):
        return self.load(STREUSLE_DIR + '/dev/streusle.ud_dev.' + self.ident + 'id.' + self.syntax + 'syn.json', input_format='json')

    def load_test(self):
        return self.load(STREUSLE_DIR + '/test/streusle.ud_test.' + self.ident + 'id.' + self.syntax + 'syn.json', input_format='json')

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
