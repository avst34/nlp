import json

from models.supersenses import vocabs
from models.supersenses.features import build_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel

hps = LstmMlpSupersensesModel.HyperParameters(
    labels_to_predict=['supersense_role', 'supersense_func'],
    use_token=True,
    use_pos=True,
    use_dep=True,
    deps_from='spacy', # 'spacy' or 'ud'
    pos_from='spacy', # 'spacy' or 'ud'
    use_spacy_ner=True,
    use_prep_onehot=True,
    use_token_internal=True,
    lemmas_from='ud',
    update_lemmas_embd=True,
    update_token_embd=True,
    token_embd_dim=200,
    token_internal_embd_dim=30,
    ud_pos_embd_dim=20,
    spacy_pos_embd_dim=20,
    ud_deps_embd_dim=20,
    spacy_deps_embd_dim=20,
    spacy_ner_embd_dim=20,
    mlp_layers=2,
    mlp_layer_dim=60,
    mlp_activation='tanh',
    lstm_h_dim=50,
    num_lstm_layers=2,
    is_bilstm=True,
    mlp_dropout_p=0.1,
    lstm_dropout_p=0.1,
    epochs=50,
    learning_rate=0.1,
    learning_rate_decay=0.01,
    mask_by='pos:IN,PRP$,RB,TO',
    mask_mwes=True
)

test_sample = LstmMlpSupersensesModel.Sample.from_dict({
  "xs": [
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 3,
      "is_part_of_mwe": True,
      "token": "If",
      "ud_lemma": "if",
      "spacy_lemma": "if",
      "ud_dep": "mark",
      "spacy_pos": "ADP",
      "spacy_ner": None,
      "ud_pos": "IN",
      "ind": 0,
      "spacy_dep": "mark",
      "spacy_head_ind": 2
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "you",
      "ud_lemma": "you",
      "spacy_lemma": "you",
      "ud_dep": "nsubj",
      "spacy_pos": "PRON",
      "spacy_ner": None,
      "ud_pos": "PRP",
      "ind": 1,
      "spacy_dep": "nsubj",
      "spacy_head_ind": 2
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "are",
      "ud_lemma": "be",
      "spacy_lemma": "be",
      "ud_dep": "cop",
      "spacy_pos": "VERB",
      "spacy_ner": None,
      "ud_pos": "VBP",
      "ind": 2,
      "spacy_dep": "advcl",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "serious",
      "ud_lemma": "serious",
      "spacy_lemma": "serious",
      "ud_dep": "advcl",
      "spacy_pos": "ADJ",
      "spacy_ner": None,
      "ud_pos": "JJ",
      "ind": 3,
      "spacy_dep": "acomp",
      "spacy_head_ind": 2
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "about",
      "ud_lemma": "about",
      "spacy_lemma": "about",
      "ud_dep": "mark",
      "spacy_pos": "ADP",
      "spacy_ner": None,
      "ud_pos": "IN",
      "ind": 4,
      "spacy_dep": "prep",
      "spacy_head_ind": 3
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "working",
      "ud_lemma": "work",
      "spacy_lemma": "work",
      "ud_dep": "advcl",
      "spacy_pos": "VERB",
      "spacy_ner": None,
      "ud_pos": "VBG",
      "ind": 5,
      "spacy_dep": "pcomp",
      "spacy_head_ind": 4
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "out",
      "ud_lemma": "out",
      "spacy_lemma": "out",
      "ud_dep": "compound:prt",
      "spacy_pos": "PART",
      "spacy_ner": None,
      "ud_pos": "RP",
      "ind": 6,
      "spacy_dep": "prt",
      "spacy_head_ind": 5
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "in",
      "ud_lemma": "in",
      "spacy_lemma": "in",
      "ud_dep": "case",
      "spacy_pos": "ADP",
      "spacy_ner": None,
      "ud_pos": "IN",
      "ind": 7,
      "spacy_dep": "prep",
      "spacy_head_ind": 5
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "a",
      "ud_lemma": "a",
      "spacy_lemma": "a",
      "ud_dep": "det",
      "spacy_pos": "DET",
      "spacy_ner": None,
      "ud_pos": "DT",
      "ind": 8,
      "spacy_dep": "det",
      "spacy_head_ind": 9
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 10,
      "is_part_of_mwe": False,
      "token": "non-commercial",
      "ud_lemma": "non-commercial",
      "spacy_lemma": "non-commercial",
      "ud_dep": "amod",
      "spacy_pos": "ADJ",
      "spacy_ner": None,
      "ud_pos": "JJ",
      "ind": 9,
      "spacy_dep": "amod",
      "spacy_head_ind": 11
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "like",
      "ud_lemma": "like",
      "spacy_lemma": "like",
      "ud_dep": "amod",
      "spacy_pos": "ADP",
      "spacy_ner": None,
      "ud_pos": "JJ",
      "ind": 10,
      "spacy_dep": "amod",
      "spacy_head_ind": 11
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "atmosphere",
      "ud_lemma": "atmosphere",
      "spacy_lemma": "atmosphere",
      "ud_dep": "obl",
      "spacy_pos": "NOUN",
      "spacy_ner": None,
      "ud_pos": "NN",
      "ind": 11,
      "spacy_dep": "pobj",
      "spacy_head_ind": 7
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "then",
      "ud_lemma": "then",
      "spacy_lemma": "then",
      "ud_dep": "advmod",
      "spacy_pos": "ADV",
      "spacy_ner": None,
      "ud_pos": "RB",
      "ind": 12,
      "spacy_dep": "advmod",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "you",
      "ud_lemma": "you",
      "spacy_lemma": "you",
      "ud_dep": "nsubj",
      "spacy_pos": "PRON",
      "spacy_ner": None,
      "ud_pos": "PRP",
      "ind": 13,
      "spacy_dep": "nsubj",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "have",
      "ud_lemma": "have",
      "spacy_lemma": "have",
      "ud_dep": "aux",
      "spacy_pos": "VERB",
      "spacy_ner": None,
      "ud_pos": "VBP",
      "ind": 14,
      "spacy_dep": "aux",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "chosen",
      "ud_lemma": "choose",
      "spacy_lemma": "choose",
      "ud_dep": "root",
      "spacy_pos": "VERB",
      "spacy_ner": None,
      "ud_pos": "VBN",
      "ind": 15,
      "spacy_dep": "ROOT",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "The",
      "ud_lemma": "the",
      "spacy_lemma": "the",
      "ud_dep": "det",
      "spacy_pos": "DET",
      "spacy_ner": None,
      "ud_pos": "DT",
      "ind": 16,
      "spacy_dep": "det",
      "spacy_head_ind": 18
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "best",
      "ud_lemma": "best",
      "spacy_lemma": "best",
      "ud_dep": "amod",
      "spacy_pos": "ADJ",
      "spacy_ner": None,
      "ud_pos": "JJS",
      "ind": 17,
      "spacy_dep": "amod",
      "spacy_head_ind": 18
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "place",
      "ud_lemma": "place",
      "spacy_lemma": "place",
      "ud_dep": "obj",
      "spacy_pos": "NOUN",
      "spacy_ner": None,
      "ud_pos": "NN",
      "ind": 18,
      "spacy_dep": "dobj",
      "spacy_head_ind": 15
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 20,
      "is_part_of_mwe": False,
      "token": "to",
      "ud_lemma": "to",
      "spacy_lemma": "to",
      "ud_dep": "mark",
      "spacy_pos": "PART",
      "spacy_ner": None,
      "ud_pos": "TO",
      "ind": 19,
      "spacy_dep": "aux",
      "spacy_head_ind": 20
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 18,
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "be",
      "ud_lemma": "be",
      "spacy_lemma": "be",
      "ud_dep": "acl",
      "spacy_pos": "VERB",
      "spacy_ner": "PERSON", # wrong but added here for testing
      "ud_pos": "VB",
      "ind": 20,
      "spacy_dep": "relcl",
      "spacy_head_ind": 18
    },
    {
      "autoid_markable": False,
      "autoid_markable_mwe": False,
      "ud_head_ind": 15,
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": ".",
      "ud_lemma": ".",
      "spacy_lemma": ".",
      "ud_dep": "punct",
      "spacy_pos": "PUNCT",
      "spacy_ner": None,
      "ud_pos": ".",
      "ind": 21,
      "spacy_dep": "punct",
      "spacy_head_ind": 15
    }
  ],
  "sample_id": "ewtb.r.005636.3",
  "ys": [
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": "Topic",
      "supersense_func": "Topic"
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": "Locus",
      "supersense_func": "Locus"
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    },
    {
      "supersense_role": None,
      "supersense_func": None
    }
  ]
})

get_token = lambda ind: test_sample.xs[ind] if ind is not None else None
tokens = lambda inds: [get_token(ind) for ind in inds]

test_sample_ud_parents = tokens([3, 3, 3, 15, 5, 3, 5, 11, 11, 10, 11, 5, 15, 15, 15, None, 18, 18, 15, 20, 18, 15])
test_sample_spacy_parents = tokens([2, 2, 15, 2, 3, 4, 5, 5, 9, 11, 11, 7, 15, 15, 15, None, 18, 18, 15, 20, 18, 15])
test_sample_spacy_grandparents = tokens([15, 15, None, 15, 2, 3, 4, 4, 11, 7, 7, 5, None, None, None, None, 15, 15, None, 18, 15, None])
test_sample_ud_grandparents = tokens([15, 15, 15, None, 3, 15, 3, 5, 5, 11, 5, 3, None, None, None, None, 15, 15, None, 18, 15, None])
test_sample_spacy_pobj_child = tokens([None, None, None, None, None, None, None, 11, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
test_sample_spacy_has_children = [False, False, True, True, True, True, False, True, False, True, False, True, False, False, False, True, False, False, True, False, True, False]
test_sample_capitalized_word_follows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False]

features = build_features(hps)

def both_none_or_attr_eq(v1, obj, attr):
    return v1 is None and obj is None or v1 == getattr(obj, attr)

def test_token_word2vec(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_ud_lemma_word2vec(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.ud_lemma

def test_token_spacy_lemma_word2vec(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.spacy_lemma

def test_token_internal(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.ud_pos

def test_token_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.spacy_pos

def test_token_ud_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.ud_dep

def test_token_spacy_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.spacy_dep

def test_token_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.spacy_ner

def test_prep_onehot(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == (x.token if vocabs.PREPS.has_word(x.token) else None)

def test_ud_parent(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_parents[ind], 'ind')

def test_ud_parent_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_parents[ind], 'ud_pos')

def test_ud_parent_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_parents[ind], 'spacy_pos')

def test_ud_parent_ud_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_parents[ind], 'ud_dep')

def test_ud_parent_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_parents[ind], 'spacy_ner')

def test_ud_grandparent(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_grandparents[ind], 'ind')

def test_ud_grandparent_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_grandparents[ind], 'ud_pos')

def test_ud_grandparent_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_grandparents[ind], 'spacy_pos')

def test_ud_grandparent_ud_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_grandparents[ind], 'ud_dep')

def test_ud_grandparent_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_ud_grandparents[ind], 'spacy_ner')

def test_spacy_parent(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_parents[ind], 'ind')

def test_spacy_parent_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_parents[ind], 'ud_pos')

def test_spacy_parent_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_parents[ind], 'spacy_pos')

def test_spacy_parent_spacy_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_parents[ind], 'spacy_dep')

def test_spacy_parent_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_parents[ind], 'spacy_ner')

def test_spacy_grandparent(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_grandparents[ind], 'ind')

def test_spacy_grandparent_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_grandparents[ind], 'ud_pos')

def test_spacy_grandparent_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_grandparents[ind], 'spacy_pos')

def test_spacy_grandparent_spacy_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_grandparents[ind], 'spacy_dep')

def test_spacy_grandparent_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_grandparents[ind], 'spacy_ner')

def test_spacy_pobj_child(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ind')

def test_spacy_pobj_child_ud_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ud_pos')

def test_spacy_pobj_child_spacy_pos(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'spacy_pos')

def test_spacy_pobj_child_spacy_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'spacy_dep')

def test_spacy_pobj_child_spacy_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'spacy_ner')

def test_spacy_has_children(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == str(test_sample_spacy_has_children[ind])

def test_capitalized_word_follows(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == str(test_sample_capitalized_word_follows[ind])


tests = {
    'token-word2vec': test_token_word2vec,
    'token-internal': test_token_internal,
    'token.ud-lemma-word2vec': test_token_ud_lemma_word2vec,
    'token.spacy-lemma-word2vec': test_token_spacy_lemma_word2vec,
    'token.ud-pos': test_token_ud_pos,
    'token.spacy-pos': test_token_spacy_pos,
    'token.ud-dep': test_token_ud_dep,
    'token.spacy-dep': test_token_spacy_dep,
    'token.spacy-ner': test_token_spacy_ner,

    # 'prep-onehot': test_prep_onehot,
    'capitalized-word-follows': test_capitalized_word_follows,

    'token-ud-parent': test_ud_parent,
    'token-ud-parent.ud-pos': test_ud_parent_ud_pos,
    'token-ud-parent.spacy-pos': test_ud_parent_spacy_pos,
    'token-ud-parent.ud-dep': test_ud_parent_ud_dep,
    'token-ud-parent.spacy-ner': test_ud_parent_spacy_ner,

    'token-ud-grandparent': test_ud_grandparent,
    'token-ud-grandparent.ud-pos': test_ud_grandparent_ud_pos,
    'token-ud-grandparent.spacy-pos': test_ud_grandparent_spacy_pos,
    'token-ud-grandparent.ud-dep': test_ud_grandparent_ud_dep,
    'token-ud-grandparent.spacy-ner': test_ud_grandparent_spacy_ner,

    'token-spacy-parent': test_spacy_parent,
    'token-spacy-parent.ud-pos': test_spacy_parent_ud_pos,
    'token-spacy-parent.spacy-pos': test_spacy_parent_spacy_pos,
    'token-spacy-parent.spacy-dep': test_spacy_parent_spacy_dep,
    'token-spacy-parent.spacy-ner': test_spacy_parent_spacy_ner,

    'token-spacy-grandparent': test_spacy_grandparent,
    'token-spacy-grandparent.ud-pos': test_spacy_grandparent_ud_pos,
    'token-spacy-grandparent.spacy-pos': test_spacy_grandparent_spacy_pos,
    'token-spacy-grandparent.spacy-dep': test_spacy_grandparent_spacy_dep,
    'token-spacy-grandparent.spacy-ner': test_spacy_grandparent_spacy_ner,

    'token-spacy-pobj-child': test_spacy_pobj_child,
    'token-spacy-pobj-child.ud-pos': test_spacy_pobj_child_ud_pos,
    'token-spacy-pobj-child.spacy-pos': test_spacy_pobj_child_spacy_pos,
    'token-spacy-pobj-child.spacy-dep': test_spacy_pobj_child_spacy_dep,
    'token-spacy-pobj-child.spacy-ner': test_spacy_pobj_child_spacy_ner,

    'token-spacy-has-children': test_spacy_has_children
}

def test_features():
    for feature in features.list():
        tests[feature.name](feature)

if __name__ == '__main__':
    test_features()

