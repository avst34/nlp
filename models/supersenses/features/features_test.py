import json
import random
from models.supersenses import vocabs
from models.supersenses.features import build_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel

hps = LstmMlpSupersensesModel.HyperParameters(
    labels_to_predict=['supersense_role', 'supersense_func'],
    use_token=True,
    use_ud_xpos=True,
    use_ud_dep=True,
    use_ner=True,
    use_prep_onehot=True,
    use_govobj=True,
    allow_empty_prediction=True,
    use_token_internal=True,
    update_lemmas_embd=True,
    update_token_embd=True,
    token_embd_dim=200,
    token_internal_embd_dim=30,
    ud_xpos_embd_dim=20,
    ud_deps_embd_dim=20,
    ner_embd_dim=20,
    govobj_config_embd_dim=20,
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
    mask_mwes=True,
    dynet_random_seed=0,
    use_lexcat=True,
    lexcat_embd_dim=10,
)

test_sample = LstmMlpSupersensesModel.Sample.from_dict({
  "xs": [
    {
      "ud_head_ind": 3,
      "is_part_of_mwe": True,
      "token": "If",
      "lemma": "if",
      "ud_dep": "mark",
      "ner": None,
      "ud_xpos": "IN",
      "ud_upos": "IN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 0,
      "gov_ind": 0 + 1,
      "obj_ind": 0 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "you",
      "lemma": "you",
      "ud_dep": "nsubj",
      "ner": None,
      "ud_xpos": "PRP",
      "ud_upos": "PRP",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 1,
      "gov_ind": 1 + 1,
      "obj_ind": 1 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "are",
      "lemma": "be",
      "ud_dep": "cop",
      "ner": None,
      "ud_xpos": "VBP",
      "ud_upos": "VBP",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 2,
      "gov_ind": 2 + 1,
      "obj_ind": 2 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "serious",
      "lemma": "serious",
      "ud_dep": "advcl",
      "ner": None,
      "ud_xpos": "JJ",
      "ud_upos": "JJ",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 3,
      "gov_ind": 3 + 1,
      "obj_ind": 3 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "about",
      "lemma": "about",
      "ud_dep": "mark",
      "ner": None,
      "ud_xpos": "IN",
      "ud_upos": "IN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 4,
      "gov_ind": 4 + 1,
      "obj_ind": 4 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 3,
      "is_part_of_mwe": False,
      "token": "working",
      "lemma": "work",
      "ud_dep": "advcl",
      "ner": None,
      "ud_xpos": "VBG",
      "ud_upos": "VBG",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 5,
      "gov_ind": 5 + 1,
      "obj_ind": 5 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "out",
      "lemma": "out",
      "ud_dep": "compound:prt",
      "ner": None,
      "ud_xpos": "RP",
      "ud_upos": "RP",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 6,
      "gov_ind": 6 + 1,
      "obj_ind": 6 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "in",
      "lemma": "in",
      "ud_dep": "case",
      "ner": None,
      "ud_xpos": "IN",
      "ud_upos": "IN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 7,
      "gov_ind": 7 + 1,
      "obj_ind": 7 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "a",
      "lemma": "a",
      "ud_dep": "det",
      "ner": None,
      "ud_xpos": "DT",
      "ud_upos": "DT",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 8,
      "gov_ind": 8 + 1,
      "obj_ind": 8 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 10,
      "is_part_of_mwe": False,
      "token": "non-commercial",
      "lemma": "non-commercial",
      "ud_dep": "amod",
      "ner": None,
      "ud_xpos": "JJ",
      "ud_upos": "JJ",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 9,
      "gov_ind": 9 + 1,
      "obj_ind": 9 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 11,
      "is_part_of_mwe": False,
      "token": "like",
      "lemma": "like",
      "ud_dep": "amod",
      "ner": None,
      "ud_xpos": "JJ",
      "ud_upos": "JJ",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 10,
      "gov_ind": 10 + 1,
      "obj_ind": 10 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 5,
      "is_part_of_mwe": False,
      "token": "atmosphere",
      "lemma": "atmosphere",
      "ud_dep": "obl",
      "ner": None,
      "ud_xpos": "NN",
      "ud_upos": "NN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 11,
      "gov_ind": 11 + 1,
      "obj_ind": 11 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "then",
      "lemma": "then",
      "ud_dep": "advmod",
      "ner": None,
      "ud_xpos": "RB",
      "ud_upos": "RB",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 12,
      "gov_ind": 12 + 1,
      "obj_ind": 12 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "you",
      "lemma": "you",
      "ud_dep": "nsubj",
      "ner": None,
      "ud_xpos": "PRP",
      "ud_upos": "PRP",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 13,
      "gov_ind": 13 + 1,
      "obj_ind": 13 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "have",
      "lemma": "have",
      "ud_dep": "aux",
      "ner": None,
      "ud_xpos": "VBP",
      "ud_upos": "VBP",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 14,
      "gov_ind": 14 + 1,
      "obj_ind": 14 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "chosen",
      "lemma": "choose",
      "ud_dep": "root",
      "ner": None,
      "ud_xpos": "VBN",
      "ud_upos": "VBN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 15,
      "gov_ind": 15 + 1,
      "obj_ind": 15 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "The",
      "lemma": "the",
      "ud_dep": "det",
      "ner": None,
      "ud_xpos": "DT",
      "ud_upos": "DT",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 16,
      "gov_ind": 16 + 1,
      "obj_ind": 16 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "best",
      "lemma": "best",
      "ud_dep": "amod",
      "ner": None,
      "ud_xpos": "JJS",
      "ud_upos": "JJS",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 17,
      "gov_ind": 17 + 1,
      "obj_ind": 17 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": "place",
      "lemma": "place",
      "ud_dep": "obj",
      "ner": None,
      "ud_xpos": "NN",
      "ud_upos": "NN",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 18,
      "gov_ind": 18 + 1,
      "obj_ind": 18 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 20,
      "is_part_of_mwe": False,
      "token": "to",
      "lemma": "to",
      "ud_dep": "mark",
      "ner": None,
      "ud_xpos": "TO",
      "ud_upos": "TO",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 19,
      "gov_ind": 19 + 1,
      "obj_ind": 19 + 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 18,
      "is_part_of_mwe": False,
      "token": "be",
      "lemma": "be",
      "ud_dep": "acl",
      "ner": "PERSON", # wrong but added here for testing,
      "ud_xpos": "VB",
      "ud_upos": "VB",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 20,
      "gov_ind": 20 + 1,
      "obj_ind": 20 - 1,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
    },
    {
      "ud_head_ind": 15,
      "is_part_of_mwe": False,
      "token": ".",
      "lemma": ".",
      "ud_dep": "punct",
      "ner": None,
      "ud_xpos": ".",
      "ud_upos": ".",
      "identified_for_pss": True,
      "lexcat": 'P',
      "ind": 21,
      "gov_ind": 21 - 1,
      "obj_ind": 21 - 2,
      "govobj_config": random.choice(['subordinating', 'predicative', 'default']),
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

test_sample_parents = tokens([3, 3, 3, 15, 5, 3, 5, 11, 11, 10, 11, 5, 15, 15, 15, None, 18, 18, 15, 20, 18, 15]),
test_sample_govs = tokens([15, 15, 15, None, 3, 15, 3, 5, 5, 11, 5, 3, None, None, None, None, 15, 15, None, 18, 15, None])
test_sample_spacy_pobj_child = tokens([None, None, None, None, None, None, None, 11, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
# test_sample_has_children = {
#     'spacy': [False, False, True, True, True, True, False, True, False, True, False, True, False, False, False, True, False, False, True, False, True, False]
# }
test_sample_capitalized_word_follows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False]

features = build_features(hps)

gettok = lambda sent, ind: sent[ind] if ind is not None and ind < len(sent) else None

def both_none_or_attr_eq(v1, obj, attr):
    return v1 is None and obj is None or v1 == getattr(obj, attr)

def test_token_word2vec(feature):    
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_lemma_word2vec(feature):      
      for ind, x in enumerate(test_sample.xs):
          assert feature.extract(x, test_sample.xs) == x.lemma

def test_token_internal(feature):    
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_ud_xpos(feature):
        for ind, x in enumerate(test_sample.xs):
          assert feature.extract(x, test_sample.xs) == x.ud_xpos

def test_token_dep(feature):
        for ind, x in enumerate(test_sample.xs):
            assert feature.extract(x, test_sample.xs) == x.ud_dep

def test_token_ner(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.ner

def test_token_lexcat(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.lexcat

def test_govobj_config(feature):
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == x.govobj_config

# def test_prep_onehot(featur_namee):
#     for ind, x in enumerate(test_sample.xs):
#         assert feature.extract(x, test_sample.xs) == (x.token if vocabs.PREPS.has_word(x.token) else None)
#
def test_obj(feature):
        for ind, x in enumerate(test_sample.xs):
              assert feature.extract(x, test_sample.xs) == x.obj_ind
            
def test_obj_ud_xpos(feature):
        for ind, x in enumerate(test_sample.xs):
              assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.obj_ind), 'ud_upos')

def test_obj_dep(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.obj_ind), 'ud_dep')

def test_obj_ner(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.obj_ind), 'ner')

def test_gov(feature):
      for ind, x in enumerate(test_sample.xs):
            assert feature.extract(x, test_sample.xs) == x.gov_ind

def test_gov_ud_xpos(feature):
        for ind, x in enumerate(test_sample.xs):
          assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.gov_ind), 'ud_upos')

def test_gov_dep(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.gov_ind), 'ud_dep')

def test_gov_ner(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), gettok(test_sample.xs, x.gov_ind), 'ner')

def test_spacy_pobj_child(feature):
  for ind, x in enumerate(test_sample.xs):
      assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ind')

def test_spacy_pobj_child_ud_xpos(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ud_xpos')

def test_spacy_pobj_child_dep(feature):
    for ind, x in enumerate(test_sample.xs):
        assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ud_dep')

def test_spacy_pobj_child_ner(feature):
        for ind, x in enumerate(test_sample.xs):
            assert both_none_or_attr_eq(feature.extract(x, test_sample.xs), test_sample_spacy_pobj_child[ind], 'ner')

def test_has_children(feature):
        for ind, x in enumerate(test_sample.xs):
            assert feature.extract(x, test_sample.xs) == str(test_sample_has_children[ind])

def test_capitalized_word_follows(feature):    
    for ind, x in enumerate(test_sample.xs):
        assert feature.extract(x, test_sample.xs) == str(test_sample_capitalized_word_follows[ind])


tests = {
    'token-word2vec': test_token_word2vec,
    'token-internal': test_token_internal,
    'token.lemma-word2vec': test_token_lemma_word2vec,
    'token.ud_xpos': test_token_ud_xpos,
    'token.dep': test_token_dep,
    'token.ner': test_token_ner,
    'token.govobj-config': test_govobj_config,
    'token.lexcat': test_token_lexcat,

    # 'prep-onehot': test_prep_onehot,
    'capitalized-word-follows': test_capitalized_word_follows,

    'token-obj': test_obj,
    'token-obj.ud_xpos': test_obj_ud_xpos,
    'token-obj.dep': test_obj_dep,
    'token-obj.ner': test_obj_ner,

    'token-gov': test_gov,
    'token-gov.ud_xpos': test_gov_ud_xpos,
    'token-gov.dep': test_gov_dep,
    'token-gov.ner': test_gov_ner,

    # 'token-spacy-pobj-child': test_spacy_pobj_child,
    # 'token-spacy-pobj-child.ud_xpos': test_spacy_pobj_child_ud_xpos,
    # 'token-spacy-pobj-child.dep': test_spacy_pobj_child_dep,
    # 'token-spacy-pobj-child.ner': test_spacy_pobj_child_ner,
    #
    # 'token-has-children': test_has_children
}

def test_features():
  for feature in features.list():
    tests[feature.name](feature)

if __name__ == '__main__':
    test_features()

