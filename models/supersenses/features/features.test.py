import json

from models.supersenses.features import build_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel

train_records, dev_records, test_records = streusle_loader.load()

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
    update_token_embd=True,
    token_embd_dim=200,
    token_internal_embd_dim=30,
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
    mask_by='pos:IN,PRP$,RB,TO' # MASK_BY_SAMPLE_YS or MASK_BY_POS_PREFIX + 'pos1,pos2,...'
)

test_sample = LstmMlpSupersensesModel.Sample.from_dict({..})

features = build_features(hps)

def test_token_word2vec(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_internal(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.token

def test_token_ud_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.ud_pos

def test_token_spacy_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.spacy_pos

def test_token_ud_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.ud_dep

def test_token_spacy_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.spacy_dep

def test_token_spacy_ner(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.spacy_ner

def test_prep_onehot(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == x.token

def test_ud_parent(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_parents[ind].ind

def test_ud_parent_ud_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_parents[ind].ud_pos

def test_ud_parent_spacy_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_parents[ind].spacy_pos

def test_ud_parent_ud_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_parents[ind].ud_dep

def test_ud_parent_spacy_ner(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_parents[ind].spacy_ner

def test_ud_grandparent(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_grandparents[ind].ind

def test_ud_grandparent_ud_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_grandparents[ind].ud_pos

def test_ud_grandparent_spacy_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_grandparents[ind].spacy_pos

def test_ud_grandparent_ud_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_grandparents[ind].ud_dep

def test_ud_grandparent_spacy_ner(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_ud_grandparents[ind].spacy_ner

def test_spacy_parent(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_parents[ind].ind

def test_spacy_parent_ud_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_parents[ind].ud_pos

def test_spacy_parent_spacy_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_parents[ind].spacy_pos

def test_spacy_parent_spacy_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_parents[ind].spacy_dep

def test_spacy_parent_spacy_ner(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_parents[ind].spacy_ner

def test_spacy_pobj_child(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_pobj_child[ind].ind

def test_spacy_pobj_child_ud_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_pobj_child[ind].ud_pos

def test_spacy_pobj_child_spacy_pos(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_pobj_child[ind].spacy_pos

def test_spacy_pobj_child_spacy_dep(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_pobj_child[ind].spacy_dep

def test_spacy_pobj_child_spacy_ner(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs).ind == test_sample_spacy_pobj_child[ind].spacy_ner

def test_spacy_has_children(feature):
    for x in test_sample.xs:
        assert feature.extract(x, test_sample.xs) == test_sample_spacy_has_children[ind]

tests = {
    'token-word2vec': test_token_word2vec,
    'token-internal': test_token_internal,
    'token.ud-pos': test_token_ud_pos,
    'token.spacy-pos': test_token_spacy_pos,
    'token.ud-dep': test_token_ud_dep,
    'token.spacy-dep': test_token_spacy_dep,
    'token.spacy-ner': test_token_spacy_ner,

    'prep-onehot': test_prep_onehot,

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
    'token-spacy-parent.ud-dep': test_spacy_parent_spacy_dep,
    'token-spacy-parent.spacy-ner': test_spacy_parent_spacy_ner,

    'token-spacy-pobj-child': test_spacy_pobj_child,
    'token-spacy-pobj-child.ud-pos': test_spacy_pobj_child_ud_pos,
    'token-spacy-pobj-child.spacy-pos': test_spacy_pobj_child_spacy_pos,
    'token-spacy-pobj-child.spacy-dep': test_spacy_pobj_child_spacy_dep,
    'token-spacy-pobj-child.spacy-ner': test_spacy_pobj_child_spacy_ner,

    'token-spacy-has-children': test_spacy_has_children
}

for feature in features:
    tests[feature.name](feature)





