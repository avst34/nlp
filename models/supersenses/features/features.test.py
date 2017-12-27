from models.supersenses.features import build_features
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel

train_records, dev_records, test_records = streusle_loader.load()

hps = LstmMlpSupersensesModel.HyperParameters(
    labels_to_predict=['supersense_role', 'supersense_func'],
    use_token=True,
    use_pos=True,
    use_dep=True,
    deps_from='spacy', # 'spacy' or 'ud'
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

features = build_features(hps)


