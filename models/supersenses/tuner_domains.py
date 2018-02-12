from hyperparameters_tuner import HyperparametersTuner
import numpy as np
import os

PS = HyperparametersTuner.ParamSettings

# For tuning
TUNER_DOMAINS_TUNING = [
    PS(name='labels_to_predict', values=[
        ('supersense_role', 'supersense_func'),
        # ('supersense_role',),
        # ('supersense_func',)
    ]),
    PS(name='use_token', values=[True]),
    PS(name='update_lemmas_embd', values=[True, False]),
    PS(name='use_token_internal', values=[True]),
    PS(name='use_ud_xpos', values=[True]),
    PS(name='use_ud_dep', values=[True]),
    PS(name='use_ner', values=[True]),
    PS(name='use_govobj', values=[True]),
    PS(name='use_prep_onehot', values=[False]),
    PS(name='use_lexcat', values=[True]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='token_internal_embd_dim', values=[10, 25, 50, 100, 300]),
    PS(name='ud_xpos_embd_dim', values=[5, 10, 25]),
    PS(name='ud_deps_embd_dim', values=[5, 10, 25]),
    PS(name='ner_embd_dim', values=[5, 10]),
    PS(name='govobj_config_embd_dim', values=[3]),
    PS(name='lexcat_embd_dim', values=[3]),
    PS(name='update_token_embd', values=[True, False]),
    PS(name='mlp_layers', values=[2]),
    PS(name='mlp_layer_dim', values=[20, 40, 80, 100]),
    PS(name='mlp_activation', values=['tanh', 'cube', 'relu']),
    PS(name='lstm_h_dim', values=[20, 40, 80, 100]),
    PS(name='num_lstm_layers', values=[2]),
    PS(name='is_bilstm', values=[True]),
    PS(name='mlp_dropout_p', values=np.arange(.51, step=.01)),
    PS(name='lstm_dropout_p', values=np.arange(.51, step=.01)),
    PS(name='epochs', values=[80]),
    PS(name='learning_rate', values=np.logspace(-2, 0, 6)),
    PS(name='learning_rate_decay', values=np.r_[0, np.logspace(-5, -1, 9)]),
    PS(name='mask_mwes', values=[False]),
    PS(name='allow_empty_prediction', values=[True, False]),
    PS(name='dynet_random_seed', values=[os.environ['DYNET_RANDOM_SEED']], enabled=True)
]



# For testing - all features
TUNER_DOMAINS_TESTING = [
    PS(name='use_token', values=[True]),
    PS(name='use_ud_xpos', values=[True]),
    PS(name='use_ud_dep', values=[True]),
    PS(name='use_prep_onehot', values=[True]),
    PS(name='use_token_internal', values=[True]),
    PS(name='token_internal_embd_dim', values=[50]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='update_token_embd', values=[True]),
    PS(name='update_pos_embd', values=[True]),
    PS(name='mlp_layers', values=[1,2,3]),
    PS(name='mlp_layer_dim', values=[50]),
    PS(name='mlp_activation', values=['tanh', 'cube', 'relu']),
    PS(name='lstm_h_dim', values=[50]),
    PS(name='num_lstm_layers', values=[2]),
    PS(name='is_bilstm', values=[True]),
    PS(name='mlp_dropout_p', values=[0.1]),
    PS(name='epochs', values=[1]),
    PS(name='validation_split', values=[0.3]),
    PS(name='learning_rate', values=[1]),
    PS(name='learning_rate_decay', values=[0])
]

# TUNER_DOMAINS = TUNER_DOMAINS_TESTING
TUNER_DOMAINS = TUNER_DOMAINS_TUNING

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS_TUNING:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)