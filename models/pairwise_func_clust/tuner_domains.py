import os

import numpy as np

from hyperparameters_tuner import HyperparametersTuner

PS = HyperparametersTuner.ParamSettings

# For tuning
TUNER_DOMAINS = [
    PS(name='use_prep', values=[True, False]),
    PS(name='use_gov', values=[True, False]),
    PS(name='use_obj', values=[True, False]),
    PS(name='use_role', values=[True, False]),
    PS(name='use_ud_dep', values=[True, False]),
    PS(name='use_govobj_config', values=[True, False]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='internal_token_embd_dim', values=[10, 25, 50, 100, 300]),
    PS(name='update_prep_embd', values=[True, False]),
    PS(name='lstm_h_dim', values=[20, 40, 80, 100]),
    PS(name='num_lstm_layers', values=[2]),
    PS(name='is_bilstm', values=[True]),
    PS(name='num_mlp_layers', values=[2]),
    PS(name='mlp_layer_dim', values=[100]),
    # PS(name='mlp_activation', values=['tanh', 'cube', 'relu']),
    PS(name='mlp_activation', values=['tanh']),
    PS(name='lstm_dropout_p', values=np.arange(.51, step=.01)),
    PS(name='epochs', values=[100]),
    PS(name='learning_rate', values=np.logspace(-2, -1, 6)),
    PS(name='learning_rate_decay', values=np.r_[0, np.logspace(-5, -1, 9)]),
    PS(name='dynet_random_seed', values=[os.environ.get('DYNET_RANDOM_SEED')], enabled=True)
]

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)