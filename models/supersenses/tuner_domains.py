from hyperparameters_tuner import HyperparametersTuner
import numpy as np

PS = HyperparametersTuner.ParamSettings

TUNER_DOMAINS = [
    PS(name='use_token', values=[True]),
    PS(name='use_pos', values=[True, False]),
    PS(name='use_dep', values=[True, False]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='pos_embd_dim', values=range(20, 101)),
    PS(name='dep_embd_dim', values=range(20, 101)),
    PS(name='update_token_embd', values=[False, True]),
    PS(name='update_pos_embd', values=[False, True]),
    PS(name='update_dep_embd', values=[False, True]),
    PS(name='mlp_layers', values=[1,2,3]),
    PS(name='mlp_layer_dim', values=range(20, 101)),
    PS(name='mlp_activation', values=['tanh', 'cube', 'relu']),
    PS(name='lstm_h_dim', values=range(20, 101, 2)),
    PS(name='num_lstm_layers', values=[2]),
    PS(name='is_bilstm', values=[True]),
    PS(name='use_head', values=[False, True]),
    PS(name='mlp_dropout_p', values=np.arange(.51, step=.01)),
    # PS(name='epochs', values=range(50, 101)),
    PS(name='epochs', values=range(1, 2)),
    PS(name='validation_split', values=[0.3]),
    PS(name='learning_rate', values=np.logspace(-5, 0, 11)),
    PS(name='learning_rate_decay', values=np.r_[0, np.logspace(-5, -1, 9)])
]