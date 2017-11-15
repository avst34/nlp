from hyperparameters_tuner import HyperparametersTuner

PS = HyperparametersTuner.ParamSettings

TUNER_DOMAINS = [
    PS(name='use_token', values=[True]),
    PS(name='use_pos', values=[False]),
    PS(name='use_dep', values=[True]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='pos_embd_dim', values=[30]),
    PS(name='dep_embd_dim', values=[30]),
    PS(name='update_token_embd', values=[True]),
    PS(name='update_pos_embd', values=[True]),
    PS(name='update_dep_embd', values=[True]),
    PS(name='mlp_layers', values=[2]),
    PS(name='mlp_layer_dim', values=[30]),
    PS(name='mlp_activation', values=['tanh']),
    PS(name='lstm_h_dim', values=[30]),
    PS(name='num_lstm_layers', values=[2]),
    PS(name='is_bilstm', values=[True]),
    PS(name='use_head', values=[True]),
    PS(name='mlp_dropout_p', values=[0.1]),
    PS(name='epochs', values=[1]),
    PS(name='validation_split', values=[0.3]),
    PS(name='learning_rate', values=[0.1]),
]