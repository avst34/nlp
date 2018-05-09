from hyperparameters_tuner import HyperparametersTuner
import numpy as np
import os

PS = HyperparametersTuner.ParamSettings

# For tuning
TUNER_DOMAINS_TUNING = [
    PS(name='max_head_distance', values=[5]),
    PS(name='internal_layer_dim', values=[None, 50, 100, 200, 300]),
    PS(name='activation', values=['tanh', 'cube', 'rectify']),
    PS(name='dropout_p', values=[0.01, 0.1, 0.3, 0.5, 0.7]),
    PS(name='learning_rate', values=[0.1, 1]),
    PS(name='learning_rate_decay', values=[0]),
    PS(name='update_embeddings', values=[True]),
    PS(name='trainer', values=["SimpleSGDTrainer", "AdagradTrainer"]),
    PS(name='epochs', values=[1])
]


TUNER_DOMAINS = TUNER_DOMAINS_TUNING

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS_TUNING:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)