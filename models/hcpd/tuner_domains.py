from hyperparameters_tuner import HyperparametersTuner
import numpy as np
import os

PS = HyperparametersTuner.ParamSettings

# For tuning
TUNER_DOMAINS_TUNING = [
    PS(name='max_head_distance', values=[1, 5]),
    PS(name='p1_vec_dim', values=[100]),
    # PS(name='p1_vec_dim', values=[50, 100, 146, 200, 300]),
    # PS(name='p1_mlp_layers', values=[1, 2, 3]),
    PS(name='p1_mlp_layers', values=[1]),
    PS(name='p2_vec_dim', values=[100]),
    # PS(name='p2_vec_dim', values=[50, 100, 146, 200, 300]),
    # PS(name='p2_mlp_layers', values=[1, 2, 3]),
    PS(name='p2_mlp_layers', values=[1]),
    # PS(name='activation', values=['tanh', 'cube', 'rectify']),
    PS(name='activation', values=['tanh']),
    PS(name='use_pss', values=[True, False]),
    PS(name='pss_embd_dim', values=[5, 10, 20, 50]),
    PS(name='pss_embd_type', values=['lookup', 'binary']),
    # PS(name='use_verb_noun_ss', values=[True, False]),
    PS(name='use_verb_noun_ss', values=[False]),
    # PS(name='dropout_p', values=[0.01, 0.1, 0.3, 0.5, 0.7]),
    PS(name='dropout_p', values=[0.5]),
    PS(name='learning_rate', values=[0.1, 1]),
    PS(name='learning_rate_decay', values=[0]),
    PS(name='update_embeddings', values=[True]),
    PS(name='fallback_to_lemmas', values=[True, False]),
    PS(name='trainer', values=["SimpleSGDTrainer", "AdagradTrainer"]),
    PS(name='epochs', values=[100])
]


TUNER_DOMAINS = TUNER_DOMAINS_TUNING

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS_TUNING:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)