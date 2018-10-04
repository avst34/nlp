from hyperparameters_tuner import HyperparametersTuner

PS = HyperparametersTuner.ParamSettings

# For tuning
TUNER_DOMAINS = [
    PS(name='use_prep', values=[True, False]),
    PS(name='use_gov', values=[True, False]),
    PS(name='use_obj', values=[True, False]),
    PS(name='use_role', values=[False]),
    # PS(name='use_role', values=[True, False]),
    PS(name='use_ud_dep', values=[True, False]),
    PS(name='use_govobj_config', values=[True, False]),
    PS(name='token_embd_dim', values=[300]),
    PS(name='cov_type', values=['spherical', 'diag', 'tied', 'full']),
    PS(name='gmm_max_iter', values=[100]),
    PS(name='gmm_means_init', values=['by_role']),
    # PS(name='gmm_means_init', values=['random', 'by_role', 'kmeans']),
]

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)