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
    PS(name='svm_gamma', values=['scale', 'auto'] + list(np.logspace(-9, 3, 13))),
    PS(name='svm_shrinking', values=[True]),
    PS(name='svm_tol', values=[0.001]),
    PS(name='svm_c', values=list(np.logspace(-2, 10, 13))),
]

if __name__ == '__main__':
    total_variations = 1
    for domain in TUNER_DOMAINS:
        total_variations *= len(domain.values)
    print('Total variations:', total_variations)