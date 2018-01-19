from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOLD_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='lemmas_from', values=['ud'], task_param=True),
    PS(name='deps_from', values=['ud'], task_param=True),
    PS(name='pos_from', values=['ud'], task_param=True),
    PS(name='mask_by', values=['sample-ys'], task_param=True),
    PS(name='mask_mwes', values=[False], task_param=True)
]])

GOLD_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='lemmas_from', values=['spacy'], task_param=True),
    PS(name='deps_from', values=['ud'], task_param=True),
    PS(name='pos_from', values=['ud'], task_param=True),
    PS(name='mask_by', values=['sample-ys'], task_param=True),
    PS(name='mask_mwes', values=[False], task_param=True)
]])
