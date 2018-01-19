from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOLD_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='lemmas_from', values=['ud']),
    PS(name='use_spacy_ner', values=[True, False]),
    PS(name='deps_from', values=['ud']),
    PS(name='pos_from', values=['ud']),
    PS(name='mask_by', values=['sample-ys']),
    PS(name='mask_mwes', values=[False])
]])

GOLD_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='lemmas_from', values=['spacy']),
    PS(name='use_spacy_ner', values=[True, False]),
    PS(name='deps_from', values=['ud']),
    PS(name='pos_from', values=['ud']),
    PS(name='mask_by', values=['sample-ys']),
    PS(name='mask_mwes', values=[False])
]])
