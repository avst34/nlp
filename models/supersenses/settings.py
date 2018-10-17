from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOLD_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False])
]])

GOLD_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False])
]])

AUTO_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False])
]])

AUTO_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False])
]])


GOLD_ID_GOLD_PREP_GOLD_ROLE = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='labels_to_predict', values=[('supersense_func',)]),
    PS(name='use_role', values=[True])
]])

TASK_SETTINGS = {
    'goldid.goldsyn': GOLD_ID_GOLD_PREP,
    'goldid.autosyn': GOLD_ID_AUTO_PREP,
    'autoid.goldsyn': AUTO_ID_GOLD_PREP,
    'autoid.autosyn': AUTO_ID_AUTO_PREP,
    'goldid.goldsyn.goldrole': GOLD_ID_GOLD_PREP_GOLD_ROLE,
}

ELMO_TASK_SETTINGS = {
    task: override_settings([
        settings,
        [
            PS(name='token_embd_dim', values=[3072]),
            PS(name='use_instance_embd', values=[True])
        ]
    ])
    for task, settings in TASK_SETTINGS.items()
}
