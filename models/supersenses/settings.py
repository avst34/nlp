from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOLD_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['word2vec'])
]])

GOLD_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['word2vec'])
]])

AUTO_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['word2vec'])
]])

AUTO_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['word2vec'])
]])


GOLD_ID_GOLD_PREP_GOLD_ROLE = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='labels_to_learn', values=[('supersense_func',)]),
    PS(name='labels_to_predict', values=[('supersense_func',)]),
    PS(name='use_role', values=[True]),
    PS(name='embd_type', values=['word2vec'])
]])

GOLD_ID_GOLD_PREP_GOLD_FUNC = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='labels_to_learn', values=[('supersense_role',)]),
    PS(name='labels_to_predict', values=[('supersense_role',)]),
    PS(name='use_func', values=[True]),
    PS(name='embd_type', values=['word2vec'])
]])

TASK_SETTINGS = {
    'goldid.goldsyn': GOLD_ID_GOLD_PREP,
    'goldid.autosyn': GOLD_ID_AUTO_PREP,
    'autoid.goldsyn': AUTO_ID_GOLD_PREP,
    'autoid.autosyn': AUTO_ID_AUTO_PREP,
    'goldid.goldsyn.goldrole': GOLD_ID_GOLD_PREP_GOLD_ROLE,
    'goldid.goldsyn.goldfunc': GOLD_ID_GOLD_PREP_GOLD_FUNC,
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

MUSE_TASK_SETTINGS = {
    task: override_settings([
        settings,
        [
            PS(name='embd_type', values=['muse']),
            PS(name='use_ud_xpos', values=[False])
        ]
    ])
    for task, settings in TASK_SETTINGS.items()
}
