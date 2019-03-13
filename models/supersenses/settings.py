from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOLD_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['fasttext_en', 'elmo'])
]])

GOLD_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['fasttext_en', 'elmo'])
]])

AUTO_ID_GOLD_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['fasttext_en', 'elmo'])
]])

AUTO_ID_AUTO_PREP = override_settings([TUNER_DOMAINS, [
    PS(name='allow_empty_prediction', values=[False]),
    PS(name='embd_type', values=['fasttext_en', 'elmo'])
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
            PS(name='use_ud_xpos', values=[True]),
            PS(name='use_govobj', values=[False]),
            PS(name='use_parent', values=[True]),
            PS(name='use_grandparent', values=[True]),
            PS(name='use_token_internal', values=[False]),
        ]
    ])
    for task, settings in TASK_SETTINGS.items()
}

FASTTEXT_TASK_SETTINGS = {
    task: override_settings([
        settings,
        [
            PS(name='embd_type', values=['fasttext_en']),
            PS(name='use_ud_xpos', values=[False, True]),
            PS(name='use_govobj', values=[False]),
            PS(name='use_parent', values=[True]),
            PS(name='use_grandparent', values=[True]),
            PS(name='use_token_internal', values=[False]),
            PS(name='use_prep', values=[False, True]),
            PS(name='prep_dropout_p', values=[0.2, 0.3]),
            PS(name='parent_dropout_p', values=[0.2, 0.3]),
            PS(name='grandparent_dropout_p', values=[0.2, 0.3]),
        ]
    ])
    for task, settings in TASK_SETTINGS.items()
}

ELMO_FASTTEXT_TASK_SETTINGS = {
    task: override_settings([
        settings,
        [
            PS(name='embd_type', values=['fasttext_en', 'elmo']),
            PS(name='epochs', values=[130]),
        ]
    ])
    for task, settings in FASTTEXT_TASK_SETTINGS.items()
}

ELMO_FASTTEXT_MIN_TASK_SETTINGS = {
    task: override_settings([
        settings,
        [
            PS(name='use_lemma', values=[False]),
            PS(name='use_capitalized_word_follows', values=[False]),
            PS(name='use_parent', values=[False]),
            PS(name='use_grandparent', values=[False]),
            PS(name='use_token_internal', values=[False]),
            PS(name='use_prep', values=[True]),
            PS(name='use_ud_xpos', values=[False]),
            PS(name='use_ud_dep', values=[False]),
            PS(name='use_ner', values=[False]),
            PS(name='use_govobj', values=[False]),
            PS(name='use_lexcat', values=[False]),
            PS(name='update_token_embd', values=[False]),
            PS(name='epochs', values=[130]),
        ]
    ])
    for task, settings in ELMO_FASTTEXT_TASK_SETTINGS.items()
}

ELMO_MIN_NELSON = override_settings([
    ELMO_FASTTEXT_MIN_TASK_SETTINGS['goldid.goldsyn'],
    {
        PS(name='embd_type', values=['elmo']),
        PS(name='elmo_layer', values=[1]),
        PS(name='trainer', values=['adam']),
        PS(name='learning_rate', values=['0.001']),
        PS(name='use_lemma', values=[False]),
        PS(name='use_capitalized_word_follows', values=[False]),
        PS(name='use_parent', values=[False]),
        PS(name='use_grandparent', values=[False]),
        PS(name='use_token_internal', values=[False]),
        PS(name='use_prep', values=[True]),
        PS(name='use_ud_xpos', values=[False]),
        PS(name='use_ud_dep', values=[False]),
        PS(name='use_ner', values=[False]),
        PS(name='use_govobj', values=[False]),
        PS(name='use_lexcat', values=[False]),
        PS(name='update_token_embd', values=[False]),
        PS(name='epochs', values=[130]),

    }
])