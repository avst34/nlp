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


TASK_SETTINGS = {
    'goldid.goldsyn': GOLD_ID_GOLD_PREP,
    'goldid.autosyn': GOLD_ID_AUTO_PREP,
    'autoid.goldsyn': AUTO_ID_GOLD_PREP,
    'autoid.autosyn': AUTO_ID_AUTO_PREP,
}