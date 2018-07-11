from hyperparameters_tuner import HyperparametersTuner, override_settings
from .tuner_domains import TUNER_DOMAINS
PS = HyperparametersTuner.ParamSettings

GOV_FUNC = override_settings([TUNER_DOMAINS, [
    PS(name='labels_to_predict', values=['supersense_func']),
    PS(name='use_gov', values=[True]),
    PS(name='use_obj', values=[False])
]])

GOV_ROLE = override_settings([TUNER_DOMAINS, [
    PS(name='labels_to_predict', values=['supersense_role']),
    PS(name='use_gov', values=[True]),
    PS(name='use_obj', values=[False])
]])

OBJ_FUNC = override_settings([TUNER_DOMAINS, [
    PS(name='labels_to_predict', values=['supersense_func']),
    PS(name='use_gov', values=[False]),
    PS(name='use_obj', values=[True])
]])

OBJ_ROLE = override_settings([TUNER_DOMAINS, [
    PS(name='labels_to_predict', values=['supersense_role']),
    PS(name='use_gov', values=[False]),
    PS(name='use_obj', values=[True])
]])


TASK_SETTINGS = {
    'gov.func': GOV_FUNC,
    'gov.role': GOV_ROLE,
    'obj.func': OBJ_FUNC,
    'obj.role': OBJ_ROLE,
}