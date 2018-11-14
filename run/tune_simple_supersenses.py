import random
import sys

from datasets.streusle_v4 import StreusleLoader
from hyperparameters_tuner import HyperparametersTuner, override_settings
from models.supersenses_simple.settings import PREP_ROLE_FUNC, PREP_FUNC_ROLE
from models.supersenses_simple.simple_mlp_supersenses_model_hyperparameters_tuner import \
    SimpleSupersensesModelHyperparametersTuner
from models.supersenses_simple.streusle_integration import streusle_record_to_simple_lstm_model_samples

PS = HyperparametersTuner.ParamSettings

print("Loading dataset")
train_recs = StreusleLoader(load_elmo=False).load_train()
dev_recs = StreusleLoader(load_elmo=False).load_dev()

print("Converting to classifier samples")
train_samples = [s for r in train_recs for s in streusle_record_to_simple_lstm_model_samples(r)]
dev_samples = [s for r in dev_recs for s in streusle_record_to_simple_lstm_model_samples(r)]

tasks = {
    # 'PREP_FUNC': PREP_FUNC,
    # 'PREP_ROLE': PREP_ROLE,
    # 'NO_PREP_OBJ_FUNC': OBJ_FUNC,
    # 'NO_PREP_GOV_FUNC': GOV_FUNC,
    # 'NO_PREP_OBJ_ROLE': OBJ_ROLE,
    # 'NO_PREP_GOV_ROLE': GOV_ROLE,
    # 'NO_PREP_GOV_OBJ_FUNC': GOV_OBJ_FUNC,
    # 'NO_PREP_GOV_OBJ_ROLE': GOV_OBJ_ROLE,
    'PREP_ROLE_FUNC': PREP_ROLE_FUNC,
    'PREP_FUNC_ROLE': PREP_FUNC_ROLE,
}
print("Tuning..")
for name, params in random.sample(tasks.items(), len(tasks)):
    print(name)
    model = SimpleSupersensesModelHyperparametersTuner(train_samples, validation_samples=dev_samples, results_csv_path=sys.argv[-1],
                                                       tuner_domains=override_settings([
                                                           params,
                                                           [
                                                            # PS(name='use_instance_embd', values=[True]),
                                                            # PS(name='token_embd_dim', values=[1024 * 3])
                                                           ]
                                                       ]),
                                                       task_name=name)
    model.tune(1)
print("Done tuning")

