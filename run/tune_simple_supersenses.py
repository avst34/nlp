import sys

from datasets.streusle_v4 import StreusleLoader
from hyperparameters_tuner import override_settings
from models.supersenses_simple.settings import GOV_FUNC, OBJ_FUNC, OBJ_ROLE, GOV_ROLE, PS
from models.supersenses_simple.simple_mlp_supersenses_model_hyperparameters_tuner import \
    SimpleSupersensesModelHyperparametersTuner
from models.supersenses_simple.streusle_integration import streusle_record_to_simple_lstm_model_samples

print("Loading dataset")
train_recs = StreusleLoader().load_train()
dev_recs = StreusleLoader().load_dev()

print("Converting to classifier samples")
train_samples = [s for r in train_recs for s in streusle_record_to_simple_lstm_model_samples(r)]
dev_samples = [s for r in dev_recs for s in streusle_record_to_simple_lstm_model_samples(r)]

tasks = {
    'OBJ_FUNC': OBJ_FUNC,
    'GOV_FUNC': GOV_FUNC,
    'OBJ_ROLE': OBJ_ROLE,
    'GOV_ROLE': GOV_ROLE,
}

print("Tuning..")
for name, params in tasks.items():
    model = SimpleSupersensesModelHyperparametersTuner(train_samples, validation_samples=dev_samples, results_csv_path=sys.argv[-1],
                                                       tuner_domains=override_settings([params, [
                                                           PS(name='use_gov', values=[False]),
                                                           PS(name='use_obj', values=[False])
                                                       ]]),
                                                       task_name=name)
    model.tune(1)
print("Done tuning")

