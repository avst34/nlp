import sys

from datasets.streusle_v4 import StreusleLoader
from hyperparameters_tuner import HyperparametersTuner
from models.pairwise_func_clust.pairwise_func_clust_model_hyperparameters_tuner import \
    PairwiseFuncClustModelHyperparametersTuner
from models.pairwise_func_clust.streusle_integration import streusle_records_to_pairwise_func_clust_model_samples
from models.pairwise_func_clust.tuner_domains import TUNER_DOMAINS

PS = HyperparametersTuner.ParamSettings

print("Loading dataset")
train_recs = StreusleLoader(load_elmo=False).load_train()
dev_recs = StreusleLoader(load_elmo=False).load_dev()

print("Converting to classifier samples")
train_samples = streusle_records_to_pairwise_func_clust_model_samples(train_recs)
dev_samples = streusle_records_to_pairwise_func_clust_model_samples(dev_recs)
print("... done")

model = PairwiseFuncClustModelHyperparametersTuner(train_samples, validation_samples=dev_samples, results_csv_path=sys.argv[-1],
                                                   tuner_domains=TUNER_DOMAINS,
                                                   task_name='pairwise_func_clust')
model.tune(1)
print("Done tuning")

