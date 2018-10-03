import sys

from datasets.streusle_v4 import StreusleLoader
from hyperparameters_tuner import HyperparametersTuner
from models.gmm_func_clust.gmm_func_clust_model_hyperparameters_tuner import \
    GmmFuncClustModelHyperparametersTuner
from models.gmm_func_clust.streusle_integration import streusle_records_to_gmm_func_clust_model_samples
from models.gmm_func_clust.tuner_domains import TUNER_DOMAINS

PS = HyperparametersTuner.ParamSettings

print("Loading dataset")
train_recs = StreusleLoader(load_elmo=False).load_train()
dev_recs = StreusleLoader(load_elmo=False).load_dev()

print("Converting to classifier samples")
train_samples = streusle_records_to_gmm_func_clust_model_samples(train_recs)
dev_samples = streusle_records_to_gmm_func_clust_model_samples(dev_recs)
print("... done")

model = GmmFuncClustModelHyperparametersTuner(train_samples, validation_samples=dev_samples, results_csv_path="/cs/usr/aviramstern/lab/results_gmm.csv" or sys.argv[-1],
                                                   tuner_domains=TUNER_DOMAINS,
                                                   task_name='gmm_func_clust')
model.tune(1)
# model.sample_execution(json.loads("""{"lstm_dropout_p": 0.04, "use_obj": false, "learning_rate": 0.015848931924611134, "learning_rate_decay": 0.0001, "token_embd_dim": 300, "mlp_activation": "tanh", "epochs": 150, "use_prep": true, "internal_token_embd_dim": 10, "update_prep_embd": true, "lstm_h_dim": 80, "num_lstm_layers": 2, "mlp_layer_dim": 100, "use_govobj_config": true, "use_ud_dep": false, "use_role": false, "is_bilstm": true, "num_mlp_layers": 2, "dynet_random_seed": null, "use_gov": false}"""))
print("Done tuning")

