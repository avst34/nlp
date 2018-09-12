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
# model.sample_execution(json.loads("""{"use_ud_dep": true, "learning_rate": 0.1,"num_mlp_layers":2, "mlp_layer_dim": 100, "mlp_activation": "tanh", "lstm_h_dim": 80, "num_lstm_layers": 2, "dynet_random_seed": null, "use_obj": true, "internal_token_embd_dim": 300, "learning_rate_decay": 0.03162277660168379, "lstm_dropout_p": 0.15, "use_role": true, "update_prep_embd": false, "epochs": 100, "use_govobj_config": true, "is_bilstm": true, "token_embd_dim": 300, "use_prep": true, "use_gov": true}"""))
print("Done tuning")

