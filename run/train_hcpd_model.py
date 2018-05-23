from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from evaluators.ppatt_evaluator import PPAttEvaluator
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples
import json

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()

print("Converting to classifier samples")
train_samples = [s for r in train_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples = [s for r in test_recs for s in boknilev_record_to_hcpd_samples(r)]

print("Training")
# model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(epochs=100))
model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(**json.loads("""{"update_embeddings": true, "activation": "rectify", "dropout_p": 0.01, "learning_rate": 0.1, "trainer": "SimpleSGDTrainer", "epochs": 12, "p1_vec_dim": 150, "p2_vec_dim": 150, "p1_mlp_layers": 3, "p2_mlp_layers": 3, "learning_rate_decay": 0, "max_head_distance": 5, "use_pss": false, "fallback_to_lemmas": false}""")))
# model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(**json.loads("""{"update_embeddings": true, "activation": "rectify", "dropout_p": 0.01, "learning_rate": 0.1, "trainer": "SimpleSGDTrainer", "epochs": 20, "internal_layer_dim": 150, "learning_rate_decay": 0, "max_head_distance": 5, "use_pss": true, "fallback_to_lemmas": false}""")))
model.fit(train_samples, validation_samples=dev_samples, show_progress=True)
print("Training complete, saving model..")
model.save('/tmp/hcpd_trained_model')
print("Done saving model")
print("Evaluating model on dev:")
PPAttEvaluator(predictor=model).evaluate(dev_samples, examples_to_show=0, predictions_csv_path='/tmp/dev_predict.csv')
print("Evaluating model on test:")
PPAttEvaluator(predictor=model).evaluate(test_samples, examples_to_show=0, predictions_csv_path='/tmp/test_predict.csv')

