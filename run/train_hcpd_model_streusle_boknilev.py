import math

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from datasets.pp_attachement.streusle.load_streusle import load_streusle
from evaluators.ppatt_evaluator import PPAttEvaluator
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples
import json

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()
strain_recs, sdev_recs, stest_recs = load_streusle()

ratio = math.ceil(len(train_recs) / len(strain_recs))
ratio_pps = math.ceil(len([pp for t in train_recs for pp in t['pps']]) / len([pp for t in strain_recs for pp in t['pps']]))
all_train_recs = train_recs + strain_recs * ratio_pps

print('Boknilev/Streusle - sentences: %d/%d' % (len(train_recs), len(strain_recs)))
print('Boknilev/Streusle - pps      : %d/%d' % (len([pp for t in train_recs for pp in t['pps']]), len([pp for t in strain_recs for pp in t['pps']])))
print('Ratio: %d' % ratio)
print('Ratio (pps): %d' % ratio_pps)
print('Final training samples size: %d sents, %d pps' % (len(all_train_recs), len([pp for t in all_train_recs for pp in t['pps']])))

print("Converting to classifier samples")
train_samples = [s for r in all_train_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_boknilev = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples_streusle = [s for r in sdev_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples_boknilev = [s for r in test_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples_streusle = [s for r in stest_recs for s in boknilev_record_to_hcpd_samples(r)]


print("Training")
# model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(epochs=100))
model = HCPDModel(
    hyperparameters=HCPDModel.HyperParameters(use_pss=False, use_verb_noun_ss=False)
    # hyperparameters=HCPDModel.HyperParameters(**json.loads("""{"update_embeddings": true, "activation": "rectify", "dropout_p": 0.01, "learning_rate": 0.1, "trainer": "SimpleSGDTrainer", "epochs": 12, "p1_vec_dim": 150, "p2_vec_dim": 150, "p1_mlp_layers": 3, "p2_mlp_layers": 3, "learning_rate_decay": 0, "max_head_distance": 5, "use_pss": true, "fallback_to_lemmas": false}""")),
    # debug_feature='pss'
)
# model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(**json.loads("""{"update_embeddings": true, "activation": "rectify", "dropout_p": 0.01, "learning_rate": 0.1, "trainer": "SimpleSGDTrainer", "epochs": 20, "internal_layer_dim": 150, "learning_rate_decay": 0, "max_head_distance": 5, "use_pss": true, "fallback_to_lemmas": false}""")))
model.fit(train_samples, validation_samples=dev_samples_boknilev, additional_validation_sets={'streusle': dev_samples_streusle}, show_progress=True)
print("Training complete, saving model..")
model.save('/tmp/hcpd_trained_model')
print("Done saving model")
print("Evaluating model on dev (boknilev):")
PPAttEvaluator(predictor=model).evaluate(dev_samples_boknilev, examples_to_show=0, predictions_csv_path='/tmp/dev_predict_boknilev.csv')
print("Evaluating model on dev (streusle):")
PPAttEvaluator(predictor=model).evaluate(dev_samples_streusle, examples_to_show=0, predictions_csv_path='/tmp/dev_predict_streusle.csv')
print("Evaluating model on test (boknilev):")
PPAttEvaluator(predictor=model).evaluate(test_samples_boknilev, examples_to_show=0, predictions_csv_path='/tmp/test_predict_boknilev.csv')
print("Evaluating model on test (streusle):")
PPAttEvaluator(predictor=model).evaluate(test_samples_streusle, examples_to_show=0, predictions_csv_path='/tmp/test_predict_streusle.csv')

