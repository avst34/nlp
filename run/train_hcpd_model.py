from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from evaluators.ppatt_evaluator import PPAttEvaluator
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()

print("Converting to classifier samples")
train_samples = [s for r in train_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples = [s for r in test_recs for s in boknilev_record_to_hcpd_samples(r)]

print("Training")
model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(epochs=100))
model.fit(train_samples, validation_samples=dev_samples, show_progress=True)
print("Training complete, saving model..")
model.save('/tmp/hcpd_trained_model')
print("Done saving model")
print("Evaluating model on test:")
PPAttEvaluator(predictor=model).evaluate(test_samples)

