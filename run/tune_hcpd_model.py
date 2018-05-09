import sys

from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from evaluators.ppatt_evaluator import PPAttEvaluator
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples
from models.hcpd.hcpd_model_tuner import HCPDModelTuner

print("Loading dataset")
train_recs, dev_recs, test_recs = load_boknilev()

print("Converting to classifier samples")
train_samples = [s for r in train_recs for s in boknilev_record_to_hcpd_samples(r)]
dev_samples = [s for r in dev_recs for s in boknilev_record_to_hcpd_samples(r)]
test_samples = [s for r in test_recs for s in boknilev_record_to_hcpd_samples(r)]

print("Tuning..")
model = HCPDModelTuner(train_samples, results_csv_path=sys.argv[-1], validation_samples=dev_samples)
model.tune(1)
print("Done tuning")

