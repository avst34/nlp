from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev
from models.hcpd.hcpd_model import HCPDModel
from models.hcpd.boknilev_integration import boknilev_record_to_hcpd_samples

train_recs, dev_recs, test_recs = load_boknilev()
train_samples = [s for r in train for s in boknilev_record_to_hcpd_samples(r)]
dev_samples = [s for r in dev for s in boknilev_record_to_hcpd_samples(r)]
test_samples = [s for r in test for s in boknilev_record_to_hcpd_samples(r)]

model = HCPDModel()

model.fit(train_samples, validation_samples=dev_samples, show_progress=True, show_epoch_eval=True)