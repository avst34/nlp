from datasets.streusle_v4 import StreusleLoader
from models.pairwise_func_clust.pairwise_func_clust_evaluator import PairwiseFuncClustEvaluator
from models.pairwise_func_clust.pairwise_func_clust_model import PairwiseFuncClustModel
from models.pairwise_func_clust.streusle_integration import streusle_records_to_pairwise_func_clust_model_samples


class PredictByRole(object):

    def predict(self, sample_x):
        return PairwiseFuncClustModel.SampleY(1 if sample_x.role1 == sample_x.role2 else 0)

class PredictByPrep(object):

    def predict(self, sample_x):
        return PairwiseFuncClustModel.SampleY(1 if sample_x.prep_tokens1 == sample_x.prep_tokens2 else 0)

class PredictByPrepAndRole(object):

    def predict(self, sample_x):
        return PairwiseFuncClustModel.SampleY(1 if sample_x.prep_tokens1 == sample_x.prep_tokens2 and sample_x.role1 == sample_x.role2 else 0)

class PredictByPrepOrRole(object):

    def predict(self, sample_x):
        return PairwiseFuncClustModel.SampleY(1 if sample_x.prep_tokens1 == sample_x.prep_tokens2 or sample_x.role1 == sample_x.role2 else 0)

print("Loading dataset")
train_recs = StreusleLoader(load_elmo=False).load_train()
dev_recs = StreusleLoader(load_elmo=False).load_dev()

print("Converting to classifier samples")
train_samples = streusle_records_to_pairwise_func_clust_model_samples(train_recs)
dev_samples = streusle_records_to_pairwise_func_clust_model_samples(dev_recs)
print("... done")

print("Dev: %d%% TRUE, %d%% FALSE" % (len([x for x in dev_samples if x.y.is_same_cluster])/len(dev_samples)*100, len([x for x in dev_samples if not x.y.is_same_cluster])/len(dev_samples)*100))

BASELINES = {
    'role1==role2': PredictByRole(),
    'prep1==prep2': PredictByPrep(),
    'role1==role2 AND prep1==prep2': PredictByPrepAndRole(),
    'role1==role2 OR  prep1==prep2': PredictByPrepOrRole(),
}

for baseline, predictor in BASELINES.items():
    print(baseline)
    print('----------')
    PairwiseFuncClustEvaluator(predictor).evaluate(dev_samples, examples_to_show=0)
    print(' ')
