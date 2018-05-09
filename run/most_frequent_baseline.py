from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from models.general.predictor_ops import CombinedPredictor
from models.general.simple_conditional_multiclass_model.model import MostFrequentClassModel
from models.general.simple_conditional_multiclass_model.streusle_integration import \
    streusle_record_to_most_frequent_class_model_sample

evaluator = PSSClasifierEvaluator()

def run(train_records, test_records):
    for features in [[], ['token']]:
        for class_names_grp in [[['supersense_role']], [['supersense_func']], [['supersense_role'], ['supersense_func']], [['supersense_role', 'supersense_func']]]:
            for include_empty in [True, False]:
                predictors = []
                for class_names in class_names_grp:
                    train_samples = [streusle_record_to_most_frequent_class_model_sample(r, class_names=class_names) for r in train_records]
                    dev_samples = [streusle_record_to_most_frequent_class_model_sample(r, class_names=class_names) for r in test_records]

                    scm = MostFrequentClassModel(include_empty=include_empty,
                                                 features=features,
                                                 n_labels_to_predict=len(class_names))
                    predictor = scm.fit(train_samples, validation_samples=dev_samples, evaluator=None)
                    predictors.append(predictor)

                scm = MostFrequentClassModel(include_empty=False,
                                             features=[],
                                             n_labels_to_predict=1)
                dev_samples = [scm.mask_sample(streusle_record_to_most_frequent_class_model_sample(r, class_names=sum(class_names_grp, []))) for r in test_records]
                print('Most Frequent PSS evaluation [Features: %s, include empty: %s, classes: %s]:' % (repr(features),
                                                                                                        str(include_empty),
                                                                                                        repr(class_names_grp)))
                evaluator.evaluate(dev_samples, predictor=CombinedPredictor(predictors), examples_to_show=1)

