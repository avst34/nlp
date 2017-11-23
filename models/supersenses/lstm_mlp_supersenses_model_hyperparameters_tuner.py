from evaluators.classifier_evaluator import ClassifierEvaluator
from hyperparameters_tuner import HyperparametersTuner
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
import json

from models.supersenses.tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation

def build_csv_rows(params, result):
    assert isinstance(result, HyperparametersTuner.ExecutionResult)
    rows_tuples = []
    for scope in ['train', 'test']:
        class_scores = result.result_data[scope]['class_scores']
        classes_ordered = sorted(class_scores.keys())
        if ClassifierEvaluator.ALL_CLASSES in classes_ordered:
            classes_ordered.remove(ClassifierEvaluator.ALL_CLASSES)
            classes_ordered = [ClassifierEvaluator.ALL_CLASSES] + classes_ordered

        for klass in classes_ordered:
            scores = class_scores[klass]
            if klass == ClassifierEvaluator.ALL_CLASSES:
                klass = '-- All Classes --'
            row_tuples = \
                [("Tuner Score", result.score)] + \
                [("Scope", scope)] + \
                [("Class", klass)] + \
                sorted({k: scores[k] for k, v in scores.items()}.items()) + \
                sorted(params.items()) + \
                [("Hyperparams Json", json.dumps(params))]
            rows_tuples.append(row_tuples)

    headers = [x[0] for x in rows_tuples[0]]
    rows = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
    return headers, rows

class LstmMlpSupersensesModelHyperparametersTuner:

    def __init__(self,
                 token_vocab=None,
                 pos_vocab=None,
                 dep_vocab=None,
                 token_onehot_vocab=None,
                 supersense_vocab=None,
                 token_embd=None,
                 pos_embd=None):
        self.init_kwargs = {
            'token_vocab': token_vocab,
            'pos_vocab': pos_vocab,
            'dep_vocab': dep_vocab,
            'token_onehot_vocab': token_onehot_vocab,
            'supersense_vocab': supersense_vocab,
            'token_embd': token_embd,
            'pos_embd': pos_embd
        }
        self.fit_kwargs = None
        self.tuner_results_getter = None
        self.tuner_score_getter = None

    def _execute(self, hyperparameters):
        lstm_mlp_model = LstmMlpSupersensesModel(hyperparameters=LstmMlpSupersensesModel.HyperParameters(**hyperparameters), **self.init_kwargs)
        lstm_mlp_model.fit(**self.fit_kwargs)
        return HyperparametersTuner.ExecutionResult(
            result_data={
                'train': self.tuner_results_getter(lstm_mlp_model.train_set_evaluation),
                'test':  self.tuner_results_getter(lstm_mlp_model.test_set_evaluation),
            },
            score=self.tuner_score_getter(lstm_mlp_model.test_set_evaluation)
        )

    def tune(self, samples, results_csv_path, validation_samples=None, n_executions=30, show_progress=True, show_epoch_eval=True,
             evaluator=ClassifierEvaluator(), tuner_score_getter=lambda evaluation: evaluation['f1'],
             tuner_results_getter=extract_classifier_evaluator_results):
        assert evaluator is not None
        self.tuner_results_getter = tuner_results_getter
        self.tuner_score_getter = tuner_score_getter
        self.fit_kwargs = {
            'samples': samples,
            'validation_samples': validation_samples,
            'show_progress': show_progress,
            'show_epoch_eval': show_epoch_eval,
            'evaluator': evaluator
        }
        tuner = HyperparametersTuner(TUNER_DOMAINS, executor=self._execute,
                                     csv_row_builder=build_csv_rows, shared_csv=True,
                                     lock_file_path=results_csv_path + '.lock')
        best_params, best_results = tuner.tune(results_csv_path, n_executions)
        return best_params, best_results
