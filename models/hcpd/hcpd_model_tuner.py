from evaluators.ppatt_evaluator import PPAttEvaluator
from hyperparameters_tuner import HyperparametersTuner
import json

from models.hcpd.hcpd_model import HCPDModel
from .tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation

def build_csv_rows(params, result):
    assert isinstance(result, HyperparametersTuner.ExecutionResult)
    result_data = result.result_data
    rows_tuples = []
    for scope, scope_data in result_data.items():
        row_tuples = \
            [("Scope", scope)] + \
            [("Best Epoch", scope_data['best_epoch'])] + \
            [("Acc", scope_data['acc'])] + \
            [(k, str(v)) for k, v in sorted(params.items())] + \
            [("Hyperparams Json", json.dumps(params))]
        rows_tuples.append(row_tuples)

    headers = [x[0] for x in rows_tuples[0]]
    rows = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
    return headers, rows


class HCPDModelTuner:

    def __init__(self,
                 samples,
                 results_csv_path,
                 tuner_domains=TUNER_DOMAINS,
                 validation_samples=None,
                 show_progress=True,
                 dump_models=False,
                 evaluator=PPAttEvaluator(),
                 tuner_score_getter=lambda e: e['acc'],
                 tuner_results_getter=extract_classifier_evaluator_results,
                 task_name=''):
        self.task_name = task_name
        self.dump_models = dump_models
        self.fit_kwargs = None
        self.tuner_results_getter = tuner_results_getter
        self.tuner_score_getter = tuner_score_getter

        assert evaluator is not None

        def dump_result(output_dir, result, params):
            if self.dump_models:
                result.predictor.save(output_dir + '/model')

        self.tuner = HyperparametersTuner(task_name=task_name,
                                          results_csv_path=results_csv_path,
                                          params_settings=tuner_domains, executor=self._execute,
                                          csv_row_builder=build_csv_rows, shared_csv=True,
                                          lock_file_path=results_csv_path + '.lock',
                                          dump_result=dump_result)

        self.fit_kwargs = {
            'samples': samples,
            'validation_samples': validation_samples,
            'show_progress': show_progress,
        }

    def _execute(self, hyperparameters):
        model = HCPDModel(hyperparameters=HCPDModel.HyperParameters(**hyperparameters))
        model.fit(**self.fit_kwargs)
        return HyperparametersTuner.ExecutionResult(
            result_data={
                'train': self.tuner_results_getter(model.train_set_evaluation),
                'test':  self.tuner_results_getter(model.test_set_evaluation),
            },
            score=self.tuner_score_getter(model.test_set_evaluation),
            predictor=model
        )

    def tune(self, n_executions=30):
        best_params, best_results = self.tuner.tune(n_executions)
        return best_params, best_results

    def sample_execution(self, params=None):
        return self.tuner.sample_execution(params)
