import json

from evaluators.simple_pss_classifier_evaluator import SimplePSSClassifierEvaluator
from hyperparameters_tuner import HyperparametersTuner
from models.supersenses_simple.simple_mlp_supersenses_model import SimpleMlpSupersensesModel
from models.supersenses_simple.tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation


class SimpleSupersensesModelHyperparametersTuner:

    def build_csv_rows(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        result_data = result.result_data
        rows_tuples = [
            [("Best Epoch", result_data['best_epoch'])] + \
            [("Train Acc", result_data['train_acc'])] + \
            [("Test Acc", result_data['test_acc'])] + \
            [(k, str(v)) for k, v in sorted(params.items())] + \
            [("Hyperparams Json", json.dumps(params))]
        ]

        headers = [x[0] for x in rows_tuples[0]]
        rows = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
        return headers, rows

    def __init__(self,
                 samples,
                 results_csv_path,
                 tuner_domains=TUNER_DOMAINS,
                 validation_samples=None,
                 show_progress=True,
                 show_epoch_eval=True,
                 report_epoch_scores=False,
                 dump_models=False,
                 shared_csv=True,
                 evaluator=SimplePSSClassifierEvaluator(),
                 task_name=''):
        self.task_name = task_name
        self.dump_models = dump_models
        self.fit_kwargs = None
        self.report_epoch_scores = report_epoch_scores

        assert evaluator is not None

        def dump_result(output_dir, result):
            if self.dump_models:
                result.predictor.save(output_dir + '/model')

        self.tuner = HyperparametersTuner(task_name=task_name,
                                          results_csv_path=results_csv_path,
                                          params_settings=tuner_domains, executor=self._execute,
                                          csv_row_builder=self.build_csv_rows, shared_csv=shared_csv,
                                          lock_file_path=results_csv_path + '.lock',
                                          dump_result=dump_result)

        self.fit_kwargs = {
            'samples': samples,
            'validation_samples': validation_samples,
            'show_progress': show_progress,
            'show_epoch_eval': show_epoch_eval,
            'evaluator': evaluator
        }

    def _execute(self, hyperparameters):
        model = SimpleMlpSupersensesModel(hyperparameters=SimpleMlpSupersensesModel.HyperParameters(**hyperparameters))
        model.fit(**self.fit_kwargs)
        return HyperparametersTuner.ExecutionResult(
            result_data={
                'train_acc': model.train_acc,
                'test_acc': model.test_acc,
                'best_epoch': model.best_epoch
            },
            score=model.test_acc,
            predictor=model
        )

    def tune(self, n_executions=30):
        best_params, best_results = self.tuner.tune(n_executions)
        return best_params, best_results

    def sample_execution(self, params=None):
        return self.tuner.sample_execution(params)
