import json

from hyperparameters_tuner import HyperparametersTuner
from models.gmm_func_clust.gmm_func_clust_evaluator import GmmFuncClustEvaluator
from models.gmm_func_clust.gmm_func_clust_model import GmmFuncClustModel
from models.gmm_func_clust.tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation


class GmmFuncClustModelHyperparametersTuner:

    def build_csv_rows(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        result_data = result.result_data
        rows_tuples = [
            # [("Train Acc", result_data['train_acc'])] + \
            [("Train P", result_data['train_p'])] + \
            [("Train R", result_data['train_r'])] + \
            [("Train F1", result_data['train_f1'])] + \
            [("Test P", result_data['test_p'])] + \
            [("Test R", result_data['test_r'])] + \
            [("Test F1", result_data['test_f1'])] + \
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
                 evaluator=GmmFuncClustEvaluator(),
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
            'evaluator': evaluator
        }

    def _execute(self, hyperparameters):
        model = GmmFuncClustModel(hyperparameters=GmmFuncClustModel.HyperParameters(**hyperparameters))
        model.fit(**self.fit_kwargs)
        return HyperparametersTuner.ExecutionResult(
            result_data={
                # 'train_acc': model.train_acc,
                'test_p': model.test_eval['p'],
                'test_r': model.test_eval['r'],
                'test_f1': model.test_eval['f1'],
                'train_p': model.train_eval['p'],
                'train_r': model.train_eval['r'],
                'train_f1': model.train_eval['f1'],
            },
            score=model.test_eval['f1'],
            predictor=model
        )

    def tune(self, n_executions=30):
        best_params, best_results = self.tuner.tune(n_executions)
        return best_params, best_results

    def sample_execution(self, params=None):
        return self.tuner.sample_execution(params)
