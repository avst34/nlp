import json

from hyperparameters_tuner import HyperparametersTuner
from models.svm_pairwise_func_clust.svm_pairwise_func_clust_evaluator import SvmPairwiseFuncClustEvaluator
from models.svm_pairwise_func_clust.svm_pairwise_func_clust_model import SvmPairwiseFuncClustModel
from models.svm_pairwise_func_clust.tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation

def prob_to_prec(p):
    return int(10000*p) / 100


class SvmPairwiseFuncClustModelHyperparametersTuner:

    def build_csv_rows(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        result_data = result.result_data
        rows_tuples = [
            [("Test P", prob_to_prec(result_data['test_p']))] + \
            [("Test R", prob_to_prec(result_data['test_r']))] + \
            [("Test F1", prob_to_prec(result_data['test_f1']))] + \
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
                 evaluator=SvmPairwiseFuncClustEvaluator(),
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
        model = SvmPairwiseFuncClustModel(hyperparameters=SvmPairwiseFuncClustModel.HyperParameters(**hyperparameters))
        model.fit(**self.fit_kwargs)
        return HyperparametersTuner.ExecutionResult(
            result_data={
                # 'train_acc': model.train_acc,
                'test_p': model.test_eval['p'],
                'test_r': model.test_eval['r'],
                'test_f1': model.test_eval['f1'],
            },
            score=model.test_eval['f1'],
            predictor=model
        )

    def tune(self, n_executions=30):
        best_params, best_results = self.tuner.tune(n_executions)
        return best_params, best_results

    def sample_execution(self, params=None):
        return self.tuner.sample_execution(params)
