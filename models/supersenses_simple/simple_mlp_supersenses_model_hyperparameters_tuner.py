import json

from evaluators.pss_classifier_evaluator import PSSClasifierEvaluator
from evaluators.simple_pss_classifier_evaluator import SimplePSSClassifierEvaluator
from hyperparameters_tuner import HyperparametersTuner
from models.supersenses.lstm_mlp_supersenses_model import SimpleMlpSupersensesModel
from models.supersenses.tuner_domains import TUNER_DOMAINS


def extract_classifier_evaluator_results(evaluation):
    return evaluation


class SimpleSupersensesModelHyperparametersTuner:

    def build_csv_rows(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        result_data = result.result_data
        rows_tuples = []
        best_epoch = max([(evaluation['f1'] or 0, epoch) for epoch, evaluation in enumerate(result_data['test'])])[1]
        for scope, scope_data in result_data.items():
            for epoch, epoch_data in enumerate(scope_data):
                class_scores = epoch_data['class_scores']
                classes_ordered = sorted(class_scores.keys(), key=lambda k: str(k))
                if PSSClasifierEvaluator.ALL_CLASSES in classes_ordered:
                    classes_ordered.remove(PSSClasifierEvaluator.ALL_CLASSES)
                    classes_ordered = [PSSClasifierEvaluator.ALL_CLASSES] + classes_ordered

                is_last_epoch = epoch == len(scope_data) - 1
                is_best_epoch = epoch == best_epoch
                for klass in classes_ordered:
                    scores = class_scores[klass]
                    if klass == PSSClasifierEvaluator.ALL_CLASSES:
                        klass = '-- All Classes --'
                    elif klass == PSSClasifierEvaluator.ALL_CLASSES_STRICT:
                        klass = '-- All Classes (strict) --'
                    else:
                        if not is_best_epoch and not is_last_epoch and not self.report_epoch_scores:
                            continue
                    row_tuples = \
                        [("Epoch", epoch)] + \
                        [("Last Epoch", "Yes" if is_last_epoch else "No")] + \
                        [("Best Epoch", "Yes" if is_best_epoch else "No")] + \
                        [("Scope", scope)] + \
                        [("Class", klass)] + \
                        sorted({k: scores[k] for k, v in scores.items()}.items()) + \
                        [(k, str(v)) for k, v in sorted(params.items())] + \
                        [("Hyperparams Json", json.dumps(params) if len(rows_tuples) == 0 else "")]
                    rows_tuples.append(row_tuples)

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
                 tuner_score_getter=lambda evaluations: max([e['f1'] or 0 for e in evaluations]),
                 tuner_results_getter=extract_classifier_evaluator_results,
                 task_name=''):
        self.task_name = task_name
        self.dump_models = dump_models
        self.fit_kwargs = None
        self.tuner_results_getter = tuner_results_getter
        self.tuner_score_getter = tuner_score_getter
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
