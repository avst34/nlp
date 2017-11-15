from evaluators.classifier_evaluator import ClassifierEvaluator
from hyperparameters_tuner import HyperparametersTuner
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
import json

def extract_classifier_evaluator_results(evaluation):
    return evaluation

def build_csv_row(params, result):
    assert isinstance(result, HyperparametersTuner.ExecutionResult)
    class_scores = result.result_data['class_scores']
    classes_ordered = sorted(class_scores.keys())
    if ClassifierEvaluator.ALL_CLASSES in classes_ordered:
        classes_ordered.remove(ClassifierEvaluator.ALL_CLASSES)
        classes_ordered = [ClassifierEvaluator.ALL_CLASSES] + classes_ordered

    rows_tuples = []
    for klass in classes_ordered:
        scores = class_scores[klass]
        if klass == ClassifierEvaluator.ALL_CLASSES:
            klass = 'All Classes'
        row_tuples = \
            [("Tuner Score", result.score)] + \
            [("Class", klass)] + \
            sorted({k: scores[k] for k, v in scores.items()}.items()) + \
            sorted(params.items()) + \
            [("Hyperparams Json", json.dumps(params))]
        rows_tuples.append(row_tuples)

    headers = [x[0] for x in rows_tuples[0]]
    row = [[x[1] for x in row_tuples] for row_tuples in rows_tuples]
    return headers, row

class LstmMlpSupersensesModelHyperparametersTuner:

    def __init__(self, token_vocab=None,
                 pos_vocab=None,
                 dep_vocab=None,
                 ss_vocab=None,
                 supersense_vocab=None,
                 token_embd=None,
                 pos_embd=None,
                 dep_embd=None):
        self.init_params = {
            'ss_vocab': ss_vocab,
            'token_vocab': token_vocab,
            'pos_vocab': pos_vocab,
            'dep_vocab': dep_vocab,
            'supersense_vocab': supersense_vocab,
            'token_embd': token_embd,
            'pos_embd': pos_embd,
            'dep_embd': dep_embd
        }
        self.fit_params = None
        self.tuner_results_getter = None
        self.tuner_score_getter = None

    def _execute(self, hyperparameters):
        lstm_mlp_model = LstmMlpSupersensesModel(hyperparameters=LstmMlpSupersensesModel.HyperParameters(**hyperparameters), **self.init_params)
        lstm_mlp_model.fit(**self.fit_params)
        return HyperparametersTuner.ExecutionResult(
            result_data=self.tuner_results_getter(lstm_mlp_model.test_set_evaluation),
            score=self.tuner_score_getter(lstm_mlp_model.test_set_evaluation)
        )

    def tune(self, samples, results_csv_path, n_executions=30, show_progress=True, show_epoch_eval=True,
             evaluator=ClassifierEvaluator(), tuner_score_getter=lambda evaluation: evaluation['f1'],
             tuner_results_getter=extract_classifier_evaluator_results):
        assert evaluator is not None
        self.tuner_results_getter = tuner_results_getter
        self.tuner_score_getter = tuner_score_getter
        self.fit_params = {
            'samples': samples,
            'show_progress': show_progress,
            'show_epoch_eval': show_epoch_eval,
            'evaluator': evaluator
        }
        PS = HyperparametersTuner.ParamSettings
        tuner = HyperparametersTuner([
            PS(name='use_token', values=[True]),
            PS(name='use_pos', values=[False]),
            PS(name='use_dep', values=[True]),
            PS(name='token_embd_dim', values=[300]),
            PS(name='pos_embd_dim', values=[30]),
            PS(name='dep_embd_dim', values=[30]),
            PS(name='mlp_layers', values=[2]),
            PS(name='mlp_layer_dim', values=[30]),
            PS(name='lstm_h_dim', values=[30]),
            PS(name='num_lstm_layers', values=[2]),
            PS(name='is_bilstm', values=[True]),
            PS(name='use_head', values=[True]),
            PS(name='mlp_dropout_p', values=[0.1]),
            PS(name='epochs', values=[30]),
            PS(name='validation_split', values=[0.3]),
        ], executor=self._execute, csv_row_builder=build_csv_row)
        best_params, best_results = tuner.tune(results_csv_path, n_executions)
        return best_params, best_results
