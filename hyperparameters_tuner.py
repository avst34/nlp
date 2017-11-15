import random
import csv
import threading
import json
from collections import OrderedDict


def build_csv_row(params, result):
    assert isinstance(result, HyperparametersTuner.ExecutionResult)
    row_tuples = \
        [("Tuner Score", result.score)] + \
        sorted(result.result_data.items()) + \
        sorted(params.items()) + \
        [("Hyperparams Json", json.dumps(params))]

    headers = [x[0] for x in row_tuples]
    row = [x[1] for x in row_tuples]
    row = ['%1.4f' % x if type(x) is float else x for x in row]
    return headers, [row]

class HyperparametersTuner:

    class ParamSettings:

        def __init__(self, name, values):
            self.name = name
            self.values = values

        def sample(self):
            return random.choice(self.values)

    class ExecutionResult:
        def __init__(self, result_data, score):
            self.score = score
            self.result_data = result_data

    def __init__(self, params_settings, executor, csv_row_builder=build_csv_row):
        assert all([isinstance(ps, HyperparametersTuner.ParamSettings) for ps in params_settings])
        self.params_settings = params_settings
        self.executor = executor
        self.csv_row_builder = csv_row_builder
        self.csv_lock = threading.Lock()
        self.csv_writer = None
        self.emitted_csv_rows = None
        self.emitted_results = None

    def sample_params(self):
        return OrderedDict({
            (setting.name, setting.sample()) for setting in self.params_settings
        })


    def sample_execution(self, params=None):
        params = params or self.sample_params()
        result = self.executor(params)
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        self.emit_result_to_csv(params, result)
        return params, result

    def sample_executions(self, n_executions, mapper=map):
        return mapper(lambda _: self.sample_execution(), range(n_executions))

    def tune(self, results_csv_path, n_executions=30, mapper=map):
        with open(results_csv_path, 'w', newline='', buffering=1) as csv_f:
            self.csv_writer = csv.writer(csv_f)
            self.emitted_csv_rows = 0
            self.emitted_results = 0
            results = self.sample_executions(n_executions, mapper)
            best_params, best_result = max(results, key=lambda result: result[1].score)
            return best_params, best_result

    def emit_result_to_csv(self, params, result):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        with self.csv_lock:
            headers, rows = self.csv_row_builder(params, result)
            headers = ['Execution ID'] + headers
            rows = [
                [self.emitted_results + 1] + row for row in rows
            ]
            if self.emitted_csv_rows == 0:
                self.csv_writer.writerow(headers)
                self.emitted_csv_rows += 1
            for row in rows:
                self.csv_writer.writerow(row)
                self.emitted_csv_rows += 1
            self.emitted_results += 1