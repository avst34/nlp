import hashlib
import time, datetime
import random
import csv
import threading
import json
import random
from collections import OrderedDict
from lockfile import Lockfile


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

        def __init__(self, name, values, enabled=True):
            self.name = name
            self.values = values
            self.enabled = enabled

        def sample(self):
            return random.choice(self.values)

    class ExecutionResult:
        def __init__(self, result_data, score):
            self.score = score
            self.result_data = result_data

    def __init__(self, params_settings, executor, results_csv_path, csv_row_builder=build_csv_row, shared_csv=False, lock_file_path=None):
        assert all([isinstance(ps, HyperparametersTuner.ParamSettings) for ps in params_settings])
        self.shared_csv = shared_csv
        if shared_csv:
            assert lock_file_path is not None
            self.csv_lock = Lockfile(lock_file_path, verbose=True)
        else:
            self.csv_lock = threading.Lock()
        self.params_settings = params_settings
        self.executor = executor
        self.csv_row_builder = csv_row_builder
        self.csv_file_path = None
        self.emitted_csv_rows = None
        self.emitted_result1s = None
        self.executor_id = self.gen_id()

        self.csv_file_path = results_csv_path
        self.emitted_csv_rows = 0
        self.emitted_results = 0


    def sample_params(self):
        return OrderedDict({
            (setting.name, setting.sample()) for setting in self.params_settings
        })

    def sample_execution(self, params=None):
        params = params or self.sample_params()
        enabled_params = {p: v for p, v in params.items() if self.param(p).enabled}
        start_time = time.time()
        result = self.executor(enabled_params)
        execution_time = time.time() - start_time
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        self.emit_result_to_csv(params, result, execution_time)
        return params, result

    def sample_executions(self, n_executions, mapper=map):
        return mapper(lambda _: self.sample_execution(), range(n_executions))

    def param(self, name):
        for ps in self.params_settings:
            if ps.name == name:
                return ps
        raise Exception("Unknown tuner param: " + name)

    def tune(self, n_executions=30, mapper=map):
        results = self.sample_executions(n_executions, mapper)
        best_params, best_result = max(results, key=lambda result: result[1].score)
        return best_params, best_result

    def gen_id(self):
        return hashlib.md5(str(random.random()).encode()).digest()[:8].hex()

    def gen_execution_id(self):
        if self.shared_csv:
            return self.gen_id()
        else:
            return self.emitted_results + 1

    def emit_result_to_csv(self, params, result, execution_time_secs):
        assert isinstance(result, HyperparametersTuner.ExecutionResult)
        execution_id = self.gen_execution_id()
        open_flags = 'a' if self.shared_csv or self.emitted_results > 0 else 'w'
        with self.csv_lock:
            with open(self.csv_file_path, open_flags) as csv_f:
                csv_writer = csv.writer(csv_f)
                headers, rows = self.csv_row_builder(params, result)
                headers = ['Time', 'Total Execution Time', 'Executor ID', 'Execution ID'] + headers
                rows = [
                    [time.strftime("%Y-%m-%d %H:%M:%S"), "%02dd%02dh%02dm%02ds" % (int(execution_time_secs / (24*60*60)),
                                                                               int(execution_time_secs % (24*60*60) / (60*60)),
                                                                               int(execution_time_secs % (60*60) / 60),
                                                                               int(execution_time_secs % 60),
                                                                               ),
                     self.executor_id, execution_id] + row for row in rows
                ]
                if csv_f.tell() == 0:
                    csv_writer.writerow(headers)
                    self.emitted_csv_rows += 1
                for row in rows:
                    csv_writer.writerow(row)
                    self.emitted_csv_rows += 1
                self.emitted_results += 1