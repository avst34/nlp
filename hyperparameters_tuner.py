import random

class HyperparametersTuner:

    class ParamSettings:

        def __init__(self, name, values):
            self.name = name
            self.values = values

        def sample(self):
            return random.choice(self.values)

    class ExecutionResult:
        def __init__(self, params, result):
            self.result = result
            self.params = params

    def __init__(self, params_settings, executor):
        self.params_settings = params_settings
        self.executor = executor

    def sample_params(self):
        return {
            setting.name: setting.sample() for setting in self.params_settings
        }

    def sample_execution(self, params=None):
        params or self.sample_params()
        result = self.executor(params)
        return HyperparametersTuner.ExecutionResult(params=params, result=result)

    def sample(self, n_executions, mapper=map):
        return mapper(lambda _: self.sample_execution(), range(n_executions))