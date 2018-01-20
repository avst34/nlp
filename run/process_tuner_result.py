# 1. find best execution, group by task
# 2. for each execution:
#     2.1 get the hyperparameters json
#     2.2 train
#     2.3 create psseval tsv for predictor
from utils import csv_to_objs


def process_tuner_results(tuner_results_csv_path):
    results = csv_to_objs(tuner_results_csv_path)
    execution_params = {}
    for result in results:
        if result['Hyperparams Json']:
            execution_params[result['Execution ID']] = result['Hyperparams Json']

    best_results_by_task = {}
    task_param_names = [ps.name for ps in params_settings if ps.task_param]
    for result in results:
        if result['Best Epoch'] != 'Yes':
            continue
        task_key = sorted([(key, value) for key, value in result if key in task_param_names])
        best_score = best_results_by_task.get(task_key, {}).get('score', 0)
        cur_score = float(result['Tuner Score'])
        if best_score > cur_score:
            best_results_by_task[task_key] = {
                'result': result,
                'score': cur_score
            }

