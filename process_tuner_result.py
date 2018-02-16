# 1. find best execution, group by task
# 2. for each execution:
#     2.1 get the hyperparameters json
#     2.2 train
#     2.3 create psseval tsv for predictor
from datasets.streusle_v4 import StreusleLoader, sys
import os
import json

from evaluators.streusle_evaluator import StreusleEvaluator
from models.general.simple_conditional_multiclass_model.model import MostFrequentClassModel
from models.general.simple_conditional_multiclass_model.streusle_integration import \
    streusle_record_to_most_frequent_class_model_sample
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
from utils import csv_to_objs

STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'


def process_tuner_results(tuner_results_csv_path, output_dir=None):
    nn_output_dir = output_dir + '/nn'
    results = csv_to_objs(tuner_results_csv_path)
    execution_params = {}
    for result in results:
        if result['Hyperparams Json']:
            execution_params[result['Execution ID']] = json.loads(result['Hyperparams Json'])

    best_results_by_task = {}
    for result in results:
        if result['Best Epoch'] != 'Yes':
            continue
        task_key = result['Task']
        best_score = best_results_by_task.get(task_key, {}).get('score', 0)
        cur_score = float(result['Tuner Score'])
        if cur_score > best_score:
            best_results_by_task[task_key] = {
                'execution_id': result['Execution ID'],
                'result': result,
                'score': cur_score
            }


    for task, best_result in best_results_by_task.items():
    # for task, best_result in {'goldid.goldsyn': best_results_by_task['goldid.goldsyn']}.items():
        print("Best results for " + task + ": " + str(best_result['score']))
        params = execution_params[best_result['execution_id']]
        if not params.get("use_lexcat"):
            params["use_lexcat"] = True
            params["lexcat_embd_dim"] = 3

        # params['epochs'] = 1
        model = LstmMlpSupersensesModel(LstmMlpSupersensesModel.HyperParameters(**params))
        evaluate_model_on_task(task, model, streusle_record_to_lstm_model_sample, nn_output_dir)

def evaluate_most_frequent_baseline_model(output_dir):
    mfc_output_dir = output_dir + '/mfc'
    tasks = [idt + '.' + syn for idt in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    for task in tasks:
        model = MostFrequentClassModel(['lemma'], include_empty=False, n_labels_to_predict=2)
        evaluate_model_on_task(task, model, streusle_record_to_most_frequent_class_model_sample, mfc_output_dir)

def evaluate_model_on_task(task, model, streusle_to_model_sample, output_dir):
    loader = StreusleLoader()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    task_output = output_dir + '/' + task
    if not os.path.exists(task_output):
        os.mkdir(task_output)

    train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
    dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
    test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')
    train_samples = [streusle_to_model_sample(r) for r in train_records]
    dev_samples = [streusle_to_model_sample(r) for r in dev_records]

    # try:
    #     predictor = LstmMlpSupersensesModel.load(task_output + '/model')
    #     print('Loaded existing predictor')
    # except:
    predictor = model.fit(train_samples, dev_samples, show_progress=False)
    print("Training done")
    predictor.save(task_output + '/model')
    print("Save model done")
    evaluator = StreusleEvaluator(predictor)
    for stype, records in [('train', train_records), ('dev', dev_records), ('test', test_records)]:
        print('Evaluating ', task, stype)
        evaluator.evaluate(records,
                           output_tsv_path=task_output + '/' + task + '.' + stype + '.psseval.tsv',
                           gold_fname_out=task_output + '/' + task + '.' + stype + '.gold.json',
                           sys_fname_out=task_output + '/' + task + '.' + stype + '.sys.' + task.split('.')[0] + '.json',
                           streusle_record_to_model_sample=streusle_to_model_sample)
    print("Evaluation done")




if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r'c:\temp\results.csv'

    output_dir = os.path.dirname(path) + '/best_results'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    process_tuner_results(path, output_dir)
    evaluate_most_frequent_baseline_model(output_dir)