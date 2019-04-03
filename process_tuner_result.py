# 1. find best execution, group by task
# 2. for each execution:
#     2.1 get the hyperparameters json
#     2.2 train
#     2.3 create psseval tsv for predictor
import json
import os
from collections import defaultdict
from itertools import chain

from datasets.streusle_v4 import StreusleLoader, sys
from datasets.streusle_v4.release.supersenses import coarsen_pss
from evaluators.streusle_evaluator import StreusleEvaluator
from models.general.simple_conditional_multiclass_model.model import MostFrequentClassModel
from models.general.simple_conditional_multiclass_model.streusle_integration import \
    streusle_record_to_most_frequent_class_model_sample
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample
from utils import csv_to_objs

STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'


filters = {
    "all": {},
    "full_syntax": {
        "use_capitalized_word_follows": True,
        "use_ud_xpos": True,
        "use_ud_dep": True,
        "use_ner": True,
        "use_govobj": True,
        "use_lexcat": True
    },
    "elmo_nosyn": {
        "embd_type": "elmo",
        "use_capitalized_word_follows": False,
        "use_ud_xpos": False,
        "use_ud_dep": False,
        "use_ner": False,
        "use_govobj": False,
        "use_lexcat": False,
    },
    "elmo_syn": {
        "embd_type": "elmo"
    },
    "fasttext_nosyn": {
        "embd_type": "fasttext_en",
        "use_capitalized_word_follows": False,
        "use_ud_xpos": False,
        "use_ud_dep": False,
        "use_ner": False,
        "use_govobj": False,
        "use_lexcat": False,
    },
    "fasttext_syn": {
        "embd_type": "fasttext_en",
    },
}

def does_filter_match(filter, row):
    for k, v in filter.items():
        if row[k] != str(v):
            return False
    return True

def process_tuner_results(tuner_results_csv_paths, output_dir=None, task_to_process=None, filter=None):
    nn_output_dir = output_dir + '/nn'
    results = [x for p in tuner_results_csv_paths for x in csv_to_objs(p)]
    execution_params = {}
    for result in results:
        if result['Hyperparams Json']:
            execution_params[result['Execution ID']] = json.loads(result['Hyperparams Json'])

    best_results_by_task = {}
    for result in results:
        if result['Best Epoch'] != 'Yes':
            continue
        task_key = result['Task']
        if task_to_process and task_key != task_to_process:
            continue
        if filter and not does_filter_match(filters[filter], result):
            continue
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
        params['allow_empty_prediction'] = False
        # if not params.get("use_lexcat"):
        #     params["use_lexcat"] = True
        #     params["lexcat_embd_dim"] = 3
        if not params.get("trainer"):
            params["trainer"] = "sgd"

        # params['epochs'] = 1
        model = LstmMlpSupersensesModel(LstmMlpSupersensesModel.HyperParameters(**params))
        evaluate_model_on_task(task, model, streusle_record_to_lstm_model_sample, nn_output_dir, n_times=5, suffix=(".%s" % filter) if filter and filter != 'all' else "")

def evaluate_most_frequent_baseline_model(output_dir):
    mfc_output_dir = output_dir + '/mf'
    tasks = [idt + '.' + syn for idt in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    for task in tasks:
        model = MostFrequentClassModel([], include_empty=False, n_labels_to_predict=2)
        evaluate_model_on_task(task, model, streusle_record_to_most_frequent_class_model_sample, mfc_output_dir, load_elmo=False)

    mfc_output_dir = output_dir + '/mf-prep'
    tasks = [idt + '.' + syn for idt in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    for task in tasks:
        model = MostFrequentClassModel(['lemma'], include_empty=False, n_labels_to_predict=2)
        evaluate_model_on_task(task, model, streusle_record_to_most_frequent_class_model_sample, mfc_output_dir, load_elmo=False)

def evaluate_model_on_task(task, model, streusle_to_model_sample, output_dir, save_model=False, n_times=1, load_elmo=True, suffix=""):
    loader = StreusleLoader(load_elmo=load_elmo)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    task_output = output_dir + '/' + task + suffix
    if not os.path.exists(task_output):
        os.mkdir(task_output)

    with open(task_output + '/hp.json', 'w') as f:
        json.dump(model.hyperparameters.__dict__, f, indent=2)

    train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
    dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
    test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')
    train_samples = [streusle_to_model_sample(r) for r in train_records]
    dev_samples = [streusle_to_model_sample(r) for r in dev_records]
    test_samples = [streusle_to_model_sample(r) for r in test_records]
    out_files = {}
    for t in range(n_times):
        # try:
        #     predictor = LstmMlpSupersensesModel.load(task_output + '/model')
        #     fitted = True
        #     print('Loaded existing predictor')
        # except:
        #     fitted = False
        predictor = model.fit(train_samples, dev_samples, test_samples, show_progress=True)
        print("Training done")
        # if save_model and fitted:
        #     predictor.save(task_output + '/model')
        # print("Save model done")
        evaluator = StreusleEvaluator(predictor)
        for stype, records in [('train', train_records), ('dev', dev_records), ('test', test_records)]:
            print('Evaluating ', task, stype)
            gold_fname = task_output + '/' + task + '.' + stype + '.gold.json'
            sys_fname = task_output + '/' + task + '.' + stype + '.sys.' + str(t) + '.' + task.split('.')[0] + '.json'
            out_tsv = task_output + '/' + task + '.' + stype + '.psseval.' + str(t) + '.tsv'
            out_files[stype] = out_files.get(stype, [])
            out_files[stype].append(out_tsv)
            evaluator.evaluate(records,
                               output_tsv_path=out_tsv,
                               gold_fname_out=gold_fname,
                               sys_fname_out=sys_fname,
                               streusle_record_to_model_sample=streusle_to_model_sample,
                               all_depths=True)
    for stype, files in out_files.items():
        StreusleEvaluator.average_evaluations(files, task_output + '/' + task + '.' + stype + '.psseval.tsv')
    print("Evaluation done")


def parse_psseval(psseval_path):
    with open(psseval_path, 'r') as f:
        rows = [x.strip().split('\t') for x in f.readlines()]
    pf = lambda f: "%1.1f" % (float(f) * 100)
    def ps(score):
        if "+-" in score:
            m, s = [float(x) for x in score.split('+-')]
        else:
            m, s = float(score), 0
        return {
            "mean": pf(m),
            "std": pf(s)
        }

    if 'autoid' in psseval_path:
        return {
            'all': {
                'id': {
                    'p': ps(rows[3][6]),
                    'r': ps(rows[3][7]),
                    'f': ps(rows[3][8])
                },
                'role': {
                    'p': ps(rows[3][10]),
                    'r': ps(rows[3][11]),
                    'f': ps(rows[3][12])
                },
                'fxn': {
                    'p': ps(rows[3][14]),
                    'r': ps(rows[3][15]),
                    'f': ps(rows[3][16])
                },
                'role_fxn': {
                    'p': ps(rows[3][18]),
                    'r': ps(rows[3][19]),
                    'f': ps(rows[3][20])
                },
            },
            'mwe': {
                'id': {
                    'p': ps(rows[8][6]),
                    'r': ps(rows[8][7]),
                    'f': ps(rows[8][8])
                },
                'role': {
                    'p': ps(rows[8][10]),
                    'r': ps(rows[8][11]),
                    'f': ps(rows[8][12])
                },
                'fxn': {
                    'p': ps(rows[8][14]),
                    'r': ps(rows[8][15]),
                    'f': ps(rows[8][16])
                },
                'role_fxn': {
                    'p': ps(rows[8][18]),
                    'r': ps(rows[8][19]),
                    'f': ps(rows[8][20])
                },
            },
            'mwp': {
                'id': {
                    'p': ps(rows[13][6]),
                    'r': ps(rows[13][7]),
                    'f': ps(rows[13][8])
                },
                'role': {
                    'p': ps(rows[13][10]),
                    'r': ps(rows[13][11]),
                    'f': ps(rows[13][12])
                },
                'fxn': {
                    'p': ps(rows[13][14]),
                    'r': ps(rows[13][15]),
                    'f': ps(rows[13][16])
                },
                'role_fxn': {
                    'p': ps(rows[13][18]),
                    'r': ps(rows[13][19]),
                    'f': ps(rows[13][20])
                },
            }
        }
    else:
        assert 'goldid' in psseval_path
        return {
            'all': {
                'role': {
                    'acc': ps(rows[3][2])
                },
                'fxn': {
                    'acc': ps(rows[3][3])
                },
                'role_fxn': {
                    'acc': ps(rows[3][4])
                },
            },
            'mwe': {
                'role': {
                    'acc': ps(rows[8][2])
                },
                'fxn': {
                    'acc': ps(rows[8][3])
                },
                'role_fxn': {
                    'acc': ps(rows[8][4])
                },
            },
            'mwp': {
                'role': {
                    'acc': ps(rows[13][2])
                },
                'fxn': {
                    'acc': ps(rows[13][3])
                },
                'role_fxn': {
                    'acc': ps(rows[13][4])
                },
            }
        }


def build_template_input(results_dir, json_output_path):
    mtypes = ['nn', 'mf', 'mf-prep']
    stypes = ['train', 'dev', 'test']
    tasks = [idt + '.' + syn for idt in [
        'autoid',
        'goldid']
             for syn in ['autosyn', 'goldsyn']]
    d = {}
    for mtype in mtypes:
        for stype in stypes:
            for task in tasks:
                for filter in filters:
                    hp_file_path = results_dir + '/' + mtype + '/' + task + '/model.hp'
                    try:
                        evl = parse_psseval(results_dir + '/' + mtype + '/' + task + ((".%s" % filter) if filter and filter != 'all' else "") + '/' + task + '.' + stype + '.psseval.tsv')
                    except:
                        import traceback
                        traceback.print_exc()
                        continue

                    f_task = (task + ((".%s" % filter) if filter and filter != 'all' else "")).replace('.', '_')
                    d[mtype] = d.get(mtype) or {}
                    d[mtype][f_task] = d[mtype].get(f_task) or {}
                    d[mtype][f_task][stype] = {}
                    d[mtype][f_task][stype]['psseval'] = evl

                    for depth in [1,2,3]:
                        p = results_dir + '/' + mtype + '/' + task + '/' + task + '.' + stype + '.psseval.depth_' + str(depth) + '.tsv'
                        if os.path.exists(p):
                            evl = parse_psseval(p)
                            d[mtype][f_task][stype]['psseval_depth_' + str(depth)] = evl

                    def format_hp(val):
                        conv = {str(10**i): '10^{%d}' % i for i in range(-10, 0)}
                        conv.update({'false': 'No', 'False': 'No', 'true': 'Yes', 'True': 'Yes'})
                        if str(val) in conv:
                            return conv[str(val)]
                        elif type(val) != float or int(val) == val:
                            return val
                        else:
                            return int(val * 100) / 100

                    if os.path.exists(hp_file_path):
                        with open(hp_file_path) as hp_file:
                            d[mtype][f_task]['hp'] = {hp: format_hp(val) for hp, val in json.load(hp_file).items()}


    with open(json_output_path, 'w') as f:
        json.dump(d, f, indent=2)


def build_confusion_matrix(sysf_path, goldf_path, depth):

    mats = {}

    with open(sysf_path) as sysf:
        sys_sents = json.load(sysf)
    with open(goldf_path) as goldf:
        gold_sents = json.load(goldf)

    def coarsen(pss, depth):
        if pss is None:
            return str(pss)
        return coarsen_pss(pss, depth)

    # for filter in ['all', 'mwe', 'mwp']:
    for filter in ['all']:
        def filter_wes(wes):
            return [we for we in wes if we['lexcat'] in ['P', 'PP', 'INF.P', 'POSS', 'PRON.POSS'] and (filter == 'all' or len(we['toknums']) > 1 and (filter != 'mwp' or we['lexcat'] != 'PP'))]

        def format_pair(s1, s2):
            return str(s1) + ',' + str(s2)

        role_mat = defaultdict(lambda: defaultdict(lambda: 0))
        fxn_mat = defaultdict(lambda: defaultdict(lambda: 0))
        exact_mat = defaultdict(lambda: defaultdict(lambda: 0))

        for sys_sent, gold_sent in zip(sys_sents, gold_sents):
            assert sys_sent['sent_id'] == gold_sent['sent_id']
            sys_wes = filter_wes(chain(sys_sent['swes'].values(), sys_sent['smwes'].values()))
            gold_wes = filter_wes(chain(gold_sent['swes'].values(), gold_sent['smwes'].values()))

            for gold_we in gold_wes:
                for sys_we in sys_wes:
                    if set(sys_we['toknums']) == set(gold_we['toknums']):
                        if gold_we['ss'] in ['??', '`$']:
                            continue
                        gold_ss, gold_ss2 = coarsen(gold_we['ss'], depth), coarsen(gold_we['ss2'], depth)
                        sys_ss, sys_ss2 = coarsen(sys_we['ss'], depth), coarsen(sys_we['ss2'], depth)
                        role_mat[gold_ss][sys_ss] += 1
                        fxn_mat[gold_ss2][sys_ss2] += 1
                        exact_mat[format_pair(gold_ss, gold_ss2)][format_pair(sys_ss, sys_ss2)] += 1

        def normalize(mat):
            return {
                k: {
                    k2: {
                        'p': mat[k][k2] / (sum(mat[k].values()) - mat[k][k]) if (sum(mat[k].values()) - mat[k][k]) else 0,
                        'n': mat[k][k2]
                    } for k2 in dict(mat[k]) if k != k2
                } for k in dict(mat)
            }

        role_mat = normalize(role_mat)
        fxn_mat = normalize(fxn_mat)
        exact_mat = normalize(exact_mat)
        mats[filter] = {
            'role': role_mat,
            'fxn': fxn_mat,
            'exact': exact_mat
        }

    return mats



def build_confusion_matrices(results_dir):
    mtypes = ['nn', 'mfc']
    stypes = ['train', 'dev', 'test']
    tasks = [idt + '.' + syn for idt in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    depths = [1,2,3,4]
    d = {}
    for mtype in mtypes:
        output_dir = results_dir + '/' + mtype
        for stype in stypes:
            for task in tasks:
                for depth in depths:
                    task_output = output_dir + '/' + task
                    gold_fname = task_output + '/' + task + '.' + stype + '.gold.json'
                    sys_fname = task_output + '/' + task + '.' + stype + '.sys.' + task.split('.')[0] + '.json'
                    conf = build_confusion_matrix(sys_fname, gold_fname, depth)
                    with open(task_output + '/' + task + '.' + stype + '.conf.depth_' + str(depth) + '.json', 'w') as conf_f:
                        json.dump(conf, conf_f, indent=2, sort_keys=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r'/cs/labs/oabend/aviramstern/full_model.csv'

    output_dir = os.environ.get('BEST_RESULTS_PATH') or r'/cs/labs/oabend/aviramstern/best_results'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    task = sys.argv[-2]
    filter = sys.argv[-1]
    csvs = sys.argv[1:-2]

    print(sys.argv)
    print("task", task)
    print("filter", filter)
    print("csvs", csvs)

    if task != 'template_params':
        process_tuner_results(csvs, output_dir, task_to_process=task, filter=filter)
    else:
        # evaluate_most_frequent_baseline_model(output_dir)
        # build_confusion_matrices(output_dir)
        template_input_path = output_dir + '/template_input.json'
        print("template_input_path", template_input_path)
        build_template_input(output_dir, template_input_path)