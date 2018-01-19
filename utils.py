import csv


def clear_nones(obj):
    if type(obj) is list:
        return [x for x in obj if x is not None]
    if type(obj) is dict:
        return {x:y for x,y in obj.items() if y is not None}

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def update_dict(d, with_d, del_keys=None):
    del_keys = del_keys or set()
    d = {k: v for k, v in d.items() if k not in del_keys}
    d.update(with_d)
    return d

def csv_to_objs(csv_file_path):
    keys = []
    objs = []
    with open(csv_file_path, 'r') as csv_f:
        csv_reader = csv.reader(csv_f)
        for ind, row in enumerate(csv_reader):
            if ind == 0:
                keys = row
            else:
                if row:
                    objs.append({k: row[i] for i, k in enumerate(keys)})
    return objs

