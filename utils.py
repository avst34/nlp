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

def parse_conll(conll):
    sents = [x for x in conll.split('\n\n') if x]
    parsed_sents = []
    for sent in sents:
        lines = sent.split('\n')
        parsed = {
            'metadata': dict([x[2:].split(' = ') for x in lines if x.startswith('#')]),
            'tokens': [
                {
                    'id': int(cols[0]),
                    'token': cols[1],
                    'lemma': cols[2],
                    'pos': cols[3],
                    'ner': cols[4],
                    'head': int(cols[5]),
                    'deprel': cols[6],
                } for l in lines if l and l[0] != '#' for cols in [[x if x != '_' else None for x in l.split('\t')]]
            ]
        }
        parsed_sents.append(parsed)

    return parsed_sents

def parse_conll_file(conll_file):
    with open(conll_file, 'r') as f:
        conll = f.read()
    return parse_conll(conll)

def parse_conllx(conllx):
    sents = [x for x in conllx.split('\n\n') if x]
    parsed_sents = []
    for sent in sents:
        lines = sent.split('\n')
        parsed = {
            'metadata': dict([x[2:].split(' = ') for x in lines if x.startswith('#')]),
            'tokens': [
                {
                    'id': int(cols[0]),
                    'token': cols[1],
                    'unk1': cols[2],
                    'pos': cols[3],
                    'unk2': cols[4],
                    'unk3': cols[5],
                    'head': int(cols[6]),
                } for l in lines if l and l[0] != '#' for cols in [[x if x != '_' else None for x in l.split('\t')]]
            ]
        }
        parsed_sents.append(parsed)

    return parsed_sents

def parse_conllx_file(conllx_file):
    with open(conllx_file, 'r') as f:
        conllx = f.read()
    return parse_conllx(conllx)

