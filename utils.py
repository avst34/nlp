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