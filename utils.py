def clear_nones(obj):
    if type(obj) is list:
        return [x for x in obj if x is not None]
    if type(obj) is dict:
        return {x:y for x,y in obj.items() if y is not None}

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)