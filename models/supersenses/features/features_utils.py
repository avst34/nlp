import string

from models.supersenses.features.feature import NoneFeatureValue


def raise_none(f):
    def f_wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        if r is None:
            raise NoneFeatureValue()
        return r
    return f_wrapped


@raise_none
def get_tok(sent, ind):
    if ind is None:
        return None
    if ind >= len(sent):
        raise Exception("Index out of bounds")
    return sent[ind]

@raise_none
def get_gov(tok, sent):
    return get_tok(sent, tok.gov_ind)

@raise_none
def get_obj(tok, sent):
    return get_tok(sent, tok.obj_ind)

@raise_none
def get_parent(tok, sent):
    parent = sent[tok.ud_head_ind]
    if parent != tok:
        return parent
    else:
        return None


@raise_none
def get_grandparent(tok, sent):
    parent = get_parent(tok, sent)
    if parent:
        return get_parent(parent, sent)


@raise_none
def get_children(tok, sent):
    return [t for t in sent if t.ud_head_ind == tok.ind and t != tok]


@raise_none
def get_child_of_type(tok, sent, child_type):
    children = [tok for tok in get_children(tok, sent) if tok.ud_dep == child_type]
    return children[0] if len(children) else None


def is_capitalized(tok):
    return tok.token and tok.token[0] in string.ascii_uppercase