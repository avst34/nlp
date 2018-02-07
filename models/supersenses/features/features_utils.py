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
def get_parent(tok, sent, deps_from):
    parent = sent[tok.head_ind(deps_from)]
    if parent != tok:
        return parent
    else:
        return None


@raise_none
def get_grandparent(tok, sent, deps_from):
    parent = get_parent(tok, sent, deps_from)
    if parent:
        return get_parent(parent, sent, deps_from)


@raise_none
def get_children(tok, sent, deps_from):
    return [t for t in sent if t.head_ind(deps_from) == tok.ind and t != tok]


@raise_none
def get_child_of_type(tok, sent, child_type, deps_from):
    children = [tok for tok in get_children(tok, sent, deps_from) if tok.dep(deps_from) == child_type]
    return children[0] if len(children) else None


def is_capitalized(tok):
    return tok.token and tok.token[0] in string.ascii_uppercase