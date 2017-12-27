from models.supersenses.features.feature import NoneFeatureValue


def raise_none(f):
    def f_wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        if r is None:
            raise NoneFeatureValue()
        return r
    return f_wrapped


@raise_none
def get_parent(tok, sent, field):
    parent = sent[getattr(tok, field)]
    if parent != tok:
        return parent
    else:
        return None


@raise_none
def get_grandparent(tok, sent, field):
    parent = get_parent(tok, sent, field)
    if parent:
        return get_parent(parent, sent, field)


@raise_none
def get_children(tok, sent, field):
    return [t for t in sent if getattr(t, field) == tok.ind and t != tok]


@raise_none
def get_child_of_type(tok, sent, child_type, head_field, type_field):
    children = [tok for tok in get_children(tok, sent, head_field) if getattr(tok, type_field) == child_type]
    return children[0] if len(children) else None

