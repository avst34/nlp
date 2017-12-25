def get_parent(tok, sent, field):
    parent = sent[getattr(tok, field)]
    if parent != tok:
        return parent
    else:
        return None

def get_grandparent(tok, sent, field):
    parent = get_parent(tok, sent, field)
    if parent:
        return get_parent(parent, sent)
