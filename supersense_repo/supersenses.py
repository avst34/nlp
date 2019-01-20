from supersense_repo import constants
from supersense_repo.constants import PSS_PARENTS

NOUN_SUPERSENSES_SET = set(constants.NOUN_SUPERSENSES)
PREPOSITION_SUPERSENSES_SET = set(constants.PREPOSITION_SUPERSENSES)
VERB_SUPERSENSES_SET = set(constants.VERB_SUPERSENSES)

SUPERSENSES_SET = NOUN_SUPERSENSES_SET.union(PREPOSITION_SUPERSENSES_SET).union(VERB_SUPERSENSES_SET)

def get_supersense_type(supersense):
    ss_type = None
    if supersense in NOUN_SUPERSENSES_SET:
        ss_type = constants.TYPES.NOUN_SUPERSENSE
    elif supersense in PREPOSITION_SUPERSENSES_SET:
        ss_type = constants.TYPES.PREPOSITION_SUPERSENSE
    elif supersense in VERB_SUPERSENSES_SET:
        ss_type = constants.TYPES.VERB_SUPERSENSE
    else:
        print("WARNING: Unknown supersense:" + supersense)
    return ss_type


def filter_non_supersense(maybe_supersense):
    if maybe_supersense in SUPERSENSES_SET:
        return maybe_supersense
    else:
        return None

def get_pss_hierarchy(pss):
    psses = []
    while pss:
        psses.append(pss)
        pss = PSS_PARENTS.get(pss)
    return psses

def get_pss_by_depth(pss, depth):
    assert depth > 0
    heirarchy = get_pss_hierarchy(pss)[::-1]
    if depth > len(heirarchy):
        return heirarchy[-1]
    else:
        return heirarchy[depth - 1]

def pss_equal(pss1, pss2, depth):
    assert pss1 in PREPOSITION_SUPERSENSES_SET and pss2 in PREPOSITION_SUPERSENSES_SET, (pss1, pss2)
    return get_pss_by_depth(pss1, depth) == get_pss_by_depth(pss2, depth)


MAX_PSS_DEPTH = max([len(get_pss_hierarchy(pss)) for pss in PREPOSITION_SUPERSENSES_SET])