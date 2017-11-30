from supersenses import constants

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