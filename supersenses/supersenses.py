from supersenses import constants

NOUN_SUPERSENSES_SET = set(constants.NOUN_SUPERSENSES)
PREPOSITION_SUPERSENSES_SET = set(constants.PREPOSITION_SUPERSENSES)
VERB_SUPERSENSES_SET = set(constants.VERB_SUPERSENSES)

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
