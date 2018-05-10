from enum import Enum

class TYPES(Enum):
    NOUN_SUPERSENSE = 'noun_supersense'
    VERB_SUPERSENSE = 'verb_supersense'
    PREPOSITION_SUPERSENSE = 'preposition_supersense'

NOUN_SUPERSENSES = (
    'ANIMAL',
    'ARTIFACT',
    'ATTRIBUTE',
    'ACT',
    'COGNITION',
    'COMMUNICATION',
    'BODY',
    'EVENT',
    'FEELING',
    'FOOD',
    'GROUP',
    'LOCATION',
    'MOTIVE',
    'NATURAL OBJECT',
    'OBJECT',
    'OTHER',
    'PERSON',
    'PHENOMENON',
    'PLANT',
    'POSSESSION',
    'PROCESS',
    'QUANTITY',
    'RELATION',
    'SHAPE',
    'STATE',
    'SUBSTANCE',
    'TIME',
)

VERB_SUPERSENSES = (
    'body',
    'change',
    'cognition',
    'communication',
    'competition',
    'consumption',
    'contact',
    'creation',
    'emotion',
    'motion',
    'perception',
    'possession',
    'social',
    'stative',
)

PREPOSITION_SUPERSENSES = (
    'Circumstance',
    'Temporal',
    'Time',
    'Participant',
    'Causer',
    'Agent',
    'StartTime',
    'EndTime',
    'Co-Agent',
    'Theme',
    'Configuration',
    'Identity',
    'Species',
    'Gestalt',
    'Possessor',
    'Frequency',
    'Co-Theme',
    'Duration',
    'Topic',
    'Characteristic',
    'Stimulus',
    'Possession',
    'Experiencer',
    'Part/Portion',
    'Interval',
    'Locus',
    'Source',
    'Goal',
    'Path',
    'Originator',
    'Recipient',
    'Stuff',
    'Accompanier',
    'InsteadOf',
    'Direction',
    'Cost',
    'Extent',
    'Beneficiary',
    'Means',
    'Whole',
    'Instrument',
    'ComparisonRef',
    'RateUnit',
    'Quantity',
    'Manner',
    'Approximator',
    'Explanation',
    'SocialRel',
    'Purpose',
    'OrgRole'
)

PSS_TREE = {
    'Circumstance': {
        'Temporal': {
            'Time': {
                'StartTime': {},
                'EndTime': {}},
            'Frequency': {},
            'Duration': {},
            'Interval': {}},
        'Locus': {
            'Source': {},
            'Goal': {}},
        'Path': {
            'Direction': {},
            'Extent': {}},
        'Means': {},
        'Manner': {},
        'Explanation': {
            'Purpose': {}}},
    'Participant': {
        'Causer': {
            'Agent': {
                'Co-Agent': {}}},
        'Theme': {
            'Co-Theme': {},
            'Topic': {}},
        'Stimulus': {},
        'Experiencer': {},
        'Originator': {},
        'Recipient': {},
        'Cost': {},
        'Beneficiary': {},
        'Instrument': {}},
    'Configuration': {
        'Identity': {},
        'Species': {},
        'Gestalt': {
            'Possessor': {},
            'Whole': {}},
        'Characteristic': {
            'Possession': {},
            'Part/Portion': {
                'Stuff': {}}},
        'Accompanier': {},
        'InsteadOf': {},
        'ComparisonRef': {},
        'RateUnit': {},
        'Quantity': {
            'Approximator': {}},
        'SocialRel': {
            'OrgRole': {}}},
}

PSS_PARENTS = {}
PSS_DEPTH = {}

queue = [[ss,None,PSS_TREE[ss]] for ss in PSS_TREE]
while queue:
    ss, par, descendants = queue.pop()
    PSS_PARENTS[ss] = par
    PSS_DEPTH[ss] = 1 if par is None else PSS_DEPTH[par] + 1
    queue.extend([[ch,ss,descendants[ch]] for ch in descendants])
del queue, ss, par, descendants
