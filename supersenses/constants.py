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

