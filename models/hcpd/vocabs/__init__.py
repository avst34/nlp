import json
import os

from .gold_pos import GOLD_POS
from .words import WORDS
from .hypernyms import HYPERNYMS
from .pss import PSS


with open(os.path.dirname(__file__) + '/words_to_lemmas.json', 'r', encoding='utf-8') as out_f:
    WORDS_TO_LEMMAS = json.load(out_f)
