def _get_vocab(type, name):
    return globals()[type.upper() + "_" + name]

from .ud_pos import UD_POS
from .spacy_pos import SPACY_POS
from .corenlp_pos import CORENLP_POS

def POS(type):
    return _get_vocab(type, 'POS')

from .spacy_ner import SPACY_NER
from .corenlp_ner import CORENLP_NER

def NER(type):
    return _get_vocab(type, 'NER')

from .preps import PREPS
from .pss import PSS
from .pss_without_none import PSS_WITHOUT_NONE

from .spacy_deps import SPACY_DEPS
from .ud_deps import UD_DEPS
from .corenlp_deps import CORENLP_DEPS

def DEPS(type):
    return _get_vocab(type, 'DEPS')

from .tokens import TOKENS

from .ud_lemmas import UD_LEMMAS
from .spacy_lemmas import SPACY_LEMMAS
from .corenlp_lemmas import CORENLP_LEMMAS

def LEMMAS(type):
    return _get_vocab(type, 'LEMMAS')

from .boolean import BOOLEAN