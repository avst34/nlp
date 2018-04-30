from models.hcpd import vocabs
from wordnet_verbnet import get_noun_hypernyms


def preprocess_sentence(tokens):
    return {
        'hypernyms': [
            get_noun_hypernyms(tok, vocabs.HYPERNYMS.all_words())
            for tok in tokens
        ]
    }
