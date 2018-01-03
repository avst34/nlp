from models.supersenses import vocabs
from models.supersenses.build_onehot_embedding import build_onehot_embedding

SPACY_POS_ONEHOT = build_onehot_embedding(vocabs.SPACY_POS)
