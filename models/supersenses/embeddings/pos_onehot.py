from models.supersenses import vocabs
from models.supersenses.build_onehot_embedding import build_onehot_embedding

POS_ONEHOT = build_onehot_embedding(vocabs.POS)
