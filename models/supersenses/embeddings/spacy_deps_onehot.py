from models.supersenses import vocabs
from models.supersenses.build_onehot_embedding import build_onehot_embedding

SPACY_DEPS_ONEHOT = build_onehot_embedding(vocabs.spacy_deps)
