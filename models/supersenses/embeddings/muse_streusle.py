from models.supersenses.embeddings import MUSE_EN, MUSE_ZH


class Embeddings(object):

    def get(self, word):
        v = MUSE_EN.get(word)
        return v if v is not None else MUSE_ZH.get(word)

    def dim(self):
        d = MUSE_EN.dim()
        assert d == MUSE_ZH.dim()
        return d

MUSE_STREUSLE = Embeddings()

