from .muse_en import MUSE_EN
from .muse_zh import MUSE_ZH


class Embeddings(object):

    def get(self, word):
        v = MUSE_EN.get(word)
        return v if v is not None else MUSE_ZH.get(word)

    def dim(self):
        d = MUSE_EN.dim()
        assert d == MUSE_ZH.dim()
        return d

MUSE_STREUSLE = Embeddings()

