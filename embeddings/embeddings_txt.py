import json
from functools import lru_cache


class EmbeddingsTxtReader(object):

    def __init__(self, txt_path, cache_size=10000):
        self.txt_path = txt_path
        self.inds_path = txt_path + '.inds'
        self.f = open(self.txt_path, 'rb')
        self._dim = None
        self.chunk_size = None
        self.getter = lru_cache(maxsize=cache_size)(self.get)
        self.load()

    def read_chunk(self, ind):
        self.f.seek(ind * (self.chunk_size or 300))
        return self.f.read(self.chunk_size or 300).rstrip(b'\x00').decode('utf8')

    def load(self):
        with open(self.inds_path) as f:
            self.inds = json.load(f)

        l = self.read_chunk(0)
        self.chunk_size, self._dim = [int(x) for x in l.split()]

    def get(self, word):
        ind = self.inds.get(word)
        if ind is None:
            return None

        line = self.read_chunk(ind)
        toks = line.split()
        fword = ' '.join(toks[:-self._dim])
        assert fword == word, "%s != %s" % (fword, word)
        vec = [float(fl) for fl in toks[-self._dim:]]
        return vec

    def dim(self):
        return self._dim