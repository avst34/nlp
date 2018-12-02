from functools import lru_cache

import h5py
import numpy as np

from datasets.streusle_v4 import StreusleLoader


class EmbeddingsHDF5Reader(object):

    def __init__(self, hdf5_path, cache_size=2**12):
        self.hdf5_path = hdf5_path
        self.hdf = h5py.File(self.hdf5_path, 'r', libver='latest')
        self.getter = lru_cache(maxsize=cache_size)(self.get)
        self._dim = None

    def get(self, word):
        word = word.encode().hex()
        try:
            return self.hdf[word][:]
        except:
            return None

    def dim(self):
        if self._dim is None:
            self._dim = self.hdf[next(iter(self.hdf.keys()))].size
        return self._dim



class EmbeddingsHDF5Writer(object):

    def __init__(self, hdf5_path, dim, dtype=np.dtype(float)):
        self.dtype = dtype
        self.dim = dim
        self.hdf5_path = hdf5_path
        self.hdf = h5py.File(self.hdf5_path, 'w')

    def set(self, word, embeddings):
        # print('"' + word + '"', len(embeddings))
        encoded = word.encode().hex()
        try:
            d = self.hdf.create_dataset(encoded, (self.dim,), dtype=self.dtype)
        except:
            print(word)
            raise
        d[:] = embeddings

    @staticmethod
    def from_muse_format(muse_pathes, fname, words=None):
        writer = None
        n = None
        c = 0
        dim = None
        skipped = []
        if type(muse_pathes) is not list:
            muse_pathes = [muse_pathes]
        for muse_path in muse_pathes:
            with open(muse_path) as musef:
                for line in musef:
                    c += 1
                    if c % 1000 == 0:
                        print(c*100/n)

                    line = line.strip()
                    if n is None:
                        n, dim = [int(x) for x in line.split()]
                        writer = EmbeddingsHDF5Writer(fname, dim)
                    else:
                        toks = line.split()
                        if len(toks) == dim:
                            toks = [' '] + toks
                        word = ' '.join(toks[:len(toks) - dim])
                        try:
                            embedding = [float(x) for x in toks[-dim:]]
                        except:
                            print(line)
                            raise
                        if words and word not in words:
                            continue
                        try:
                            writer.set(word, embedding)
                        except Exception as e:
                            if "name already exists" in repr(e):
                                print('SKIPPING: ', word)
                                skipped.append(word)
                print('skipped:', len(skipped), skipped)



if __name__ == '__main__':
    # EmbeddingsHDF5Writer.from_muse_format('/cs/usr/aviramstern/lab/muse/embeddings/vectors-en.txt', '/cs/usr/aviramstern/lab/muse/embeddings/vectors-en.hd5')
    # EmbeddingsHDF5Writer.from_muse_format('/cs/usr/aviramstern/lab/muse/embeddings/vectors-en.txt', '/cs/usr/aviramstern/lab/muse/embeddings/vectors-en-2.hd5')
    records = StreusleLoader().load()
    records += StreusleLoader().load(conllulex_path='/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/chinese/lp.chinese.all.json', input_format='json')
    words = {t.token for rec in records for t in rec.tagged_tokens}
    print(len(words), 'words')
    EmbeddingsHDF5Writer.from_muse_format(
        ['/cs/usr/aviramstern/lab/muse/embeddings/vectors-en.txt',
         '/cs/usr/aviramstern/lab/muse/embeddings/vectors-zh.txt'],
                                          '/cs/usr/aviramstern/lab/muse/embeddings/vectors-muse-en-zh-streusle.hd5'
        , words)
