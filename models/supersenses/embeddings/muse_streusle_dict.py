import os

from models.supersenses.embeddings import MUSE_EN

zh_en = {}
with open(os.path.dirname(__file__) + '/zh-en.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip()
        if l:
            zh, en = l.split()
            zh_en[zh] = [en]
with open(os.path.dirname(__file__) + '/cedict_ts.u8', 'r') as f:
    for l in f.readlines():
        l = l.strip()
        if l and not l.startswith('#'):
            zhs = l[:l.index('[')].strip().split()
            zhs = [z for z in zhs if zhs]
            trans = l[l.index(']') + 1:].split('/')
            trans = [t.strip() for t in trans if t.strip()]
            for zh in zhs:
                zh_en[zh] = trans


class Embeddings(object):

    def get(self, word):
        v = MUSE_EN.get(word)
        if v is None:
            for t in zh_en.get(word, []):
                v = MUSE_EN.get(t)
                if v is not None:
                    break
        return v

    def dim(self):
        d = MUSE_EN.dim()
        return d

MUSE_STREUSLE_DICT = Embeddings()

