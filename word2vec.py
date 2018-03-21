import pickle
import os

class Word2VecModel:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def load_google_model(trained_model_bin_path=os.environ.get('GOOGLE_NEWS_W2V_PATH') or '/cs/labs/oabend/aviramstern/word2vec/GoogleNews-vectors-negative300.bin'):
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(trained_model_bin_path, binary=True)
        return Word2VecModel(model)

    @staticmethod
    def load(f):
        return Word2VecModel.loads(f.read())

    @staticmethod
    def loads(s):
        return Word2VecModel(pickle.loads(s))

    def __getitem__(self, word):
        return self.model[word]

    def __contains__(self, word):
        return item in self.model

    def get(self, word):
        return [float(x) for x in list(self.model[word])] if word in self.model else None

    def collect_missing(self, words):
        return [w for w in words if w not in self.model]

    def dumps(self, words, skip_missing=False):
        if skip_missing:
            words = [w for w in words if w in self.model]
        return pickle.dumps({w: self.model[w] for w in words})

    def dump(self, words, f, skip_missing=False):
        f.write(self.dumps(words, skip_missing=skip_missing))

    def as_dict(self):
        return {x: y for x,y in self.model.items()}