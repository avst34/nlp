from collections import namedtuple

import pickle
import json


class MostFrequentClassModel:

    Sample = namedtuple('Sample', ['xs', 'ys', 'mask'])

    def __init__(self, features, include_empty, n_labels_to_predict):
        self.n_labels_to_predict = n_labels_to_predict
        self.include_empty = include_empty
        self.masker = self.build_masker()

        self.features = features
        self._cond_counts = {}
        self._most_likely_y = {}

    def build_masker(self):
            return lambda x: x["identified_for_pss"] == True

    def get_sample_mask(self, xs):
        return [self.masker(x) for x in xs]

    def mask_sample(self, sample):
        return MostFrequentClassModel.Sample(sample.xs, sample.ys, mask=self.get_sample_mask(sample.xs))

    def tuplize_x(self, x):
        return tuple([x[k] for k in sorted(x.keys()) if k in self.features])

    def fit(self, samples, validation_samples=None, validation_split=0.2, evaluator=None, show_progress=False):
        samples = [self.mask_sample(sample) for sample in samples]
        if not validation_samples:
            test = samples[:int(len(samples)*validation_split)]
            train = samples[int(len(samples)*validation_split):]
        else:
            validation_samples = [self.mask_sample(sample) for sample in validation_samples]
            test = validation_samples
            train = samples

        self._cond_counts = {}
        for sample in train:
            mask = self.get_sample_mask(sample.xs)
            for ind, (x, y) in enumerate(zip(sample.xs, sample.ys)):
                if mask[ind] and (self.include_empty or any(y)):
                    xtup = self.tuplize_x(x)
                    self._cond_counts[xtup] = self._cond_counts.get(xtup, {})
                    self._cond_counts[xtup][y] = self._cond_counts[xtup].get(y, 0)
                    self._cond_counts[xtup][y] += 1
                    self._cond_counts['ALL'] = self._cond_counts.get('ALL', {})
                    self._cond_counts['ALL'][y] = self._cond_counts['ALL'].get(y, 0)
                    self._cond_counts['ALL'][y] += 1

        self._most_likely_y = {}
        for xtup in self._cond_counts:
            y_max = max(self._cond_counts[xtup].items(), key=lambda item: item[1])[0]
            print('Most likely for %s: %s' % (str(xtup), str(y_max)))
            self._most_likely_y[xtup] = y_max

        print('--------------------------------------------')
        print('Training is complete (%d samples)' % (len(train)))
        if evaluator:
            print('Test data evaluation:')
            evaluator.evaluate(test, examples_to_show=3, predictor=self)
            print('Training data evaluation:')
            evaluator.evaluate(train, examples_to_show=3, predictor=self)
        print('--------------------------------------------')

        return self

    def get_most_likely_y(self, sample_x):
        mly = self._most_likely_y.get(self.tuplize_x(sample_x)) or tuple([None] * self.n_labels_to_predict)
        if not all(mly) and not self.include_empty:
            mly = self._most_likely_y.get('ALL')
        assert self.include_empty or all(mly)
        return mly

    def predict(self, sample_xs, mask=None):
        if mask is None:
            mask = self.get_sample_mask(sample_xs)
        r = [self.get_most_likely_y(x) if mask[ind] else tuple([None] * self.n_labels_to_predict) for ind, x in enumerate(sample_xs)]
        return r

    def save(self, base_path):
        with open(base_path + '.pickle', 'wb') as out_f:
            pickle.dump({
                'n_labels_to_predict': self.n_labels_to_predict,
                'include_empty': self.include_empty,
                'features': self.features,
                '_cond_counts': self._cond_counts,
                '_most_likely_y':  self._most_likely_y
            }, out_f)