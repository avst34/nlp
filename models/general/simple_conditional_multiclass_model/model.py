from collections import namedtuple

class MostFrequentClassModel:

    Sample = namedtuple('Sample', ['xs', 'ys', 'mask'])

    def __init__(self, features, mask_by, include_empty, n_labels_to_predict):
        self.n_labels_to_predict = n_labels_to_predict
        self.mask_by = mask_by
        self.include_empty = include_empty
        self.masker = self.build_masker()

        self.features = features
        self._cond_counts = {}
        self._most_likely_y = {}

    def build_masker(self):
        if self.mask_by is None:
            return lambda x: True
        field = self.mask_by.split(':')[0].strip()
        values = [x.strip() for x in self.mask_by.split(':')[1].split(',')]
        return lambda x: x[field] in values

    def mask_xs(self, xs):
        return [self.masker(x) for x in xs]

    def mask_sample(self, sample):
        return MostFrequentClassModel.Sample(sample.xs, sample.ys, mask=self.mask_xs(sample.xs))

    def tuplize_x(self, x):
        return tuple([x[k] for k in sorted(x.keys()) if k in self.features])

    def fit(self, samples, validation_split=0.2, validation_samples=None, evaluator=None):
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
            mask = self.mask_xs(sample.xs)
            for ind, (x, y) in enumerate(zip(sample.xs, sample.ys)):
                if mask[ind] and (self.include_empty or any(y)):
                    xtup = self.tuplize_x(x)
                    self._cond_counts[xtup] = self._cond_counts.get(xtup, {})
                    self._cond_counts[xtup][y] = self._cond_counts[xtup].get(y, 0)
                    self._cond_counts[xtup][y] += 1

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
        return self._most_likely_y.get(self.tuplize_x(sample_x)) or tuple([None] * self.n_labels_to_predict)

    def predict(self, sample_xs, mask=None):
        if mask is None:
            mask = self.mask_xs(sample_xs)
        return [self.get_most_likely_y(x) if mask[ind] else tuple([None] * self.n_labels_to_predict) for ind, x in enumerate(sample_xs)]