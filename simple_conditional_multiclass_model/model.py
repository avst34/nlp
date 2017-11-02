from collections import namedtuple

Sample = namedtuple('Sample', ['xs', 'ys'])

class SimpleConditionalMulticlassModel:

    def __init__(self):
        self._cond_counts = {}
        self._most_likely_y = {}

    def tuplize_x(self, x):
        return tuple([x[k] for k in sorted(x.keys())])

    def fit(self, samples, validation_split=0.2, evaluator=None):
        test = samples[:int(len(samples)*validation_split)]
        train = samples[int(len(samples)*validation_split):]

        self._cond_counts = {}
        for sample in train:
            for x, y in zip(sample.xs, sample.ys):
                if y is not None:
                    xtup = self.tuplize_x(x)
                    self._cond_counts[xtup] = self._cond_counts.get(xtup, {})
                    self._cond_counts[xtup][y] = self._cond_counts[xtup].get(y, 0)
                    self._cond_counts[xtup][y] += 1

        self._most_likely_y = {}
        for xtup in self._cond_counts:
            y_max = max(self._cond_counts[xtup].items(), key=lambda item: item[1])[0]
            self._most_likely_y[xtup] = y_max

        print('--------------------------------------------')
        print('Training is complete (%d samples)' % (len(train)))
        if evaluator:
            print('Test data evaluation:')
            evaluator.evaluate(test, examples_to_show=0, predictor=self)
            print('Training data evaluation:')
            evaluator.evaluate(train, examples_to_show=0, predictor=self)
        print('--------------------------------------------')

    def get_most_likely_y(self, sample_x):
        return self._most_likely_y.get(self.tuplize_x(sample_x))

    def predict(self, sample_xs):
        return [self.get_most_likely_y(x) for x in sample_xs]