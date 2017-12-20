class CombinedPredictor:

    def __init__(self, predictors):
        self.predictors = predictors

    def predict(self, sample_xs, mask=None):
        ys = [[] for _ in sample_xs]
        for predictor in self.predictors:
            p_ys = predictor.predict(sample_xs, mask=mask)
            for ind, p_t in enumerate(p_ys):
                ys[ind] = tuple(list(ys[ind]) + list(p_t))

        return ys
