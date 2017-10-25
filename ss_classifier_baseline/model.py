import supersenses


class SupersensesClassifierBaselineModel(object):

    def __init__(self,
                 ss_types_to_predict=None,
                 use_pos_tags=True):
        if ss_types_to_predict is None:
            ss_types_to_predict = list(supersenses.constants.TYPES)
        self._ss_types_to_predict = ss_types_to_predict
        self._use_pos_tags = use_pos_tags

