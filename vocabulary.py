class Vocabulary(object):

    def __init__(self, name="Unnamed"):
        self._word_to_ind = {}
        self._ind_to_word = []
        self.name = name

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if self.get_index_if_exists(word) is None:
            self._ind_to_word.append(word)
            self._word_to_ind[word] = len(self._ind_to_word) - 1

    def get_index_if_exists(self, word):
        return self._word_to_ind.get(word)

    def get_index(self, word):
        ind = self.get_index_if_exists(word)
        if ind is None:
            raise Exception('Word "%s" does not appear in the vocabulary' % word)
        return ind

    def get_word(self, ind):
        n = self.size()
        if ind >= n:
            raise Exception('Index %d is out of bounds, vocabulary size is %d' % (ind, n))
        return self._ind_to_word[ind]

    def size(self):
        return len(self._ind_to_word)

    def all_words(self):
        return self._word_to_ind.keys()

    def __repr__(self):
        return 'Vocabulary: "%s" with %d words' % (self.name, self.size())

class VocabularyBuilder(object):

    def __init__(self, record_words_extractor_f):
        self._record_words_extractor_f = record_words_extractor_f

    def feed_records_to_vocab(self, records, vocabulary):
        for record in records:
            vocabulary.add_words(self._record_words_extractor_f(record))
