from datasets.streusle import streusle
from word2vec import Word2VecModel
import json

loader = streusle.StreusleLoader()
records = loader.load()

def enhance_word2vec():
    # collect word2vec vectors for words in the data
    all_tokens = set()
    for rec in records:
        for tagged_token in rec.tagged_tokens:
            all_tokens.add(tagged_token.token)

    wvm = Word2VecModel.load_google_model()
    missing_words = wvm.collect_missing(all_tokens)
    with open(streusle.ENHANCEMENTS.WORD2VEC_PATH, 'wb') as f:
        wvm.dump(all_tokens, f, skip_missing=True)

    with open(streusle.ENHANCEMENTS.WORD2VEC_MISSING_PATH, 'w') as f:
        json.dump(missing_words, f, indent=2)

    print('Enhanced with word2vec, %d words in total (%d skipped)' % (len(all_tokens), len(missing_words)))

if __name__ == '__main__':
    enhance_word2vec()


