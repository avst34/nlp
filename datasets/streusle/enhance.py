from collections import namedtuple

import spacy
from spacy.tokens import Doc

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

def apply_spacy_pipeline(tokens):
    nlp = spacy.load('en_core_web_sm')
    doc = Doc(nlp.vocab, words=tokens)
    for name, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc

TreeNode = namedtuple('TreeNode', ['head_ind', 'dep'])
def enhance_dependency_trees():
    trees = {}
    for ind, rec in enumerate(records):
        doc = apply_spacy_pipeline([tt.token for tt in rec.tagged_tokens])
        trees[rec.id] = [
            TreeNode(head_ind=token.head.i, dep=token.dep_)
            for token in doc
        ]
        print('enhance_dependency_trees: %d/%d' % (ind + 1, len(records)))
    with open(streusle.ENHANCEMENTS.SPACY_DEP_TREES, 'w') as f:
        json.dump(trees, f, indent=2)
    print('Enhanced with spacy dep trees, %d trees in total' % (len(trees)))

if __name__ == '__main__':
    # enhance_dependency_trees()
    enhance_word2vec()


