from datasets.streusle_v4.release.govobj import findgovobj

from models.supersenses import vocabs
from models.supersenses.preprocessing.corenlp import run_corenlp
import json

from word2vec import Word2VecModel



w2v = Word2VecModel.load_google_model()

def preprocess_sentence(tokens):
    corenlp_out = json.loads(run_corenlp(tokens, format='json'))
    sent = corenlp_out['sentences'][0]
    parsed_tokens = sent['tokens']
    parsed_deps = sorted(sent['basicDependencies'], key=lambda dep: dep['dependent'])
    assert len(corenlp_out['sentences']) == 1
    assert len(parsed_tokens) == len(tokens)
    assert all(tok['index'] == ind + 1 for ind, tok in enumerate(parsed_tokens))
    assert all(dep['dependentGloss'] == tok for dep, tok in zip(parsed_deps, tokens))
    s = {
        'ud_xpos': [tok['pos'] for tok in parsed_tokens],
        'ner': [tok['ner'] for tok in parsed_tokens],
        'lemma': [tok['lemma'] for tok in parsed_tokens],
        'lemma_w2v': [w2v.get(tok['lemma']) if False and not vocabs.LEMMAS.has_word(tok['lemma']) else None for tok in parsed_tokens],
        'token_w2v': [w2v.get(tok) if False and not vocabs.TOKENS.has_word(tok) else None for tok in tokens],
        'ud_dep': [dep['dep'] for dep in parsed_deps],
        'ud_head_ind': [dep['governor'] - 1 if dep['governor'] else None for dep in parsed_deps],
        'govobj': [findgovobj({
            'lexlemma': tok['lemma'],
            'toknums': [tok['index']]
        }, {
            'toks': [{
                'head': d['governor'],
                'xpos': t['pos'],
                'deprel': d['dep'],
                '#': t['index'],
                'lemma': t['lemma']
            } for t, d in zip(parsed_tokens, parsed_deps)]
        }) for tok in parsed_tokens]
    }

    return s
