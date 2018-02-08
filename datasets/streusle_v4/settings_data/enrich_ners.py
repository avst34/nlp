import copy

from utils import parse_conll


def enrich_ners(sents_json, conlls):
    enriched_sents = []
    conlls = [parse_conll('\n'.join(conll))[0] for conll in conlls]
    for sent, conll_sent in zip(sents_json, conlls):
        sent = copy.deepcopy(sent)
        for jtok, ctok in zip(sent['toks'], conll_sent['tokens']):
            assert str(jtok['#']) == str(ctok['id'])
            jtok['ner'] = ctok['ner']
        enriched_sents.append(sent)
    return enriched_sents


