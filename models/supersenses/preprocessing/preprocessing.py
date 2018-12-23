import json
import os
from tempfile import NamedTemporaryFile

from datasets.streusle_v4 import StreusleRecord
from datasets.streusle_v4.release.govobj import findgovobj
from datasets.streusle_v4.settings_data.enrich_autoid import enrich_autoid
from models.supersenses.preprocessing.corenlp import run_corenlp
from models.supersenses.streusle_integration import streusle_record_to_lstm_model_sample


# w2v = Word2VecModel.load_google_model()

def preprocess_sentence(sentence, identify=True):
    corenlp_out = json.loads(run_corenlp(sentence.split(), format='json'))
    csent = corenlp_out['sentences'][0]
    parsed_tokens = csent['tokens']
    assert len(corenlp_out['sentences']) == 1
    assert all(tok['index'] == ind + 1 for ind, tok in enumerate(parsed_tokens))
    # s = {
    #     'word': [tok['word'] for tok in parsed_tokens],
    #     'xpos': [tok['pos'] for tok in parsed_tokens],
    #     'ner': [tok['ner'] for tok in parsed_tokens],
    #     'lemma': [tok['lemma'] for tok in parsed_tokens],
    #     # 'lemma_w2v': [w2v.get(tok['lemma']) if False and not vocabs.LEMMAS.has_word(tok['lemma']) else None for tok in parsed_tokens],
    #     # 'token_w2v': [w2v.get(tok) if False and not vocabs.TOKENS.has_word(tok) else None for tok in tokens],
    #     'ud_dep': [dep['dep'] for dep in parsed_deps],
    #     'ud_head_ind': [dep['governor'] - 1 if dep['governor'] else None for dep in parsed_deps],
    # }
    deprec = lambda t: [x for x in csent['basicDependencies'] if x['dependent'] == t['index']][0]
    s_toks = [
        {
            "#": str(ctok['index']),
            "word": ctok['word'],
            "lemma": ctok['lemma'],
            "upos": None,
            "xpos": ctok['pos'],
            "feats": None,
            "head": int(deprec(ctok)['governor']),
            "grandparent_override": None,
            "deprel": deprec(ctok)['dep'],
            "edeps": None,
            "misc": None,
            "smwe": None,
            "wmwe": None,
            "lextag": None,
            "ner": ctok['ner'],
            "hidden": False,
        }
        for ctok in corenlp_out['sentences'][0]['tokens']
    ]

    sent = {
        "sent_id": "sentid",
        "text": sentence,
        "streusle_sent_id": "sentid",
        "toks": s_toks,
    }

    if identify:
        conllu = run_corenlp(sentence.split(), format='conllu')
        conllu = '\n'.join([l for l in conllu.split('\n') if '.' not in l.split('\t')[0]])
        tempf = NamedTemporaryFile('w', delete=False)
        try:
            tempf.write(conllu.strip())
            tempf.close()
            ids = enrich_autoid(tempf.name, tempf.name)
            for t in ids:
                for id in ids[t]:
                    findgovobj(ids[t][id], sent)
            with open(tempf.name) as f:
                enriched_conllu = f.read()
                print(enriched_conllu)
            sent.update(ids)

        finally:
            try:
                os.unlink(tempf.name)
            finally:
                print("WARNING: unable to delete tempfile:", tempf.name)

    return streusle_record_to_lstm_model_sample(StreusleRecord(sent['sent_id'], sent["text"], sent))


if __name__ == '__main__':
    s = preprocess_sentence("The school was right in front of my house".split(), identify=True)
    print(s)