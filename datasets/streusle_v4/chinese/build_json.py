import json
import os

from datasets.streusle_v4.chinese.corenlp import run_corenlp
from datasets.streusle_v4.release.govobj import findgovobj
from supersense_repo import SUPERSENSES_SET, get_supersense_type
from supersense_repo.constants import TYPES


def fix_ss(full_ss):
    full_ss = full_ss.strip()
    if not full_ss or full_ss in ["`d"]:
        return None
    ss1,ss2 = ((full_ss + '~')*2).split('~')[:2]
    sss = {
        'ss1': ss1,
        'ss2': ss2
    }

    for k, ss in sss.items():
        found = False
        for pss in SUPERSENSES_SET:
            if ss.replace('/','').lower() == pss.replace('/','').lower():
                sss[k] = pss
                found = True
        if not found:
            raise Exception("Unknown pss:" + ss)
        ss = sss[k]
        if get_supersense_type(ss) != TYPES.PREPOSITION_SUPERSENSE:
            return None

    return 'p.'+sss['ss1'] + '~' + 'p.'+sss['ss2']

def build_chinese_streusle_json(txt_path=os.path.dirname(__file__) + '/lpp_zho.txt'):
    with open(txt_path, 'r') as f:
        text = f.read().replace('\r\n', '\n')

    chapters = [[x.strip() for x in chapter.strip().splitlines()][1:] for chapter in text.split('\n\n')]
    lines = sum(chapters, [])

    sents = []
    for ind, line in enumerate(lines):
        tokens_with_pss = line.split(' ')
        tokens = [t if ":" not in t else t[:t.index(':')] for t in tokens_with_pss]
        corenlp_out_lines = run_corenlp(tokens, format='conll').strip().splitlines()
        corenlp_out_tuples = [tuple(l.split('\t')) for l in corenlp_out_lines]
        assert all([x.strip() for x in corenlp_out_lines]), 'corenlp split a sentence'

        s_toks = [
            {
                "#": ctok[0],
                "word": ctok[1],
                "lemma": ctok[2],
                "upos": None,
                "xpos": ctok[3],
                "feats": None,
                "head": int(ctok[5]),
                "deprel": ctok[6],
                "edeps": None,
                "misc": None,
                "smwe": None,
                "wmwe": None,
                "lextag": None,
                "ner": ctok[4],
                "full_ss": fix_ss(otok[otok.index(':') + 1:]) if ":" in otok else None
            }
            for ctok,otok in zip(corenlp_out_tuples, tokens_with_pss)
        ]

        sent = {
            "sent_id": "chinese-lp-%05d" % ind,
            "text": ' '.join(tokens),
            "streusle_sent_id": "chinese-lp-%05d" % ind,
            "toks": s_toks,
            "swes": {
                stok["#"]: {
                    "lexlemma": stok["lemma"],
                    "lexcat": "P",
                    "ss": stok["full_ss"].split("~")[0],
                    "ss2": stok["full_ss"].split("~")[1],
                    "toknums": [
                        int(stok["#"])
                    ]
                }
            for stok in s_toks if stok["full_ss"]},
            "smwes": {},
            "wmwes": {}
        }

        for swe in sent['swes'].values():
            findgovobj(swe, sent)

        sents.append(sent)
    return sents

json_path = os.path.dirname(__file__) + '/lp.chinese.all.json'
# d = json.load(open(json_path, 'r'), indent=2)
cj = build_chinese_streusle_json()
json.dump(cj, open(json_path, 'w'), indent=2)

# records = StreusleLoader().load(conllulex_path=json_path, input_format='json')
# records = StreusleLoader().load()
# print(len(records))
#
# reader = EmbeddingsHDF5Reader('/cs/usr/aviramstern/lab/muse/embeddings/vectors-en-streusle.hd5')
# found = 0
# tot = 0
# with_pss = 0
# for rec in records:
#     for tok in rec.tagged_tokens:
#         if reader.get(tok.token) is not None or reader.get(tok.token.lower()) is not None:
#             found += 1
#         if tok.supersense_role:
#             with_pss += 1
#         tot += 1
# print(found, tot, with_pss, found/tot)