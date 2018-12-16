import json
import os

from datasets.streusle_v4.chinese.attach_eng import align_lpp_amr, attach_eng
from datasets.streusle_v4.chinese.corenlp import run_corenlp
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


def load_alignment():
    with open(os.path.dirname(__file__) + '/alignment.txt') as f:
        lines = [l.strip() for l in f.readlines()]
    alignments = []
    for l in lines:
        alignments.append({int(t.split('-')[0]):int(t.split('-')[1]) for t in l.split()})
    return alignments


def build_chinese_streusle_json(txt_path=os.path.dirname(__file__) + '/lpp_zho.txt'):
    # with open(txt_path, 'r') as f:
    #     text = f.read().replace('\r\n', '\n')
    #
    # chapters = [[x.strip() for x in chapter.strip().splitlines()][1:] for chapter in text.split('\n\n')]
    # lines = sum(chapters, [])

    lines = align_lpp_amr()

    sents = []
    for ind, line in enumerate(lines):
        tokens_with_pss = line
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
        #
        # for swe in sent['swes'].values():
        #     findgovobj(swe, sent)

        sents.append(sent)
    return sents

def build_translated_chinese_streusle_json(txt_path=os.path.dirname(__file__) + '/lpp_zho.txt'):
    # with open(txt_path, 'r') as f:
    #     text = f.read().replace('\r\n', '\n')
    #
    # chapters = [[x.strip() for x in chapter.strip().splitlines()][1:] for chapter in text.split('\n\n')]
    # lines = sum(chapters, [])

    zh_sents = align_lpp_amr()
    eng_sents = attach_eng(zh_sents)

    with open(os.path.dirname(__file__) + '/eng.txt', 'w') as f:
        f.write('\n'.join([' '.join(e) for e in eng_sents]))
    with open(os.path.dirname(__file__) + '/zh.txt', 'w') as f:
        f.write('\n'.join([' '.join(z) for z in zh_sents]))

    alignments = load_alignment()
    rev_alignments = [{v: k for k,v in a.items()} for a in alignments]
    assert len(alignments) == len(zh_sents) and len(alignments) == len(eng_sents)
    sents = []
    for ind, (zh_sent, eng_sent, alignment, rev_alignment) in enumerate(zip(zh_sents, eng_sents, alignments, rev_alignments)):
        zh_tokens_with_pss = zh_sent
        zh_tokens = [t if ":" not in t else t[:t.index(':')] for t in zh_tokens_with_pss]
        en_tokens = eng_sent

        corenlp_out_lines = run_corenlp(en_tokens, format='conll', port=9000).strip().splitlines()
        corenlp_out_tuples = [tuple(l.split('\t')) for l in corenlp_out_lines]

        corenlp_out_lines_zh = run_corenlp(zh_tokens, format='conll', port=9001).strip().splitlines()
        corenlp_out_tuples_zh = [tuple(l.split('\t')) for l in corenlp_out_lines_zh]

        zh_dep = {t[0]: t[6] for t in corenlp_out_tuples_zh}
        zh_parent = {t[0]: t[5] for t in corenlp_out_tuples_zh}
        zh_grandparent = {t[0]: zh_parent.get(t[5]) or t[0] for t in corenlp_out_tuples_zh}

        assert all([x.strip() for x in corenlp_out_lines]), 'corenlp split a sentence'

        gp_override = {}

        for t_ind, t in enumerate(zh_tokens_with_pss):
            if ":" in t:
                if t_ind not in alignment:
                    print("Missing prep! ", t)
                    id = str(len(corenlp_out_tuples) + 1)
                    corenlp_out_tuples.append((
                        str(len(corenlp_out_tuples) + 1),
                        'MISSING_PREP_' + str(t_ind) + t[t.index(':'):],
                        'MISSING_PREP_' + str(t_ind),
                        'P',
                        None,
                        zh_parent[str(t_ind + 1)],
                        zh_dep[str(t_ind + 1)],
                        None,
                    ))
                    gp_override[id] = zh_grandparent[str(t_ind + 1)]
                else:
                    print("Found prep!", corenlp_out_tuples[alignment[t_ind]][1])


        s_toks = [
            {
                "#": ctok[0],
                "word": ctok[1],
                "lemma": ctok[2],
                "upos": None,
                "xpos": ctok[3],
                "feats": None,
                "head": int(ctok[5]),
                "grandparent_override": gp_override.get(ctok[0]),
                "deprel": ctok[6],
                "edeps": None,
                "misc": None,
                "smwe": None,
                "wmwe": None,
                "lextag": None,
                "ner": ctok[4],
                "full_ss": fix_ss(otok[otok.index(':') + 1:]) if ":" in otok else fix_ss(ctok[1][ctok[1].index(':') + 1:]) if ":" in ctok[1] else None,
                "hidden": ctok[1].startswith("MISSING_PREP_")
            }
            for ctok in corenlp_out_tuples for otok in ([zh_tokens_with_pss[rev_alignment[int(ctok[0]) - 1]]] if (int(ctok[0]) - 1) in rev_alignment else [""])
        ]

        sent = {
            "sent_id": "chinese-lp-%05d" % ind,
            "zh_text": ' '.join(zh_sent),
            "en_text": ' '.join(eng_sent),
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
        #
        # for swe in sent['swes'].values():
        #     findgovobj(swe, sent)

        sents.append(sent)
    return sents

json_path = os.path.dirname(__file__) + '/lp.eng.zh_pss.all.json'
# d = json.load(open(json_path, 'r'))
cj = build_translated_chinese_streusle_json()
json.dump(cj, open(json_path, 'w'), indent=2)

# records = StreusleLoader().load(conllulex_path=json_path, input_format='json')
# toks = [t.token for r in records for t in r.tagged_tokens]
# embds = [MUSE_STREUSLE_DICT.get(t) for t in toks]
# missing = len([e for e in embds if e is None])
# missing_trans = [w for w in toks if zh_en.get(w) is None]
# for w in missing_trans[:5]:
#     print(w)
# existing = len(embds) - missing
# print("Missing embeddings: %d, existing embeddings: %d, missing percentage: %d, missing trans: %d, missing percentage: %d" % (
#             missing,
#             existing,
#             missing/(missing + existing)*100,
#             len(missing_trans),
#             len(missing_trans)/(missing + existing)*100
#         )
#       )

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