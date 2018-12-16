import os

json_path = os.path.dirname(__file__) + '/lp.chinese.all.json'
amr_path = os.path.dirname(__file__) + '/amr.txt'
lpp_path =  os.path.dirname(__file__) + '/lpp_zho.txt'

def load_amr_sents():
    with open(amr_path, 'r') as f:
        amr_lines = [l.strip() for l in f.readlines()]
    sents = []
    eng, zh = None, None
    for line in amr_lines:
        if "::snt" in line:
            eng = line.replace('# ::snt ', '').strip()
        if "::zh" in line:
            zh = line.replace('# ::zh ', '').strip()
            assert eng and zh
            sents.append({
                "eng": eng,
                "zh": zh
            })

    return sents

def load_lpp_sents():
    with open(lpp_path) as f:
        text = f.read()
    chapters = [[x.strip() for x in chapter.strip().splitlines()][1:] for chapter in text.split('\n\n')]
    lines = sum(chapters, [])

    return [{
        "orig": l.split(),
        "clean": [t.replace('NP_[', '').replace(']', '') if ":" not in t else t[:t.index(':')] for t in l.split() if t]
    } for l in lines]


def align_lpp_amr():
    amr_sents = load_amr_sents()
    lpp_sents = load_lpp_sents()

    lpp_toks_clean = [t for s in lpp_sents for t in s['clean']]
    lpp_toks_orig = [t for s in lpp_sents for t in s['orig']]

    q_amr, q_lpp = list(amr_sents), list(lpp_sents)

    aligned_sents = []
    skipped = []
    for amr_sent in amr_sents:
        sent = []
        amr_zh = amr_sent["zh"].replace(' ', '')
        amr_zh_orig = amr_zh
        while amr_zh and len(lpp_toks_orig):
            ltok_orig, ltok_clean = lpp_toks_orig.pop(0), lpp_toks_clean.pop(0)
            if not amr_zh.startswith(ltok_clean):
                print("Skipping:", amr_zh_orig)
                lpp_toks_orig.insert(0, ltok_orig)
                lpp_toks_clean.insert(0, ltok_clean)
                skipped.append(amr_zh_orig)
                sent = None
                break
            else:
                sent.append(ltok_orig)
                amr_zh = amr_zh[len(ltok_clean):]
        if len(lpp_toks_orig) == 0:
            break
        if sent:
            aligned_sents.append(sent)

    return aligned_sents






def attach_eng(all_st_sents):
    amr_sents = load_amr_sents()

    print("amr: %d, st: %d" % (len(amr_sents), len(all_st_sents)))

    all_amr_sents = list(amr_sents)
    st_sents = list(all_st_sents)
    eng_sents = []
    while amr_sents and st_sents:
        amr_sent = amr_sents.pop(0)
        if ''.join([t.replace('NP_[','').replace(']', '') if ":" not in t else t[:t.index(':')] for t in st_sents[0]]) == amr_sent["zh"].replace(' ', ''):
            st_sents.pop(0)
            eng_sents.append(amr_sent["eng"].split())
        else:
            continue

    if st_sents:
        raise Exception("Unable to match %d out of %d sents" % (len(st_sents), len(all_st_sents)))

    return eng_sents

if __name__ == '__main__':
    attach_eng()