import json
import os
from collections import Counter
from glob import glob

import copy
from nltk.corpus import ptb

dropped = set()

def read_mrg(mrg_path):
    ptb_sents = ptb.tagged_sents(mrg_path)
    for tagged_sent in ptb_sents:
        sent = [t[0] for t in tagged_sent]
        assert not any([' ' in w for w in sent])
        for w in sent:
            if w[0] == '*':
                dropped.add(w)

    def process(t):
        def process_w(w, allowNone=True):
            if w[0] == '*':
                w = None
            w = {
                    '-LRB-': '(',
                    '-RRB-': ')',
                    '-LSB-': '[',
                    '-RSB-': ']',
                    '-LCB-': '{',
                    '-RCB-': '}',
                    # '``': '"',
                    # "''": '"',
                    # "--": '-',
                    '0': None
                }.get(w, w)
            assert w or allowNone
            return w
        return process_w(t[0]), process_w(t[1])

    processed_sents = [[process(t) for t in x if process(t)[0]] for x in ptb_sents]
    processed_sents = [
        {
            'sent': [t[0] for t in sent],
            'pos': [t[1] for t in sent]
        }
        for sent in processed_sents
    ]
    return processed_sents


def collect_sents(mrg_dir=os.path.dirname(__file__) + '/wsj', outf_path=os.path.dirname(__file__) + '/wsj/sents.json'):
    all_sents = []
    for mrg_path in sorted(glob(mrg_dir + '/*/*.mrg')):
        print(mrg_path)
        mrg_name = os.path.basename(os.path.realpath(mrg_path))
        sents = read_mrg(mrg_path)
        for ind, sent in enumerate(sents):
            sent['id'] = mrg_name + '_%06d' % ind
            all_sents.append(sent)
    with open(outf_path, 'w') as f:
        json.dump(all_sents, f, indent=2, sort_keys=True)
    return all_sents


def collect_pp_annotations(base_files=(
    os.path.dirname(__file__) + '/data/pp-data-english/wsj.2-21.txt.dep.pp',
    os.path.dirname(__file__) + '/data/pp-data-english/wsj.23.txt.dep.pp'
), outf_path=os.path.dirname(__file__) + '/data/pp-data-english/annotations.json'):
    annotations = []
    for base_file in base_files:
        fields = [
            'preps.words',
            'children.words',
            'heads.pos',
            'heads.next.pos',
            'heads.words',
            'labels',
            'nheads'
        ]
        fields_data = {}
        for field in fields:
            with open(base_file + '.' + field, 'r') as f:
                fields_data[field] = [l.strip().replace('\t', ' ').split(' ') for l in f.readlines()]
        cur_annotations = [{fld: data for fld, data in zip(fields, ann)} for ann in zip(*[fields_data[field] for field in fields])]
        for ind, ann in enumerate(cur_annotations):
            ann['id'] = base_file + '_%06d' % ind
            annotations.append(ann)
    with open(outf_path, 'w') as out_f:
        json.dump(annotations, out_f, indent=2)
    return annotations


def check_sent_match(ann, sent, match_ind):
    # check the token
    if ann['preps.words'][0] != sent['sent'][match_ind]:
        return False

    # check heads
    prevs = sent['sent'][max(match_ind - 10, 0): match_ind]
    prevs = [x.lower() for x in prevs]
    if not all(w in prevs for w in ann['heads.words']):
        return False

    return True

def match_sentence_annotations(sent, anns):
    anns.sort(key=lambda ann: ann['id'])
    pps = [ann['preps.words'][0] for ann in anns]
    sent_pps = [(pp, ind) for ind, pp in enumerate(sent['sent']) if pp in pps]

    matches = {}

    q_anns = list(anns)
    q_sent_pps = list(sent_pps)
    last_ann = None
    while q_anns and q_sent_pps:
        sent_pp, sent_ind = q_sent_pps[0]
        q_sent_pps = q_sent_pps[1:]
        if check_sent_match(q_anns[0], sent, sent_ind):
            ann = q_anns[0]
            last_ann = ann
            q_anns = q_anns[1:]
            matches[ann['id']] = sent_ind
        # elif last_ann and check_sent_match(last_ann, sent, sent_ind):
        #     del matches[last_ann['id']]
        #     last_ann = None

    return matches


def match_annotations(annotations, sents, cur_matches=None):
    match_counts = []
    matches = cur_matches or {}
    match_cands = {ann: [match] for ann, match in matches.items()}
    sents = copy.deepcopy(sents)
    for sent in sents:
        sent['word_next_pos'] = [(sent['sent'][ind], sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_next_pos_lc'] = [(sent['sent'][ind].lower(), sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['sent_lc'] = [x.lower() for x in sent['sent']]
    for ind, ann in enumerate(annotations):
        if ann['id'] in matches:
            continue
        fsents = []
        for sent_key, wnp_key in [('sent', 'word_next_pos'), ('sent_lc', 'word_next_pos_lc')]:
            for sent in sents:
                if ann['preps.words'][0] not in sent[sent_key]:
                    continue
                if ann['children.words'][0] not in sent[sent_key]:
                    continue
                words = list(sent[sent_key])
                found = True
                for h in ann['heads.words']:
                    try:
                        words = words[words.index(h) + 1:]
                    except ValueError:
                        found = False
                if not found:
                    continue
                wnps = list(sent[wnp_key])
                found = True
                for wnp in zip(ann['heads.words'], ann['heads.next.pos']):
                    try:
                        wnps = wnps[wnps.index(wnp) + 1:]
                    except ValueError:
                        found = False
                if not found:
                    continue
                if not all([wnp in sent[wnp_key] for wnp in zip(ann['heads.words'], ann['heads.next.pos'])]):
                    continue
                fsents.append(sent)
            if fsents:
                break

        fsents = list(fsents)
        if not fsents:
            print(ann)

        match_cands[ann['id']] = [x['id'] for x in fsents]
        match_counts.append(len(fsents))
        if len(fsents) == 1:
            matches[ann['id']] = fsents[0]['id']
        print("%d/%d" % (ind, len(annotations)))
    print(Counter(match_counts))

    sent_id_to_ind = {s['id']: ind for ind, s in enumerate(sents)}
    ann_id_to_ind = {ann['id']: ind for ind, ann in enumerate(annotations)}

    while True:
        n_matches = len(matches)
        for ind, ann in enumerate(annotations):
            cands = match_cands[ann['id']]
            if len(cands) <= 1:
                continue
            nbrs = [ann for ann in annotations[ind - 5: ind + 5] if len(match_cands[ann['id']]) == 1]
            nbrs_sent_inds = [sent_id_to_ind[match_cands[ann['id']][0]] for ann in nbrs]
            cand_inds = [sent_id_to_ind[cand_id] for cand_id in match_cands[ann['id']]]
            closest = sorted([(min([abs(c - cand_ind) for c in nbrs_sent_inds] + [1000000]), cand_ind) for cand_ind in cand_inds])
            if closest[0][0] < 10 and closest[1][0] >= 100:
                matches[ann['id']] = sents[closest[0][1]]['id']
        if len(matches) == n_matches:
            break

    print('%d/%d' % (len(matches), len(annotations)))

    matches_inds = {}

    matches_per_sent = {}
    for ann_id, sent_id in matches.items():
        matches_per_sent[sent_id] = matches_per_sent.get(sent_id) or []
        matches_per_sent[sent_id].append(ann_id)

    mismatches = 0
    for sent_id, match_ids in matches_per_sent.items():
        sent = sents[sent_id_to_ind[sent_id]]
        anns = [annotations[ann_id_to_ind[ann_id]] for ann_id in match_ids]
        sent_matches = match_sentence_annotations(sent, anns)
        if len(sent_matches) != len(match_ids):
            mismatches += 1
        else:
            for ann_id, match_ind in sent_matches.items():
                matches_inds[ann_id] = (sent_id, match_ind)

    print("mismatches:", mismatches)

    return matches

if __name__ == '__main__':
    # sents = collect_sents()
    # annotations = collect_pp_annotations()
    try:
        with open(os.path.dirname(__file__) + '/data/pp-data-english/annotations.json', 'r') as f:
            annotations = json.load(f)
    except:
        annotations = collect_pp_annotations()
    try:
        with open(os.path.dirname(__file__) + '/wsj/sents.json', 'r') as f:
            sents = json.load(f)
    except:
        sents = collect_sents()
    try:
        with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_matches.json', 'r') as f:
            matches = json.load(f)
    except:
        matches = {}
    matches = match_annotations(annotations, sents, matches)
    with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_matches.json', 'w') as f:
        json.dump(matches, f)

    # print(dropped)




