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
        mrg_name = os.path.basename(mrg_path)
        sents = read_mrg(mrg_path)
        for ind, sent in enumerate(sents):
            sent['id'] = mrg_name + '_' + str(ind)
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
            ann['id'] = base_file + '_' + str(ind)
            annotations.append(ann)
    with open(outf_path, 'w') as out_f:
        json.dump(annotations, out_f, indent=2)
    return annotations


def match_annotations(annotations, sents, cur_matches=None):
    match_counts = []
    matches = cur_matches or {}
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

        match_counts.append(len(fsents))
        if len(fsents) == 1:
            matches[ann['id']] = fsents[0]['id']
        print("%d/%d" % (ind, len(annotations)))
    print(Counter(match_counts))
    return matches

if __name__ == '__main__':
    # sents = collect_sents()
    # annotations = collect_pp_annotations()
    with open(os.path.dirname(__file__) + '/data/pp-data-english/annotations.json', 'r') as f:
        annotations = json.load(f)
    with open(os.path.dirname(__file__) + '/wsj/sents.json', 'r') as f:
        sents = json.load(f)
    # with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_matches.json', 'r') as f:
    #     matches = json.load(f)
    matches = match_annotations(annotations, sents)
    with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_matches.json', 'w') as f:
        json.dump(matches, f)

    # print(dropped)




