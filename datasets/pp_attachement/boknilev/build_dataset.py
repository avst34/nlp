import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import copy

import sys
from nltk.corpus import ptb

# from models.supersenses.preprocessing import preprocess_sentence
from utils import parse_conllx, parse_conllx_file

dropped = set()

def read_mrg(mrg_path):
    conllx_sents = parse_conllx_file(mrg_path + '.conllx')
    for conllx_sent in conllx_sents:
        sent = [x['token'] for x in conllx_sent['tokens']]
        assert not any([' ' in w for w in sent])

    def process(t):
        return t['token'], t['pos'], t['head']

    processed_sents = [[process(t) for t in x['tokens'] if process(t)[0]] for x in conllx_sents]
    processed_sents = [
        {
            'sent': [t[0] for t in sent],
            'pos': [t[1] for t in sent],
            'head': [t[2] for t in sent]
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
            ann['id'] = os.path.basename(base_file) + '_%06d' % ind
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
    match_cands = {ann: [tuple(match)] for ann, match in matches.items()}
    sents = copy.deepcopy(sents)
    for sent in sents:
        sent['word_next_pos'] = [(sent['sent'][ind], sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_next_pos_lc'] = [(sent['sent'][ind].lower(), sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_head'] = [(sent['sent'][ind], sent['head'][ind]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_head_lc'] = [(sent['sent'][ind].lower(), sent['head'][ind]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['sent_lc'] = [x.lower() for x in sent['sent']]
    for ann_ind, ann in enumerate(annotations[:200]):
        if ann['id'] in match_cands:
            continue
        match_cands[ann['id']] = []
        for sent_key, wnp_key, wh_key in [('sent', 'word_next_pos', 'word_head'), ('sent_lc', 'word_next_pos_lc', 'word_head_lc')]:
            for sent in sents:
                # if 'wsj.23' in ann['id'] and 'wsj.23' not in sent['id']:
                #     continue
                # if 'wsj.2-21' in ann['id'] and 'wsj.2-21' not in sent['id']:
                #     continue
                if ann['children.words'][0] not in sent[sent_key]:
                    continue

                for ind, (word, head) in enumerate(sent[wh_key]):
                    if word == ann['preps.words'][0] and \
                       0 <= (head - 1) and \
                       sent[sent_key][head - 1] == ann['heads.words'][int(ann['labels'][0]) - 1]:
                            window = sent[sent_key][max(ind - 10, 0):ind]
                            found = True
                            for h in ann['heads.words']:
                                try:
                                    window = window[window.index(h) + 1:]
                                except ValueError:
                                    found = False
                            if not found:
                                continue
                            window = sent[wnp_key][max(ind - 10, 0):ind]
                            found = True
                            for wnp in zip(ann['heads.words'], ann['heads.next.pos']):
                                try:
                                    window = window[window.index(wnp) + 1:]
                                except ValueError:
                                    found = False
                            if not found:
                                continue
                            match_cands[ann['id']].append((sent['id'], ind))
            if match_cands[ann['id']]:
                break

        if not match_cands[ann['id']]:
            print(ann)

        match_counts.append(len(match_cands[ann['id']]))
        print("%d/%d" % (ann_ind, len(annotations)))

    print(Counter(match_counts))
    sent_id_to_ind = {s['id']: ind for ind, s in enumerate(sents)}
    ann_id_to_ind = {ann['id']: ind for ind, ann in enumerate(annotations)}

    # missing = [ann_id for ann_id in match_cands if ann_id not in matches]
    # print(sorted(missing))
    print('train:', len([ann_id for ann_id in match_cands if 'wsj.2-21.txt.dep.' in ann_id]))
    print('test:', len([ann_id for ann_id in match_cands if 'wsj.23.txt.dep.' in ann_id]))

    # for ann_id in sorted(match_cands):
    #     print(ann_id, matches.get(ann_id))

    matches_per_sent_pp = {}
    for ann_id, ann_matches in match_cands.items():
        for match in ann_matches:
            print(match)
            matches_per_sent_pp[match] = matches_per_sent_pp.get(match) or []
            matches_per_sent_pp[match].append(ann_id)

    closed_matches = {ann_ids[0]: match for match, ann_ids in matches_per_sent_pp.items() if len(ann_ids) == 1}

    for ann_id, (sent_id, tok_ind) in closed_matches.items():
        ann = annotations[ann_id_to_ind[ann_id]]
        ann['sent_id'] = sent_id
        ann['tok_ind'] = tok_ind

    print("Matched %d/%d" % (len(closed_matches), len(annotations)))
    return closed_matches


def build_sample(sent, anns):
    lc_toks = [t.lower() for t in sent['sent']]
    sample = {
        "tokens": sent["sent"],
        "sent_id": sent["id"],
        "pps": [{
            "ind": ann['tok_ind'],
            "head_inds": [max(ann['tok_ind'] - 10, 0) + lc_toks[max(ann['tok_ind'] - 10, 0): ann['tok_ind']].index(head) for head in ann['heads.words']],
        } for ann in anns]
    }

    for (pp, ann) in zip(sample['pps'], anns):
        assert sample['tokens'][pp['ind']].lower() == ann['preps.words'][0]
        assert [sample['tokens'][head_ind].lower() for head_ind in pp['head_inds']] == ann['heads.words']
    return sample


def build_samples(sents, annotations):
    sent_id_to_sent = {s['id']: s for s in sents}

    annotations = [ann for ann in annotations if 'sent_id' in ann]

    sent_to_anns = {}
    for ann in annotations:
        sent_to_anns[ann['sent_id']] = sent_to_anns.get(ann['sent_id']) or []
        sent_to_anns[ann['sent_id']].append(ann)

    train = [build_sample(sent_id_to_sent[sent_id], anns) for sent_id, anns in sent_to_anns.items() if all(['2-21' in ann['id'] for ann in anns])]
    test = [build_sample(sent_id_to_sent[sent_id], anns) for sent_id, anns in sent_to_anns.items() if all(['wsj.23' in ann['id'] for ann in anns])]

    return train, test


def preprocess_samples(samples):
    preprocessed = []
    def process(sample):
        sample = copy.copy(sample)
        sample['preprocessing'] = preprocess_sentence(sample['tokens'])
        preprocessed.append(sample)
        print('%d/%d' % (len(preprocessed), len(samples)))
    with ThreadPoolExecutor(10) as tpe:
        list(tpe.map(process, samples))
    return preprocessed


def dump_dataset(train_samples, test_samples, train_path, test_path):
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_samples, f, indent=2)


if __name__ == '__main__':
    # sents = collect_sents()
    # annotations = collect_pp_annotations()
    train_path = os.path.dirname(__file__) + '/data/pp-data-english/train.json'
    test_path = os.path.dirname(__file__) + '/data/pp-data-english/test.json'

    if True:
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

        sys.exit(0)
        train_samples, test_samples = build_samples(sents, annotations)
    else:
        with open(train_path, 'r') as f:
            train_samples = json.load(f)
        with open(test_path, 'r') as f:
            test_samples = json.load(f)

    train_samples = preprocess_samples(train_samples)
    test_samples = preprocess_samples(test_samples)
    dump_dataset(train_samples, test_samples, train_path=train_path, test_path=test_path)

    # print(dropped)




