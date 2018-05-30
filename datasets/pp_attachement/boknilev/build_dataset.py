import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import copy

import sys
from nltk.corpus import ptb

from models.supersenses.preprocessing import preprocess_sentence as supersenses_preprocess
from models.hcpd.preprocessing import preprocess_sentence as hcpd_preprocess
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
    for sent in processed_sents:
        children = []
        for tok_ind, tok in enumerate(sent['sent']):
            children.append([i + 1 for i, head in enumerate(sent['head']) if head == tok_ind + 1])
        sent['children'] = children
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
    os.path.dirname(__file__) + '/data/pp-data-english/wsj.22.txt.dep.pp',
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
            'nheads',
            'sentinds'
            'sentids'
        ]
        fields_data = {}
        for field in fields:
            try:
                with open(base_file + '.' + field, 'r') as f:
                    fields_data[field] = [l.strip().replace('\t', ' ').split(' ') for l in f.readlines()]
            except FileNotFoundError as e:
                if field not in ['sentinds', 'sentids']:
                    raise
        cur_annotations = [{fld: data for fld, data in zip(fields, ann)} for ann in zip(*[fields_data[field] for field in fields if field in fields_data])]
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


def match_annotations(annotations, sents):
    ann_id_to_ann = {ann['id']: ann for ann in annotations}
    sent_id_to_sent = {sent['id']: sent for sent in sents}
    match_counts = []
    mismatch_reason = {}
    sents = copy.deepcopy(sents)
    for sent in sents:
        sent['word_next_pos'] = [(sent['sent'][ind], sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_next_pos_lc'] = [(sent['sent'][ind].lower(), sent['pos'][ind + 1]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_head'] = [(sent['sent'][ind], sent['head'][ind]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['word_head_lc'] = [(sent['sent'][ind].lower(), sent['head'][ind]) for ind, _ in enumerate(sent['sent'][:-1])]
        sent['sent_lc'] = [x.lower() for x in sent['sent']]

    try:
        with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_cache.json', 'r') as f:
            match_cands = {k: [tuple(vv) for vv in v]  for k, v in json.load(f).items() if 'wsj.22' not in k}
            match_cands_orig = copy.deepcopy(match_cands)
    except:
        match_cands = {}
        match_cands_orig = {}

    first_22_sent_ind = sents.index([x for x in sents if 'wsj_22' in x['id']][0])

    for ann_ind, ann in enumerate(annotations):
        if not match_cands.get(ann['id']) or len(match_cands.get(ann['id'])[0]) != 3 or 'wsj.22' in ann['id']:
            match_cands[ann['id']] = []
            if ann.get('sentinds'):
                sent_ind, pp_ind = [int(x) for x in ann['sentinds'][0].split(':')]
                assert 'wsj.22' in ann['id']
                sent_ind += first_22_sent_ind
                match_cands[ann['id']] = [(sents[sent_ind]['id'], pp_ind, True)]
                continue
            for filter_label in [True, False]:
                for sent_key, wnp_key, wh_key in [('sent', 'word_next_pos', 'word_head'), ('sent_lc', 'word_next_pos_lc', 'word_head_lc')]:
                    for sent in sents:
                        if 'wsj.23' in ann['id'] and 'wsj_23' not in sent['id']:
                            continue
                        if 'wsj.2-21' in ann['id'] and all([('wsj_%02d' % i) not in sent['id'] for i in range(2,22)]):
                            continue
                        if ann['children.words'][0] not in sent[sent_key]:
                            continue

                        for ind, (word, head) in enumerate(sent[wh_key]):
                            if word == ann['preps.words'][0] and \
                              (not(filter_label) or 0 <= (head - 1) and
                                sent[sent_key][head - 1] == ann['heads.words'][int(ann['labels'][0]) - 1]):
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
                                    match_cands[ann['id']].append((sent['id'], ind, filter_label))
                    if match_cands[ann['id']]:
                        break
                if match_cands[ann['id']]:
                    break

        if not match_cands[ann['id']]:
            print(ann)
            mismatch_reason[ann['id']] = 'No sentences matched'

        match_counts.append(len(match_cands[ann['id']]))
        print("%d/%d" % (ann_ind, len(annotations)))

    print('matched for wsj.2-21.txt.dep.pp_031739:', match_cands['wsj.2-21.txt.dep.pp_031739'])

    with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_cache.json', 'w') as f:
        json.dump(match_cands, f)

    for ann_id, cands in match_cands.items():
        if cands and not cands[0][2]:
            print("bad label", ann_id_to_ann[ann_id])
            if len(cands) > 1:
                mismatch_reason[ann_id] = 'Only matched sentences with incorrect labels'
                match_cands[ann_id] = []

    for ann_id, cands in match_cands.items():
        match_cands[ann_id] = [(x[0], x[1]) for x in cands]

    print(Counter(match_counts))

    # missing = [ann_id for ann_id in match_cands if ann_id not in matches]
    # print(sorted(m]issing))
    print('train:', len([ann_id for ann_id in match_cands if 'wsj.2-21.txt.dep.' in ann_id]))
    print('test:', len([ann_id for ann_id in match_cands if 'wsj.23.txt.dep.' in ann_id]))

    # for ann_id in sorted(match_cands):
    #     print(ann_id, matches.get(ann_id))

    matches_per_sent_pp = {}
    for ann_id, ann_matches in match_cands.items():
        for match in ann_matches:
            matches_per_sent_pp[match] = matches_per_sent_pp.get(match) or []
            matches_per_sent_pp[match].append(ann_id_to_ann[ann_id])

    print("Matches: %d Anns: %d" % (len(matches_per_sent_pp), len(annotations)))

    def m():
        print('Clearing by single..')
        i = 0
        while True:
            took_action = False
            closed_matches = {ann_id: matches[0] for ann_id, matches in match_cands.items() if len(matches) == 1}
            for ann_id, match in list(closed_matches.items()):
                if ann_id not in closed_matches:
                    continue
                for ann in list(matches_per_sent_pp[match]):
                    if ann['id'] != ann_id:
                        took_action = True
                        assert match in match_cands[ann['id']]
                        match_cands[ann['id']].remove(match)
                        if closed_matches.get(ann['id']) == match:
                            del closed_matches[ann['id']]
                        matches_per_sent_pp[match].remove(ann)
                        if not match_cands[ann['id']]:
                            print(i, "Matches cleared due to a better fit (single option)", ann['id'])
                            mismatch_reason[ann['id']] = "Matches cleared due to a better fit (single option)"
            print("plen", len(closed_matches))
            if not took_action:
                break
            i += 1

    m()

    for match, anns in matches_per_sent_pp.items():
        anns.sort(key=lambda ann:-len(ann['heads.words']))
        longest = [ann for ann in anns if len(ann['heads.words']) == len(anns[0]['heads.words'])]
        for ann in list(anns):
            if ann not in longest:
                match_cands[ann['id']].remove(match)
                matches_per_sent_pp[match].remove(ann)
                if not match_cands[ann['id']]:
                    mismatch_reason[ann['id']] = "Matches cleared due to a better fit (more heads)"

    m()

    closed_matches = {ann_id: matches[0] for ann_id, matches in match_cands.items() if len(matches) == 1}

    # closed_matches_missed_label = {anns[0]['id']: match for match, anns in matches_per_sent_pp.items() if len(anns) == 1 and not match[2]}

    new_annotations = []

    handled = {}
    for ann_id in list(match_cands):
        if ann_id in closed_matches or ann_id in mismatch_reason:
            continue

        ann_matches = match_cands[ann_id]
        assert len(ann_matches) > 1

        all_anns = [ann for match in ann_matches for ann in matches_per_sent_pp[match]]
        all_anns = [ann for ind, ann in enumerate(all_anns) if all_anns.index(ann) == ind]
        all_matches = ann_matches

        found_inconsistency = False
        for ann1 in all_anns:
            for ann2 in all_anns:
                if ann1['heads.words'] != ann2['heads.words'] or \
                   ann1['preps.words'] != ann2['preps.words'] or \
                   ann1['children.words'] != ann2['children.words'] or \
                   ann1['labels'] != ann2['labels']:
                    found_inconsistency = True
                    print(ann1)
                    print(ann2)
                    break
            if found_inconsistency:
                break

        handled.update({ann['id']: True for ann in all_anns})
        if found_inconsistency:
            print('WARNING: inconsistency found, skipping %d anns' % len(all_anns))
            for ann in all_anns:
                mismatch_reason[ann['id']] = 'Found inconsistency'
            continue

        if len(all_anns) == len(all_matches):
            print("Found matching ann-match sets of size", len(all_anns))
            for mann, mmatch in zip(all_anns, all_matches):
                match_cands[mann['id']] = [mmatch]
        elif len(all_anns) > len(all_matches):
            print("More anns than matches! %d > %d" % (len(all_anns), len(all_matches)))
            for ann in all_anns:
                mismatch_reason[ann['id']] = 'More anns than matches'
        else:
            print("Less anns than matches! %d < %d" % (len(all_anns), len(all_matches)))
            for i in range(len(all_matches) - len(all_anns)):
                nann = copy.deepcopy(all_anns[0])
                nann['copy_of'] = nann['id']
                nann['id'] = nann['id'] + '_' + str(i)
                new_annotations.append(nann)
                ann_id_to_ann[nann['id']] = nann
                all_anns.append(nann)
                for match in all_matches:
                    matches_per_sent_pp[match].append(nann)
            for mann, mmatch in zip(all_anns, all_matches):
                match_cands[mann['id']] = [mmatch]
        closed_matches.update({ann_id: match_cands[ann_id][0] for ann in all_anns for ann_id in [ann['id']] if len(match_cands[ann_id]) == 1})

    print("Matched %d/%d+%d" % (len(closed_matches), len(annotations), len(new_annotations)))
    # print("Matched (missed label) %d/%d" % (len(closed_matches_missed_label), len(annotations)))
    # print("Matched (total) %d/%d" % (len(closed_matches_missed_label) + len(closed_matches), len(annotations)))
    print("Unmatched: %d" % (len(annotations) + len(new_annotations) - len(closed_matches)))

    print("With at least 1 match: %d" % len([ann_id for ann_id, cands in match_cands.items() if cands]))
    print("Mismatched with reason: %d" % len(mismatch_reason))
    print(Counter(mismatch_reason.values()))

    print("Matches: %d Anns: %d" % (len([x for x in matches_per_sent_pp.values() if x]), len(annotations)))

    closed_matches_per_sent_pp = {}
    for ann_id, match in closed_matches.items():
        closed_matches_per_sent_pp[match] = closed_matches_per_sent_pp.get(match) or []
        closed_matches_per_sent_pp[match].append(ann_id_to_ann[ann_id])

    print("Per pp:", Counter([len(x) for x in closed_matches_per_sent_pp.values()]))

    # missing = [ann_id_to_ann[ann_id] for ann_id in match_cands if ann_id not in closed_matches]
    # orig_matches_per_sent_pp = {}
    # for ann_id, ann_matches in match_cands_orig.items():
    #     for match in ann_matches:
    #         orig_matches_per_sent_pp[(match[0], match[1])] = orig_matches_per_sent_pp.get((match[0], match[1])) or []
    #         orig_matches_per_sent_pp[(match[0], match[1])].append(ann_id_to_ann[ann_id])

    for ann_id, reason in mismatch_reason.items():
        if reason == 'No sentences matched':
            print(ann_id)

    # for ann in missing:
    #     cands = [(x[0], x[1]) for x in match_cands_orig[ann['id']]]
    #     available = [c for c in cands if c not in closed_matches_per_sent_pp]
    #     print("%d/%d available for %s (%s)" % (len(available), len(cands), ann['id'], mismatch_reason[ann['id']]))
    #     if len(available) == 1 and len(cands) == 1 and mismatch_reason[ann['id']] == 'Matches cleared due to a better fit (single option)':
    #         m = available[0]
    #         print(m, 'matched:', [(x['id'], mismatch_reason.get(x['id'])) for x in orig_matches_per_sent_pp[m]])
    #
    for ann_id, (sent_id, tok_ind) in closed_matches.items():
        ann = ann_id_to_ann[ann_id]
        ann['sent_id'] = sent_id
        # ann['sent_tokens'] = sent_id_to_sent[sent_id]['sent']
        ann['tok_ind'] = tok_ind

    print('len(closed_matches)', len(closed_matches))

    annotations.extend(new_annotations)
    return closed_matches


def build_sample(sent, anns):
    lc_toks = [t.lower() for t in sent['sent']]
    try:
        sample = {
            "tokens": sent["sent"],
            "sent_id": sent["id"],
            "pps": [{
                "id": ann['id'],
                "copy_of": ann.get('copy_of') or ann['id'],
                "ind": ann['tok_ind'],
                "child_ind": lc_toks.index(ann['children.words'][0]),
                # "child_ind": ann['tok_ind'] + lc_toks[ann['tok_ind']: ann['tok_ind'] + 200].index(ann['children.words'][0]),
                "head_cand_inds": [max(ann['tok_ind'] - 10, 0) + lc_toks[max(ann['tok_ind'] - 10, 0): ann['tok_ind']].index(head) for head in ann['heads.words']],
            } for ann in anns]
        }
        for pp, ann in zip(sample['pps'], anns):
            pp['head_ind'] = pp['head_cand_inds'][int(ann['labels'][0]) - 1]

    except:
        raise

    for (pp, ann) in zip(sample['pps'], anns):
        assert sample['tokens'][pp['ind']].lower() == ann['preps.words'][0]
        assert [sample['tokens'][head_ind].lower() for head_ind in pp['head_cand_inds']] == ann['heads.words']
    return sample


def build_samples(sents, annotations):
    sent_id_to_sent = {s['id']: s for s in sents}

    print('len(annotations) - before', len(annotations))
    annotations = [ann for ann in annotations if 'sent_id' in ann]
    print('len(annotations) - after', len(annotations))

    sent_to_anns = {}
    for ann in annotations:
        sent_to_anns[ann['sent_id']] = sent_to_anns.get(ann['sent_id']) or []
        sent_to_anns[ann['sent_id']].append(ann)

    train = [build_sample(sent_id_to_sent[sent_id], anns) for sent_id, anns in sent_to_anns.items() if all(['2-21' in ann['id'] for ann in anns])]
    dev = [build_sample(sent_id_to_sent[sent_id], anns) for sent_id, anns in sent_to_anns.items() if all(['wsj.22' in ann['id'] for ann in anns])]
    test = [build_sample(sent_id_to_sent[sent_id], anns) for sent_id, anns in sent_to_anns.items() if all(['wsj.23' in ann['id'] for ann in anns])]

    return train, dev, test


def preprocess_samples(samples):
    preprocessed = []
    def process(sample):
        sample = copy.copy(sample)
        sample['preprocessing'] = supersenses_preprocess(sample['tokens'])
        sample['preprocessing'].update(hcpd_preprocess(sample['tokens']))
        preprocessed.append(sample)
        # print('%d/%d' % (len(preprocessed), len(samples)))
    with ThreadPoolExecutor(10) as tpe:
        list(tpe.map(process, samples))
    return preprocessed


def set_head_ind(samples, anns):
    ann_id_to_ann = {x['id']: x for x in anns}
    for sample in samples:
        for pp in sample['pps']:
            ann = ann_id_to_ann[pp['copy_of']]
            assert len(pp['head_cand_inds']) == len(ann['heads.words'])
            assert len(pp['head_cand_inds']) == len(ann['heads.pos'])
            assert len(ann['heads.next.pos']) in [len(pp['head_cand_inds']), len(pp['head_cand_inds']) - 1]
            pp['head_cand_inds'] = pp.get('head_cand_inds') or pp['head_inds']
            pp['head_cands'] = [
                {
                    "ind": head_token_ind,
                    "gold": {
                        "is_verb": ann['heads.pos'][head_ind] == "1",
                        "is_noun": ann['heads.pos'][head_ind] == "-1",
                        "next_pos": (ann['heads.next.pos'] + [None])[head_ind]
                    }
                }
                for head_ind, head_token_ind in enumerate(pp.get('head_cand_inds') or pp['head_inds'])
            ]
            pp['head_ind'] = pp['head_cand_inds'][int(ann['labels'][0]) - 1]
            lc_toks = [t.lower() for t in sample['tokens']]
            try:
                pp["child_ind"] = pp.get('child_ind') or pp['ind'] + lc_toks[pp['ind']: pp['ind'] + 200].index(ann['children.words'][0]),
            except:
                try:
                    pp["child_ind"] = pp.get('child_ind') or max(pp['ind'] - 1, 0) + lc_toks[max(pp['ind'] - 1, 0): pp['ind'] + 200].index(ann['children.words'][0]),
                except:
                    try:
                        pp["child_ind"] = pp.get('child_ind') or max(pp['ind'] - 20, 0) + lc_toks[max(pp['ind'] - 20, 0): pp['ind'] + 200].index(ann['children.words'][0]),
                    except:
                        print(sample)
                        # raise

def add_preprocessing(samples):
    for s in samples:
        s['preprocessing'].update(hcpd_preprocess(s['tokens']))

def dump_dataset(train_samples, dev_samples, test_samples, train_path, dev_path, test_path):
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(dev_path, 'w') as f:
        json.dump(dev_samples, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_samples, f, indent=2)


if __name__ == '__main__':
    # sents = collect_sents()
    try:
        with open(os.path.dirname(__file__) + '/data/pp-data-english/annotations.json', 'r') as f:
            annotations = json.load(f)
    except:
        print('Collecting annotations')
        annotations = collect_pp_annotations()
    train_path = os.path.dirname(__file__) + '/data/pp-data-english/train.json'
    dev_path = os.path.dirname(__file__) + '/data/pp-data-english/dev.json'
    test_path = os.path.dirname(__file__) + '/data/pp-data-english/test.json'

    if False:
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

        matches = match_annotations(annotations, sents)
        with open(os.path.dirname(__file__) + '/data/pp-data-english/wsj_matches.json', 'w') as f:
            json.dump(matches, f)

        print("annotations:", len(annotations))
        train_samples, dev_samples, test_samples = build_samples(sents, annotations)
        print("00 train", len([pp for t in train_samples for pp in t['pps']]),
              "dev", len([pp for t in dev_samples for pp in t['pps']]),
              "test", len([pp for t in test_samples for pp in t['pps']]),
              "all", len([pp for t in test_samples for pp in t['pps']]) + len([pp for t in train_samples for pp in t['pps']])
              )
    else:
        with open(train_path, 'r') as f:
            train_samples = json.load(f)
        with open(dev_path, 'r') as f:
            dev_samples = json.load(f)
        with open(test_path, 'r') as f:
            test_samples = json.load(f)

    # set_head_ind(train_samples, annotations)
    # set_head_ind(dev_samples, annotations)
    # set_head_ind(test_samples, annotations)

    add_preprocessing(train_samples)
    add_preprocessing(dev_samples)
    add_preprocessing(test_samples)

    # train_samples = preprocess_samples(train_samples)
    # dev_samples = preprocess_samples(dev_samples)
    # test_samples = preprocess_samples(test_samples)

    print("train", len([pp for t in train_samples for pp in t['pps']]),
          "test", len([pp for t in test_samples for pp in t['pps']]),
          "all", len([pp for t in test_samples for pp in t['pps']]) + len([pp for t in train_samples for pp in t['pps']])
          )
    dump_dataset(train_samples, dev_samples, test_samples, train_path=train_path, dev_path=dev_path, test_path=test_path)

    # print(dropped)




