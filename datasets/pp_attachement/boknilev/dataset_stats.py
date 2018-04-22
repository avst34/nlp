import os
import json


def print_stats():
    anns_path = os.path.dirname(__file__) + '/data/pp-data-english/annotations.json'
    with open(anns_path) as f:
        anns = json.load(f)

    splits = [
        {
            'name': 'train',
            'data_path': os.path.dirname(__file__) + '/data/pp-data-english/train.json',
            'orig_ann_ids': [x['id'] for x in anns if 'wsj.2-21' in x['id'] and x['id'].count('_') == 1]
        },
        {
            'name': 'dev',
            'data_path': os.path.dirname(__file__) + '/data/pp-data-english/dev.json',
            'orig_ann_ids': [x['id'] for x in anns if 'wsj.22' in x['id'] and x['id'].count('_') == 1]
        },
        {
            'name': 'test',
            'data_path': os.path.dirname(__file__) + '/data/pp-data-english/test.json',
            'orig_ann_ids': [x['id'] for x in anns if 'wsj.23' in x['id'] and x['id'].count('_') == 1]
        }
    ]

    for split in splits:
        with open(split['data_path']) as f:
            data = json.load(f)
        sent_ids = [x['sent_id'] for x in data]
        assert len(sent_ids) == len(set(sent_ids))
        n_sents = len(sent_ids)

        ann_ids = [x['id'] for s in data for x in s['pps']]
        assert len(ann_ids) == len(set(ann_ids))
        n_anns = len(ann_ids)

        new_ann_ids = [x for x in ann_ids if x.count('_') > 1]
        n_new_anns = len(new_ann_ids)

        missing_old_ann_ids = set(split['orig_ann_ids']) - set(ann_ids)
        # print(missing_old_ann_ids)
        n_missing_old_ann_ids = len(missing_old_ann_ids)

        n_cands_per_sample = [len(x['head_cand_inds']) for s in data for x in s['pps']]
        assert all(n_cands_per_sample)
        avg_n_cands = sum(n_cands_per_sample) / len(n_cands_per_sample)

        n_govobj_hit = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] == (s['preprocessing']['govobj'][x['ind']]['gov'] or 0) - 1])
        n_govobj_near_miss = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] != (s['preprocessing']['govobj'][x['ind']]['gov'] or 0) - 1 and (s['preprocessing']['govobj'][x['ind']]['gov'] or 0) - 1 in x['head_cand_inds']])
        n_govobj_far_miss = len([x['id'] for s in data for x in s['pps'] if (s['preprocessing']['govobj'][x['ind']]['gov'] or 0) - 1 not in x['head_cand_inds']])

        n_udhead_hit = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] == s['preprocessing']['ud_head_ind'][x['ind']]])
        n_udhead_near_miss = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] != s['preprocessing']['ud_head_ind'][x['ind']] and s['preprocessing']['ud_head_ind'][x['ind']] in x['head_cand_inds']])
        n_udhead_far_miss = len([x['id'] for s in data for x in s['pps'] if s['preprocessing']['ud_head_ind'][x['ind']] not in x['head_cand_inds']])

        for s in data:
            s['ud_grandparents'] = [s['preprocessing']['ud_head_ind'][ind] if ind else None for ind in s['preprocessing']['ud_head_ind']]
        n_udgrandparent_hit = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] == s['ud_grandparents'][x['ind']]])
        n_udgrandparent_near_miss = len([x['id'] for s in data for x in s['pps'] if x['head_ind'] != s['ud_grandparents'][x['ind']] and s['ud_grandparents'][x['ind']] in x['head_cand_inds']])
        n_udgrandparent_far_miss = len([x['id'] for s in data for x in s['pps'] if s['ud_grandparents'][x['ind']] not in x['head_cand_inds']])
        print("Name: %s Anns: %d Sents: %d NewAnns: %d MissingAnns: %d AvgNCands: %d" % (split['name'], n_anns, n_sents, n_new_anns, n_missing_old_ann_ids, avg_n_cands))
        print("---  GovObjHit: %d GovObjNearMiss: %d GovObjFaMiss: %d" % (n_govobj_hit, n_govobj_near_miss, n_govobj_far_miss))
        print("---  UdHeadHit: %d UdHeadNearMiss: %d UdHeadFaMiss: %d" % (n_udhead_hit, n_udhead_near_miss, n_udhead_far_miss))
        print("---  UdGrandparentHit: %d UdGrandparentNearMiss: %d UdGrandparentFaMiss: %d" % (n_udgrandparent_hit, n_udgrandparent_near_miss, n_udgrandparent_far_miss))

if __name__ == '__main__':
    print_stats()