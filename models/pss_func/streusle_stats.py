from collections import namedtuple, Counter

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.prepositions_ordering import DROP, TYPOS

Tag = namedtuple('Tag', ['prep', 'ss_role', 'ss_func'])

def collect_train_dev_tags():
    train_records = StreusleLoader().load_train()
    dev_records = StreusleLoader().load_dev()
    records = train_records + dev_records

    tags = []
    for rec in records:
        for ttok in rec.tagged_tokens:
            if ttok.supersense_role:
                if ttok.is_part_of_wmwe:
                    continue
                prep = ' '.join([rec.get_tok_by_ud_id(toknum) for toknum in ttok.we_toknums]).lower()
                if prep in DROP:
                    continue
                prep = TYPOS.get(prep, prep)
                tags.append(Tag(prep, ttok.supersense_role, ttok.supersense_func))

    return tags

def prep_frequency(tags):
    return Counter([t.prep for t in tags])

def label_distribution(tags, pss_type='ss_role', bottom_k=None, normalize=False):
    assert pss_type in 'ss_role', 'ss_func'
    prep_dist = {}
    for tag in tags:
        ss = getattr(tag, pss_type)
        prep_dist[tag.prep] = prep_dist.get(tag.prep, {})
        prep_dist[tag.prep][ss] = prep_dist[tag.prep].get(ss, 0)
        prep_dist[tag.prep][ss] += 1
    if normalize:
        for prep in prep_dist:
            total = sum(prep_dist[prep].values())
            for ss in prep_dist[prep]:
                prep_dist[prep_dist][ss] /= total
    if bottom_k:
        freq = prep_frequency(tags)
        bottom_preps = sorted(freq, key=lambda prep: freq[prep])[:bottom_k]
        prep_dist = {k: v for k,v in prep_dist.items() if k in bottom_preps}
    return prep_dist

def prep_role_func_divergence(tags):
    preps_freq = prep_frequency(tags)
    preps_div = {
        p: len([tag for tag in tags if tag.ss_func != tag.ss_role and tag.prep == p]) / total
        for p, total in preps_freq.items()
    }
    return preps_div

