from collections import defaultdict

from datasets.streusle_v4 import StreusleLoader

DROP = ['circa', 're', 'plus', "", 'and']
TYPOS = {
    'fo': 'for',
    'fot': 'for',
    "it's": 'its',
    'thru': 'through',
    'int': 'in',
    's': "'s",
    "'": "'s",
    'abou': 'about',
    'a': 'at',
    'you': 'your',
    'thier': 'their',
    'till': 'untill',
    'it': 'its',
    '@': 'at',
    'ur': 'your',
    'btwn': 'between',
    '4': 'for',
    'untill': 'until'
}

def calc_ss_per_pp_dist(records, pss="role", as_probabilities=True):
    assert pss in ['role', 'func']
    dropped = 0
    pps = 0
    pss_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for rec in records:
        for ttok in rec.tagged_tokens:
            if ttok.supersense_role:
                if not ttok.is_part_of_wmwe and not ttok.is_part_of_smwe:
                    pp = ttok.token.lower()
                    if pp in DROP:
                        dropped += 1
                        continue
                    pps += 1
                    pp = TYPOS.get(pp, pp)
                    pss_counts[pp][ttok.supersense_role if pss == 'role' else ttok.supersense_func] += 1
                else:
                    dropped += 1

    if as_probabilities:
        for prep, counts in pss_counts.items():
            tot = sum(counts.values())
            for pss in counts:
                counts[pss] /= tot

    return {p: {pss: c for pss, c in pss_counts[p].items()} for p in pss_counts}

if __name__ == '__main__':
    records = StreusleLoader().load()
    ss_per_pp = calc_ss_per_pp_dist(records, as_probabilities=False)
    print(ss_per_pp)