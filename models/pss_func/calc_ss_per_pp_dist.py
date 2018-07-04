from collections import defaultdict

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.prepositions_ordering import DROP, TYPOS


def calc_ss_per_pp_dist(records, pss="role", as_probabilities=True, top_k=None):
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

    preps = [x[0] for x in sorted({(p, sum(pss_counts[p].values())) for p in pss_counts.keys()})[::-1]]
    if top_k:
        preps = preps[:top_k]
    return {p: {pss: c for pss, c in pss_counts[p].items()} for p in preps}

if __name__ == '__main__':
    records = StreusleLoader().load()
    ss_per_pp = calc_ss_per_pp_dist(records, as_probabilities=False)
    print(ss_per_pp)