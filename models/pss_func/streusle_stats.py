from collections import namedtuple, Counter
from pprint import pprint

import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.stats import pearsonr

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.eval_func_pred import eval_type_level_func_pred_token_level
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
                prep = ' '.join([rec.get_tok_by_ud_id(toknum).token for toknum in ttok.we_toknums]).lower()
                if prep in DROP:
                    continue
                prep = TYPOS.get(prep, prep)
                tags.append(Tag(prep, ttok.supersense_role, ttok.supersense_func))

    return tags

def prep_frequency(tags, normalize=False):
    dist = Counter([t.prep for t in tags])
    if normalize:
        dist = {k: v/len(tags) for k, v in dist.items()}
    return dist

def label_distribution_per_prep(tags, pss_type='ss_role', bottom_k=None, normalize=False):
    assert pss_type in ['ss_role', 'ss_func']
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

def label_distribution(tags, pss_type='ss_role', bottom_k=None, normalize=False):
    assert pss_type in ['ss_role', 'ss_func']
    prep_dist = label_distribution_per_prep(tags, pss_type, bottom_k, normalize=False)
    dist = {}
    for prep, ss_dist in prep_dist.items():
        for ss, count in ss_dist.items():
            dist[ss] = dist.get(ss) or 0
            dist[ss] += count
    if normalize:
        for prep, ss_dist in prep_dist.items():
            for ss in ss_dist:
                dist[ss] /= len(tags)
    return dist

def prep_role_func_divergence(tags, normalized=True):
    preps_freq = prep_frequency(tags)
    preps_div = {
        p: len([tag for tag in tags if tag.ss_func != tag.ss_role and tag.prep == p]) / (total if normalized else 1)
        for p, total in preps_freq.items()
    }
    return preps_div

def overall_role_func_divergence(tags):
    prep_div = prep_role_func_divergence(tags, normalized=False)
    return sum(prep_div.values())/len(tags)

def mf_pss_per_prep(tags, pss_type='ss_role'):
    assert pss_type in ['ss_role', 'ss_func']
    dist = label_distribution_per_prep(tags, pss_type)
    mf_per_prep = {prep: max(dist[prep].keys(), key=lambda label: dist[prep][label]) for prep in dist}
    return mf_per_prep


def eval_mf_baseline(tags):
    mf_per_prep = mf_pss_per_prep(tags, 'ss_role')
    return eval_type_level_func_pred_token_level(mf_per_prep, tags)

def eval_mf_baseline_per_prep(tags):
    mf_per_prep = mf_pss_per_prep(tags, 'ss_role')
    prep_acc = {}
    for prep in mf_per_prep:
        prep_acc[prep] = eval_type_level_func_pred_token_level({prep: mf_per_prep[prep]}, tags)
    return prep_acc


def plot(points, x_label, y_label, x_tick_labels=None, y_tick_labels=None, title=None, show=True, color=None, label=None, lin_reg=False):
    def numerize(values):
        try:
            values = [float(x) for x in values]
            return values, None
        except ValueError as e:
            values_set = sorted(set(values))
            return [values_set.index(v) for v in values], values_set

    x_values, _ = numerize([p[0] for p in points])
    y_values, _ = numerize([p[1] for p in points])
    plt.scatter(x_values, y_values, color=color, label=label)
    if x_tick_labels:
        plt.xticks([p[0] for p in points], x_tick_labels, wrap=True, rotation='vertical', size='xx-small')
    if y_tick_labels:
        plt.yticks([p[1] for p in points], y_tick_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    reg_str = None
    if lin_reg:
        x, y = zip(*points)
        fit = polyfit(x, y, 1)
        fit_fn = poly1d(fit)
        plt.plot(x, fit_fn(x), 'k')
        # reg_str = str(fit)
        correl = pearsonr(x_values, y_values)[0]
        reg_str = str(correl)
    if title:
        if reg_str:
            title += ' (%s)' % reg_str
        plt.title(title)
    if show:
        if label:
            plt.legend()
        plt.show()

def generate_reports():
    tags = collect_train_dev_tags()
    print('%d tags' % len(tags))

    print('Overall divergence:', overall_role_func_divergence(tags))
    # prep frequency
    prep_freq = prep_frequency(tags)
    pprint(list(enumerate(sorted(prep_freq.items(), key=lambda x: -x[1]))))
    plot([(x[0], x[1][1]) for x in list(enumerate(sorted(prep_freq.items(), key=lambda x: x[1])))],
         x_label='prep',
         y_label='freq',
         x_tick_labels=[x[0] for x in sorted(prep_freq.items(), key=lambda x: x[1])],
         title='Prepositions Frequency'
     )

    # dist tail ends at ~ 181


    # # role/func label dist
    role_label_dist = label_distribution(tags, 'ss_role')
    func_label_dist = label_distribution(tags, 'ss_func')
    preps = [p for p in func_label_dist if p not in role_label_dist] + sorted(role_label_dist.keys(), key=lambda prep: role_label_dist[prep])
    #
    # plot([(preps.index(prep), role_label_dist.get(prep, 0)) for prep in preps],
    #      x_label='role',
    #      y_label='freq',
    #      x_tick_labels=preps,
    #      title='Role/Func Frequency',
    #      color='blue',
    #      label='role',
    #      show=False
    #      )
    #
    # # func label dist
    # plot([(preps.index(prep), func_label_dist.get(prep, 0)) for prep in preps],
    #      x_label='PSS',
    #      y_label='freq',
    #      x_tick_labels=preps,
    #      title='Role/Function Frequency',
    #      color='red',
    #      label='function',
    #      )

    # # role/func label dist
    # role_label_dist = label_distribution(tags, 'ss_role', bottom_k=181)
    # func_label_dist = label_distribution(tags, 'ss_func', bottom_k=181)
    # # preps = [p for p in func_label_dist if p not in role_label_dist] + sorted(role_label_dist.keys(), key=lambda prep: role_label_dist[prep])
    #
    # plot([(preps.index(prep), role_label_dist.get(prep, 0)) for prep in preps],
    #      x_label='role',
    #      y_label='freq',
    #      x_tick_labels=preps,
    #      title='Role/Func Frequency (bottom 181)',
    #      color='blue',
    #      label='role',
    #      show=False
    #      )
    #
    # # func label dist
    # plot([(preps.index(prep), func_label_dist.get(prep, 0)) for prep in preps],
    #      x_label='PSS',
    #      y_label='freq',
    #      x_tick_labels=preps,
    #      title='Role/Function Frequency (bottom 181)',
    #      color='red',
    #      label='function',
    #      )

    # role/func label dist
    role_label_dist_full = label_distribution(tags, 'ss_role', normalize=True)
    func_label_dist_full = label_distribution(tags, 'ss_func', normalize=True)
    role_label_dist = label_distribution(tags, 'ss_role', bottom_k=181, normalize=True)
    func_label_dist = label_distribution(tags, 'ss_func', bottom_k=181, normalize=True)
    # preps = [p for p in func_label_dist if p not in role_label_dist] + sorted(role_label_dist.keys(), key=lambda prep: role_label_dist[prep])

    plot([(preps.index(prep), role_label_dist.get(prep, 0)/role_label_dist_full.get(prep) if role_label_dist_full.get(prep) else -1) for prep in preps],
         x_label='role',
         y_label='freq',
         x_tick_labels=preps,
         title='Role/Func Frequency (bottom 181)',
         color='blue',
         label='role',
         show=False
         )

    # func label dist
    plot([(preps.index(prep), func_label_dist.get(prep, 0)/func_label_dist_full.get(prep) if func_label_dist_full.get(prep) else -1) for prep in preps],
         x_label='PSS',
         y_label='freq',
         x_tick_labels=preps,
         title='Role/Function Frequency (bottom 181)',
         color='red',
         label='function',
         )

    # # role label dist (bottom 181)
    # role_label_dist = label_distribution(tags, 'ss_role', bottom_k=181)
    # plot([(x[0], x[1][1]) for x in list(enumerate(sorted(role_label_dist.items(), key=lambda x: x[1])))],
    #      x_label='role',
    #      y_label='freq',
    #      x_tick_labels=[x[0] for x in sorted(role_label_dist.items(), key=lambda x: x[1])],
    #      title='Role Frequency (over bottom 181 preps)',
    #      color='blue',
    #      label='role',
    #      show=False
    #      )
    #
    # # func label dist (bottom 181)
    # func_label_dist = label_distribution(tags, 'ss_func', bottom_k=181)
    # plot([(x[0], x[1][1]) for x in list(enumerate(sorted(func_label_dist.items(), key=lambda x: x[1])))],
    #      x_label='func',
    #      y_label='freq',
    #      x_tick_labels=[x[0] for x in sorted(func_label_dist.items(), key=lambda x: x[1])],
    #      title='Role/Function Frequency (over bottom 181 preps)',
    #      color='red',
    #      label='func',
    # )

    # # Freq (normalized) and divergence for each prep
    # prep_div = prep_role_func_divergence(tags)
    # prep_freq_norm = prep_frequency(tags, normalize=True)
    # preps = sorted(prep_div.keys(), key=lambda prep: prep_freq_norm[prep])
    # plot(points=[(ind, prep_freq_norm[prep]) for ind, prep in enumerate(preps)],
    #      x_label='prep', y_label='freq/div', title='Preposition Frequency vs. Divergence',
    #      color='blue', label='frequency', show=False)
    # plot(points=[(ind, prep_div[prep]) for ind, prep in enumerate(preps)],
    #      x_label='prep', y_label='freq/div', title='Preposition Frequency vs. Divergence',
    #      x_tick_labels=preps,
    #      color='red', label='divergence')

    # # Freq vs. divergence
    # prep_div = prep_role_func_divergence(tags)
    # prep_freq = prep_frequency(tags)
    # prep_freq_norm = prep_frequency(tags, normalize=True)
    # preps = sorted(prep_div.keys(), key=lambda prep: prep_freq_norm[prep])
    # plot(points=[(prep_freq_norm[prep], prep_div[prep]) for ind, prep in enumerate(preps) if prep_freq[prep] > 20],
    #      x_label='prep-freq', y_label='prep-divergence', title='Preposition Frequency vs. Divergence (2)', lin_reg=True)

    # # MF accuracy per prep
    # mf_acc_per_prep = eval_mf_baseline_per_prep(tags)
    # preps = sorted(mf_acc_per_prep.keys(), key=lambda prep: mf_acc_per_prep[prep]['acc'])
    # plot(points=[(ind, mf_acc_per_prep[prep]['acc']) for ind, prep in enumerate(preps)],
    #      x_label='prep', y_label='MF Basline Acc', title='MF Basline Acc per Prep', x_tick_labels=preps)

    # # Freq (normalized) and MF acc for each prep
    # mf_acc_per_prep = eval_mf_baseline_per_prep(tags)
    # prep_freq_norm = prep_frequency(tags, normalize=True)
    # preps = sorted(mf_acc_per_prep.keys(), key=lambda prep: prep_freq_norm[prep])
    # plot(points=[(ind, prep_freq_norm[prep]) for ind, prep in enumerate(preps)],
    #      x_label='prep', y_label='freq/acc', title='Preposition Frequency vs. MF Baseline Acc.',
    #      color='blue', label='frequency', show=False)
    # plot(points=[(ind, mf_acc_per_prep[prep]['acc']) for ind, prep in enumerate(preps)],
    #      x_label='prep', y_label='freq/acc', title='Preposition Frequency vs. MF Baseline Acc.',
    #      x_tick_labels=preps,
    #      color='red', label='MF accuracy')

    # # Freq vs. Acc
    # mf_acc_per_prep = eval_mf_baseline_per_prep(tags)
    # prep_freq = prep_frequency(tags)
    # prep_freq_norm = prep_frequency(tags, normalize=True)
    # preps = sorted(mf_acc_per_prep.keys(), key=lambda prep: prep_freq_norm[prep])
    # plot(points=[(prep_freq_norm[prep], mf_acc_per_prep[prep]['acc']) for ind, prep in enumerate(preps) if prep_freq[prep] > 5],
    #      x_label='prep-freq', y_label='MF Acc.', title='Preposition Frequency vs. MF Acc (2)', lin_reg=True)

    # mf_baseline_token_level_acc = eval_mf_baseline(tags)['acc']
    # print('Token level MF baseline accuracy:', mf_baseline_token_level_acc)

if __name__ == '__main__':
    print(generate_reports())