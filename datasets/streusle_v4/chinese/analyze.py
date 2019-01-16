import json
import os
from collections import Counter

sents = json.load(open(os.path.dirname(__file__) + '/lp.eng.zh_pss.all.json'))


def get(sent, ind):
    if not ind:
        return "None"
    return sent['toks'][ind - 1]['word']


def gp(sent, ind):
    if not ind:
        return None
    p = sent['toks'][ind - 1]
    return get(sent, p['head'])


matched = []
unmatched = []
for sent in sents:
    for t in sent['toks']:
        if t['full_ss']:
            if not t['word'].startswith('MISSING'):
                matched.append(t)
            else:
                unmatched.append(t)
print('matched', len(matched), 'unmatched', len(unmatched))

psses = []
for sent in sents:
    for t in sent['toks']:
        if t['full_ss']:
            if not t['word'].startswith('MISSING'):
                psses.extend(t['full_ss'].split('~'))

for t in sorted(Counter(psses).items(), key=lambda x: -x[1]):
    print(t, len(psses))

for sent in sents[:100]:
    print(sent['text'])
    print(sent['zh_text'])
    for t in sent['toks']:
        if t['full_ss']:
            if t['word'].startswith('MISSING'):
                print("X %s parent: %s, grandparent: %s     -->    %s" % (t['zh'], get(sent, t['head'] if t['head'] else None), get(sent, t['grandparent_override'] if t['grandparent_override'] else None), t['full_ss']))
            else:
                print("V %s parent: %s, grandparent: %s     -->    %s" % (t['word'], get(sent, t['head']), gp(sent, t['head']), t['full_ss']))
    print()
    print()

