import json
import sys

words = sys.argv[1:]

with open('sents.json') as f:
    sents = json.load(f)
    for sent in sents:
        if all(w in [t.lower() for t in sent['sent']] for w in words):
            print(sent['id'])
            print('--------')
            for t, p in zip(sent['sent'], sent['pos']):
                print(t, p)
