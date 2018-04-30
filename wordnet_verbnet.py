from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn



def get_noun_hypernyms(word, hypernyms_set=None):
    hypernyms = set()
    queue = wn.synsets(word, 'n')
    while queue:
        ss = queue.pop()
        queue.extend(ss.hypernyms())
        if not hypernyms_set or ss.name() in hypernyms_set:
            hypernyms.add(ss.name())
    return list(hypernyms)

def is_word_in_verb_frames(verb, word):
    classids = vn.classids(verb)
    frames = [frame for cid in classids for frame in vn.frames(cid)]
    for frame in frames:
        if word.lower() in frame['example'].lower().replace('.', '').split(' '):
            return True
    return False

