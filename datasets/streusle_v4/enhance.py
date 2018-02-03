import os
from collections import namedtuple
import random
import math

import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import subprocess

import os

import supersenses
import spacy

from utils import parse_conll

try:
    nlp = spacy.load('en')
except:
    nlp = spacy.load('en_core_web_sm')


from spacy.tokens import Doc
from numpy import dot
from numpy.linalg import norm

from datasets.streusle_v4 import streusle
from word2vec import Word2VecModel
import json

import conllu_parser

loader = streusle.StreusleLoader()
records_list = loader.load()
if len(records_list) == 2:
    train_records, dev_records, test_records = records_list[0], [], records_list[1]
elif len(records_list) == 3:
    train_records, dev_records, test_records = records_list

records = sum(records_list, [])
id_to_record = {r.id: r for r in records}
print("records: %d" % len(records))

def enhance_word2vec():
    # collect word2vec vectors for words in the data
    all_tokens = set()
    for rec in records:
        for tagged_token in rec.tagged_tokens:
            all_tokens.add(tagged_token.token)

    wvm = Word2VecModel.load_google_model()
    missing_words = wvm.collect_missing(all_tokens)
    with open(streusle.ENHANCEMENTS.WORD2VEC_PATH, 'wb') as f:
        wvm.dump(all_tokens, f, skip_missing=True)

    with open(streusle.ENHANCEMENTS.WORD2VEC_MISSING_PATH, 'w') as f:
        json.dump(missing_words, f, indent=2)

    print('Enhanced with word2vec, %d words in total (%d skipped)' % (len(all_tokens), len(missing_words)))

def enhance_ud_lemmas_word2vec():
    # collect word2vec vectors for words in the data
    all_lemmas = set()
    for rec in records:
        for tagged_token in rec.tagged_tokens:
            all_lemmas.add(tagged_token.ud_lemma)

    wvm = Word2VecModel.load_google_model()
    missing_words = wvm.collect_missing(all_lemmas)
    with open(streusle.ENHANCEMENTS.UD_LEMMAS_WORD2VEC_PATH, 'wb') as f:
        wvm.dump(all_lemmas, f, skip_missing=True)

    with open(streusle.ENHANCEMENTS.UD_LEMMAS_WORD2VEC_MISSING_PATH, 'w') as f:
        json.dump(missing_words, f, indent=2)

    print('Enhanced with word2vec, %d words in total (%d skipped)' % (len(all_lemmas), len(missing_words)))

def enhance_spacy_lemmas_word2vec():
    # collect word2vec vectors for words in the data
    all_lemmas = set()
    for rec in records:
        for tagged_token in rec.tagged_tokens:
            all_lemmas.add(tagged_token.spacy_lemma)

    wvm = Word2VecModel.load_google_model()
    missing_words = wvm.collect_missing(all_lemmas)
    with open(streusle.ENHANCEMENTS.SPACY_LEMMAS_WORD2VEC_PATH, 'wb') as f:
        wvm.dump(all_lemmas, f, skip_missing=True)

    with open(streusle.ENHANCEMENTS.SPACY_LEMMAS_WORD2VEC_MISSING_PATH, 'w') as f:
        json.dump(missing_words, f, indent=2)

    print('Enhanced with word2vec, %d words in total (%d skipped)' % (len(all_lemmas), len(missing_words)))

def enhance_corenlp_lemmas_word2vec():
    # collect word2vec vectors for words in the data
    all_lemmas = set()
    for rec in records:
        for tagged_token in rec.tagged_tokens:
            all_lemmas.add(tagged_token.corenlp_lemma)

    wvm = Word2VecModel.load_google_model()
    missing_words = wvm.collect_missing(all_lemmas)
    with open(streusle.ENHANCEMENTS.CORENLP_LEMMAS_WORD2VEC_PATH, 'wb') as f:
        wvm.dump(all_lemmas, f, skip_missing=True)

    with open(streusle.ENHANCEMENTS.CORENLP_LEMMAS_WORD2VEC_MISSING_PATH, 'w') as f:
        json.dump(missing_words, f, indent=2)

    print('Enhanced with word2vec, %d words in total (%d skipped)' % (len(all_lemmas), len(missing_words)))

def apply_spacy_pipeline(tokens):
    doc = Doc(nlp.vocab, words=tokens)
    for name, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc


TreeNode = namedtuple('TreeNode', ['id', 'head_id', 'dep'])


def enhance_spacy_dependency_trees():
    trees = {}
    for ind, rec in enumerate(records):
        doc = apply_spacy_pipeline([tt.token for tt in rec.tagged_tokens])
        assert len(doc) == len(rec.tagged_tokens)
        trees[rec.id] = {
            streusle_token.ud_id: TreeNode(id=streusle_token.ud_id, head_id=rec.tagged_tokens[spacy_token.head.i].ud_id, dep=spacy_token.dep_)
            for spacy_token, streusle_token in zip(doc, rec.tagged_tokens)
        }
        print('enhance_dependency_trees: %d/%d' % (ind + 1, len(records)))
    with open(streusle.ENHANCEMENTS.SPACY_DEP_TREES, 'w') as f:
        json.dump(trees, f, indent=2)
    print('Enhanced with spacy dep trees, %d trees in total' % (len(trees)))


def enhance_spacy_ners():
    def process(t):
        ind, rec = t
        doc = apply_spacy_pipeline([tt.token for tt in rec.tagged_tokens])
        print('enhance ners: %d/%d' % (ind + 1, len(records)))
        return [
            token.ent_type_ or None
            for token in doc
        ]

    with ThreadPoolExecutor(1) as tpe:
        ners_list = list(tpe.map(process, enumerate(records)))
        ners = {rec.id: {tok.ud_id: ner for tok, ner in zip(rec.tagged_tokens, ners)} for rec, ners in zip(records, ners_list) }
        with open(streusle.ENHANCEMENTS.SPACY_NERS, 'w') as f:
            json.dump(ners, f, indent=2)

def enhance_spacy_lemmas():
    def process(t):
        ind, rec = t
        doc = apply_spacy_pipeline([tt.token for tt in rec.tagged_tokens])
        print('enhance lemmas: %d/%d' % (ind + 1, len(records)))
        return [
            token.lemma_ or None
            for token in doc
        ]

    with ThreadPoolExecutor(1) as tpe:
        lemmas_list = list(tpe.map(process, enumerate(records)))
        lemmas = {rec.id: {tok.ud_id: lemma for tok, lemma in zip(rec.tagged_tokens, rec_lemmas)} for rec, rec_lemmas in zip(records, lemmas_list) }
        with open(streusle.ENHANCEMENTS.SPACY_LEMMAS, 'w') as f:
            json.dump(lemmas, f, indent=2)

def enhance_spacy_pos():
    def process(t):
        ind, rec = t
        doc = apply_spacy_pipeline([tt.token for tt in rec.tagged_tokens])
        print('enhance spacy pos: %d/%d' % (ind + 1, len(records)))
        return [
            token.pos_ or None
            for token in doc
        ]


    with ThreadPoolExecutor(1) as tpe:
        pos_list = list(tpe.map(process, enumerate(records)))
        poses = {rec.id: {tok.ud_id: pos for tok, pos in zip(rec.tagged_tokens, rec_poses)} for rec, rec_poses in zip(records, pos_list)}
        with open(streusle.ENHANCEMENTS.SPACY_POS, 'w') as f:
            json.dump(poses, f, indent=2)


def enhance_dev_sentences_old():
    def prod(vec1, vec2):
        return dot(vec1, vec2)

    def dist_score(dist1_vec, dist2_vec):
        return prod(dist1_vec, dist2_vec)/norm(dist1_vec)/norm(dist2_vec)

    ORDERED_PSS = list(supersenses.PREPOSITION_SUPERSENSES_SET)
    def get_dist_vec(dist):
        return [dist[pss] for pss in ORDERED_PSS]

    def update_dist(dist, add_record, remove_record):
        dist = dict(dist)
        for tok in add_record.pss_tokens:
            if tok.supersense in dist:
                dist[tok.supersense] += 1
        for tok in remove_record.pss_tokens:
            if tok.supersense in dist:
                dist[tok.supersense] -= 1
        return dist

    # records = train_records + dev_records

    records = train_records + dev_records
    best_split_score = -1
    best_split = None

    for s_ind in range(200):
        cand_dev = random.sample(records, len(test_records))
        cand_train = [x for x in records if x not in cand_dev]
        train_dist = streusle.StreusleLoader.get_dist(cand_train)
        dev_dist = streusle.StreusleLoader.get_dist(cand_dev)
        train_dist_vec = get_dist_vec(train_dist)
        dev_dist_vec = get_dist_vec(dev_dist)
        split_score = prod(train_dist_vec, dev_dist_vec)/norm(train_dist_vec)/norm(dev_dist_vec)
        if best_split is None or split_score > best_split_score:
            best_split = (cand_train, cand_dev)
            best_split_score = split_score
        print('sample:', s_ind, split_score)

    print('stage 1: best_score', best_split_score)

    cand_train, cand_dev = best_split
    # cand_dev = random.sample(records, len(test_records))
    # cand_train = [x for x in records if x not in cand_dev]

    for s_ind in range(100):
        dev_dist = streusle.StreusleLoader.get_dist(cand_dev)
        train_dist = streusle.StreusleLoader.get_dist(cand_train)
        dev_dist_vec = get_dist_vec(dev_dist)
        train_dist_vec = get_dist_vec(train_dist)
        current_score = dist_score(train_dist_vec, dev_dist_vec)
        print('before switch %d:' % s_ind, current_score)
        best_switch_score = -1
        best_switch = None
        for dind, dev_rec in enumerate(cand_dev):
            for train_rec in cand_train:
                sw_train_dist = update_dist(train_dist, dev_rec, train_rec)
                sw_dev_dist = update_dist(dev_dist, train_rec, dev_rec)
                sw_train_dist_vec = get_dist_vec(sw_train_dist)
                sw_dev_dist_vec = get_dist_vec(sw_dev_dist)
                switch_score = dist_score(sw_dev_dist_vec, sw_train_dist_vec)
                if (best_switch is None or switch_score > best_switch_score) and switch_score > current_score:
                    best_switch = (train_rec, dev_rec)
                    best_switch_score = switch_score
        if best_switch is None:
            print('All switches decrease score, breaking early')
            break
        train_rec, dev_rec = best_switch
        cand_train = [x for x in cand_train if x != train_rec] + [dev_rec]
        cand_dev = [x for x in cand_dev if x != dev_rec] + [train_rec]

    print("stage 2: score %1.4f" % dist_score(get_dist_vec(streusle.StreusleLoader.get_dist(cand_train)),
                                              get_dist_vec(streusle.StreusleLoader.get_dist(cand_dev))))

    with open(streusle.ENHANCEMENTS.DEV_SET_SENTIDS, 'w') as f:
        f.write("\n".join([r.id for r in cand_dev]))

    loader.dump_split_dist('/tmp/split.csv')


def enhance_ud_dependency_trees():
    UD_BASE = os.environ.get('UD_BASE') or '/cs/labs/oabend/aviramstern/ud/UD_English'
    UD_FILES = [
        UD_BASE + '/en-ud-dev.conllu',
        UD_BASE + '/en-ud-test.conllu',
        UD_BASE + '/en-ud-train.conllu'
    ]
    sents = {}
    for f_path in UD_FILES:
        with open(f_path, 'r', encoding='utf8') as f:
            sents.update(conllu_parser.parse(f.read()))

    trees = {}
    ID_PATTERN = re.compile(r'# sent_id \= reviews-(\d+)-0+(\d+)')
    for ind, (sent_id, sent) in enumerate(sents.items()):
        if "363685-0017" in sent_id:
            print(sent_id)
        match = ID_PATTERN.match(sent_id)
        if match:
            assert(all([tok['id'] is not None for tok in sent]))
            streusle_id = "ewtb.r." + match.group(1) + '.' + match.group(2)
            trees[streusle_id] = {
                token['id']: TreeNode(id=token['id'], head_id=token['head'] if token['head'] else token['id'], dep=token['deprel'])
                for token in sent
            }
        print('enhance_dependency_trees: %d/%d' % (ind + 1, len(sents)))
    with open(streusle.ENHANCEMENTS.UD_DEP_TREES, 'w') as f:
        json.dump(trees, f, indent=2)
    print('Enhanced with spacy dep trees, %d trees in total' % (len(trees)))

def enhance_corenlp_dependency_trees():
    sents = parse_conll(streusle.ENHANCEMENTS.STANFORD_CORE_NLP_OUTPUT)

    trees = {}
    for ind, sent in enumerate(sents):
        assert(all([tok['id'] is not None for tok in sent['tokens']]))
        streusle_id = sent['metadata']['streusle_sent_id']
        trees[streusle_id] = {
            token['id']: TreeNode(id=token['id'], head_id=token['head'] if token['head'] else token['id'], dep=token['deprel'])
            for token in sent['tokens']
        }
        print('enhance_dependency_trees: %d/%d' % (ind + 1, len(sents)))
    with open(streusle.ENHANCEMENTS.CORENLP_DEP_TREES, 'w') as f:
        json.dump(trees, f, indent=2)
    print('Enhanced with corenlp dep trees, %d trees in total' % (len(trees)))

def enhance_corenlp_lemmas():
    sents = parse_conll(streusle.ENHANCEMENTS.STANFORD_CORE_NLP_OUTPUT)
    lemmas = {
        sent['metadata']['streusle_sent_id']: {tok['id']: tok['lemma'] for tok in sent['tokens']} for sent in sents
    }
    with open(streusle.ENHANCEMENTS.CORENLP_LEMMAS, 'w') as f:
        json.dump(lemmas, f, indent=2)

def enhance_corenlp_ners():
    sents = parse_conll(streusle.ENHANCEMENTS.STANFORD_CORE_NLP_OUTPUT)
    lemmas = {
        sent['metadata']['streusle_sent_id']: {tok['id']: tok['ner'] for tok in sent['tokens']} for sent in sents
    }
    with open(streusle.ENHANCEMENTS.CORENLP_NERS, 'w') as f:
        json.dump(lemmas, f, indent=2)

def enhance_corenlp_pos():
    sents = parse_conll(streusle.ENHANCEMENTS.STANFORD_CORE_NLP_OUTPUT)
    lemmas = {
        sent['metadata']['streusle_sent_id']: {tok['id']: tok['pos'] for tok in sent['tokens']} for sent in sents
    }
    with open(streusle.ENHANCEMENTS.CORENLP_POS, 'w') as f:
        json.dump(lemmas, f, indent=2)

def enhance_pss_autoid():
    with open('./streusle_4alpha/streusle_autoid.conllulex', 'w') as f:
        subprocess.run(['python', './streusle_4alpha/identify.py', './streusle_4alpha/streusle.conllulex', '-m'], stdout=f)


def enhance_stanford_core_nlp():
    with open('./streusle_4alpha/streusle.conllulex', 'r') as f:
        lines = f.readlines()
        texts = [x.replace('# text = ', '').strip() for x in lines if x.startswith('# text = ')]
        sent_ids = [x.replace('# sent_id = ', '').strip() for x in lines if x.startswith('# sent_id = ')]
        streusle_sent_ids = [x.replace('# streusle_sent_id = ', '').strip() for x in lines if x.startswith('# streusle_sent_id = ')]
        mwes = [x.replace('# mwe = ', '').strip() for x in lines if x.startswith('# mwe = ')]

    input_files = []
    for streusle_id in streusle_sent_ids:
        inp_file = '/tmp/input_' + streusle_id + '.txt'
        with open(inp_file, 'w') as f:
            rec = id_to_record[streusle_id]
            assert not any([t for t in rec.tagged_tokens if " " in t.token])
            assert not any([t for t in rec.tagged_tokens if "." in str(t.ud_id)])
            f.write(' '.join([t.token for t in rec.tagged_tokens]))
        input_files.append(inp_file)

    with open('/tmp/input_files.txt', 'w') as f:
        f.write('\n'.join(input_files))

    os.chdir('/tmp')
    # os.system('java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat conll -filelist /tmp/input_files.txt -depparse BasicDependenciesAnnotation -ssplit.isOneSentence true -tokenize.whitespace true')

    outs = []
    for input_file in input_files:
        with open(input_file + '.conll', 'r') as f:
            s = f.read().split('\n')
        outs.append(s)

    with open(streusle.ENHANCEMENTS.STANFORD_CORE_NLP_OUTPUT, 'w') as f:
        for text, sent_id, streusle_sent_id, mwe, out in zip(texts, sent_ids, streusle_sent_ids, mwes, outs):
            lines = [
                        '# sent_id = ' + sent_id,
                        '# text = ' + text,
                        '# streusle_sent_id = ' + streusle_sent_id,
                        '# mwe = ' + mwe
                    ] + out
            f.write('\n'.join(lines))

if __name__ == '__main__':
    # enhance_spacy_dependency_trees()
    # enhance_spacy_ners()
    # enhance_spacy_pos()
    # # enhance_dev_sentences()
    # enhance_ud_dependency_trees()
    # enhance_spacy_lemmas()
    # # enhance_pss_autoid()
    # # enhance_stanford_core_nlp()
    # enhance_corenlp_ners()
    # enhance_corenlp_pos()
    # enhance_corenlp_lemmas()
    # enhance_corenlp_dependency_trees()

    enhance_word2vec()
    enhance_spacy_lemmas_word2vec()
    enhance_ud_lemmas_word2vec()
    enhance_corenlp_lemmas_word2vec()

