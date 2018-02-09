import os
from concurrent.futures import ThreadPoolExecutor

import requests

from datasets.streusle_v4.streusle_4alpha import conllulex2json

cache = {}

def run_corenlp_on_conllulex(conllulex_fpath, format='conllu', tmp_dir='/tmp', use_server=True):
    assert format in ['conllu', 'conll']
    with open(conllulex_fpath, 'r', encoding='utf-8') as f:
        sents = list(conllulex2json.load_sents(f, input_type='conllulex'))

    input_files = []
    input_files = []
    for sent in sents:
        inp_file = tmp_dir + '/input_' + sent['streusle_sent_id'] + '.txt'
        with open(inp_file, 'w', encoding='utf-8') as f:
            f.write(' '.join([t['word'] for t in sent['toks']]))
        input_files.append(inp_file)

    with open(tmp_dir + '/input_files.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(input_files))

    outs = []
    if use_server:
        # cache = {}
        def process(input_file):
            if not cache.get((input_file, format)):
                with open(input_file, 'r', encoding='utf-8') as f:
                    s = f.read()
                    r = requests.post('http://127.0.0.1:9000/', params={
                        'outputFormat': format,
                        'ssplit.isOneSentence': 'true',
                        'tokenize.whitespace': 'true',
                        'annotators': "tokenize,ssplit,pos,lemma,ner,parse,dcoref,udfeats"
                    }, data=s.encode('utf-8'))
                    try:
                        int(r.text.split('\t')[0])
                    except:
                        print("ERROR: Bad response from corenlp server")
                        print(r.text)
                        print(s)
                        raise
                    cache[(input_file, format)] = r.text.replace('\r\n', '\n').split('\n')
            print("%d/%d" % (len(cache), len(input_files)))
        with ThreadPoolExecutor(6) as tpe:
            list(tpe.map(process, input_files))
        for input_file in input_files:
            outs.append(cache[(input_file, format)])
    else:
        os.chdir(tmp_dir)
        os.system('java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat ' + format + ' -filelist ' + tmp_dir + '/input_files.txt -depparse BasicDependenciesAnnotation -ssplit.isOneSentence true -tokenize.whitespace true')

        outs = []
        for input_file in input_files:
            with open(input_file + '.' + format, 'r') as f:
                s = f.read().split('\n')
            outs.append(s)

    return outs
