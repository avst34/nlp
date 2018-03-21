from tempfile import NamedTemporaryFile
import os
import requests


def run_corenlp(tokens, format='conllu', use_server=True):
    assert format in ['conllu', 'conll', 'json']
    sentence = ' '.join(tokens)
    if use_server:
        req = {
            'params': {
                'outputFormat': format,
                'ssplit.isOneSentence': 'true',
                'tokenize.whitespace': 'true',
                'annotators': "tokenize,ssplit,pos,lemma,ner,parse,dcoref,udfeats"
            },
            'data': sentence
        }
        r = requests.post('http://127.0.0.1:9000/', params=req['params'], data=req['data'].encode('utf-8'))
        out = r.text.replace('\r\n', '\n')
    else:
        input_file = NamedTemporaryFile(delete=False)
        try:
            input_file.write(sentence)
            input_file.close()
            os.system('java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat ' + format + ' -filelist ' + input_file.name + ' -depparse BasicDependenciesAnnotation -ssplit.isOneSentence true -tokenize.whitespace true')
            with open(input_file + '.' + format, 'r') as f:
                out = f.read()
        finally:
            os.unlink(input_file.name)

    return out