import os
import subprocess
import time
from tempfile import NamedTemporaryFile

import requests

CORENLP_SERVER_PORT = 9000

class CoreNLPServer(object):
    def __init__(self):
        self.handle = None

    def start(self, port=CORENLP_SERVER_PORT):
        corenlp_home = os.path.dirname(__file__) + '/../../../corenlp/stanford-corenlp-full-2017-06-09'
        files = [x for x in os.listdir(corenlp_home) if os.path.isfile(x)]
        self.handle = subprocess.Popen(
            args=('java -mx4g -cp * edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port ' + str(port) + ' -timeout 15000').split(),
            cwd=corenlp_home
        )
        time.sleep(1)
        if self.handle.poll() is not None:
            raise Exception("Error starting CoreNLP server")

    def stop(self):
        self.handle.terminate()


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
        r = requests.post('http://127.0.0.1:%d/' % CORENLP_SERVER_PORT, params=req['params'], data=req['data'].encode('utf-8'))
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