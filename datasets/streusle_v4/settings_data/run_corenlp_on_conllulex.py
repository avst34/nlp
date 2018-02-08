import os

from datasets.streusle_v4.streusle_4alpha import conllulex2json


def run_corenlp_on_conllulex(conllulex_fpath, format='conllu', tmp_dir='/tmp'):
    assert format in ['conllu', 'conll']
    with open(conllulex_fpath, 'r') as f:
        sents = list(conllulex2json.load_sents(f, input_type='conllulex'))

    input_files = []
    for sent in sents:
        inp_file = tmp_dir + '/input_' + sent['streusle_sent_id'] + '.txt'
        with open(inp_file, 'w') as f:
            f.write(' '.join([t['word'] for t in sent['toks']]))
        input_files.append(inp_file)

    with open(tmp_dir + '/input_files.txt', 'w') as f:
        f.write('\n'.join(input_files))

    os.chdir(tmp_dir)
    os.system('java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat ' + format + ' -filelist ' + tmp_dir + '/input_files.txt -depparse BasicDependenciesAnnotation -ssplit.isOneSentence true -tokenize.whitespace true')

    outs = []
    for input_file in input_files:
        with open(input_file + '.' + format, 'r') as f:
            s = f.read().split('\n')
        outs.append(s)

    return outs
