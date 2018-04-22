import os
import subprocess
from datasets.pp_attachement.boknilev.load_boknilev import load_boknilev

def convert_sample_pp_to_tsv_row(sample, pp):
    cols = [
        pp['head_ind'] + 1,
    ]
    cols.append(' '.join([tok + '_' + pos for tok, pos in zip(sample['tokens'], sample['preprocessing']['ud_xpos'])]))
    return '\t'.join([str(x) for x in cols])

def convert_sample_to_tsv_rows(sample):
    return [convert_sample_pp_to_tsv_row(sample, pp) for pp in sample['pps']]

def convert_samples_to_tsv_rows(samples):
    return sum([convert_sample_to_tsv_rows(s) for s in samples], [])

def convert_samples_to_tsv(samples):
    return '\n'.join(convert_samples_to_tsv_rows(samples))

def save_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def train_with_boknilev():
    train, dev, test = load_boknilev()
    train_tsv, dev_tsv, test_tsv = convert_samples_to_tsv(train), convert_samples_to_tsv(dev), convert_samples_to_tsv(test)
    train_path = os.path.dirname(__file__) + '/data/train.tsv'
    dev_path = os.path.dirname(__file__) + '/data/dev.tsv'
    test_path = os.path.dirname(__file__) + '/data/test.tsv'
    save_file(train_path, train_tsv)
    save_file(dev_path, dev_tsv)
    save_file(test_path, test_tsv)

    PYTHON_PATH = os.environ.get('PYTHON_PATH') or r'c:\anaconda3\envs\tensorflow\python'

    output = subprocess.check_output([PYTHON_PATH,
                                      'model_pp_attachment.py',
                                      '--train_file',
                                      train_path,
                                      '--test_file',
                                      dev_path,
                                      '--embedding_file',
                                      'autoextend_wn3_glove.100d.txt.gz',
                                      '--embed_dim',
                                      '100',
                                      '--tune_embedding',
                                      '--bidirectional',
                                      '--onto_aware',
                                      '--use_attention',
                                      '--num_senses',
                                      '3',
                                      '--num_hyps',
                                      '5',
                                      '--embedding_dropout',
                                      '0.5',
                                      '--encoder_dropout',
                                      '0.2'])

if __name__ == '__main__':
    train_with_boknilev()




