import subprocess

from io import StringIO
from tempfile import NamedTemporaryFile

import os

out_model_path = os.path.dirname(__file__) + '/idmodel'
conllulex_train = os.path.dirname(__file__) + '/../release/train/streusle.ud_train.conllulex'
identify_script_path = os.path.dirname(__file__) + '/../release/identify.py'

def train_autoid(conllulex_train_path=conllulex_train, identify_script_path=identify_script_path, out_model_path=out_model_path, eval=True):
    subprocess.run(['python', identify_script_path, conllulex_train_path+'.temp', '-f', conllulex_train_path, '-m', '-L', '-o', out_model_path]).check_returncode()
    if eval:
        with open(out_model_path + '.autoid_eval', 'w') as f:
            subprocess.run(['python', identify_script_path, conllulex_train_path+'.temp', '-f', conllulex_train_path, '-m', '-L', '-o', out_model_path, '-e']).check_returncode()

train_autoid()

def enrich_autoid(out_path, conllulex_to_enrich_path, identify_script_path=identify_script_path):
    tempf = NamedTemporaryFile(delete=False)
    try:
        subprocess.run(['python', identify_script_path, conllulex_to_enrich_path, '-M', out_model_path, '-m', '-L'], stdout=tempf).check_returncode()
        if eval:
            with open(out_path + '.autoid_eval', 'w') as f:
                subprocess.run(['python', identify_script_path, conllulex_to_enrich_path, '-M', out_model_path, '-m', '-e', '-L'], stdout=f).check_returncode()
        tempf.close()
        with open(tempf.name, 'r', encoding='utf8') as tempf_r:
            output = tempf_r.read()
        with open(out_path, 'w', encoding='utf8') as f:
            f.write(output)
    finally:
        try:
            os.unlink(tempf.name)
        finally:
            print("WARNING: unable to delete tempfile:", tempf.name)
