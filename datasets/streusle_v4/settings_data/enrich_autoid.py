import subprocess

from io import StringIO
from tempfile import NamedTemporaryFile

import os


def enrich_autoid(conllulex_train_path, conllulex_to_enrich_path, identify_script_path, out_path, eval=True):
    tempf = NamedTemporaryFile(delete=False)
    try:
        subprocess.run(['python', identify_script_path, conllulex_to_enrich_path, '-f', conllulex_train_path, '-m'], stdout=tempf)
        if eval:
            with open(out_path + '.autoid_eval', 'w') as f:
                subprocess.run(['python', identify_script_path, conllulex_to_enrich_path, '-f', conllulex_train_path, '-m', '-e'], stdout=f)
        tempf.close()
        with open(tempf.name, 'r', encoding='utf8') as tempf_r:
            output = tempf_r.read()
        with open(out_path, 'w', encoding='utf8') as f:
            f.write(output)
    finally:
        os.unlink(tempf.name)
