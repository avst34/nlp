import subprocess

from io import StringIO
from tempfile import NamedTemporaryFile

import os


def enrich_autoid(conllulex_train_path, conllulex_to_enrich_path, identify_script_path, out_path):
    tempf = NamedTemporaryFile(delete=False)
    try:
        subprocess.run(['python', identify_script_path, conllulex_to_enrich_path, '-f', conllulex_train_path, '-m'], stdout=tempf)
        tempf.close()
        with open(tempf.name, 'r') as tempf_r:
            output = tempf_r.read()
        with open(out_path, 'w') as f:
            f.write(output)
    finally:
        os.unlink(tempf.name)
