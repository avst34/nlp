import os
import subprocess

default_script_path = os.path.dirname(__file__) + '/../release/govobj.py'

def enrich_govobj(fpath, govobj_script_path=default_script_path):
    output = subprocess.check_output(['python', govobj_script_path, fpath])
    with open(fpath, 'wb') as f:
        f.write(output)
