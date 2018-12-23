import os
import subprocess
from tempfile import NamedTemporaryFile

out_model_path = os.path.dirname(__file__) + '/idmodel'
conllulex_train = os.path.dirname(__file__) + '/../release/train/streusle.ud_train.conllulex'
identify_script_path = os.path.dirname(__file__) + '/../release/identify.py'

def train_autoid(conllulex_train_path=conllulex_train, identify_script_path=identify_script_path, out_model_path=out_model_path, eval=True):
    with open(conllulex_train_path+'.temp','w') as f_temp:
        with open(conllulex_train_path,'r') as f:
            f_temp.write(f.read())
    subprocess.run(['python', identify_script_path, conllulex_train_path+'.temp', '-f', conllulex_train_path, '-m', '-L', '-o', out_model_path], stdout=subprocess.DEVNULL).check_returncode()
    if eval:
        with open(out_model_path + '.autoid_eval', 'w') as f:
            subprocess.run(['python', identify_script_path, conllulex_train_path+'.temp', '-f', conllulex_train_path, '-m', '-L', '-o', out_model_path, '-e'], stdout=f).check_returncode()

train_autoid()

def enrich_autoid(out_path, conllulex_to_enrich_path, identify_script_path=identify_script_path, eval=False):
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

        wes = {}
        smwes = {}
        lines = [l.split('\t') for l in output.strip().split('\n')]
        for ind, l in enumerate(lines):
            cols = l
            if cols[-2] == '*':
                wes[cols[0]] = {
                        "lexcat": cols[-1],
                        "lexlemma": cols[2],
                        "toknums": [
                            int(cols[0])
                        ]
                    }
            elif cols[-2].endswith('**'):
                id = cols[-2][cols[-2].index(':')]
                toks = [cols[0]]
                lemmas = [cols[2]]
                i = ind + 1
                while ':' in lines[i][-1]:
                    toks.append(lines[i][0])
                    lemmas.append(lines[i][2])
                    i += 1
                smwes[id] = {
                    "toknums": [int(i) for i in toks],
                    "lexcat": cols[-1],
                    "lexlemma": " ".join(toks)
                }
        return {
            "swes": wes,
            "smwes": smwes,
            "wmwes": {}
        }
    finally:
        try:
            os.unlink(tempf.name)
        finally:
            print("WARNING: unable to delete tempfile:", tempf.name)
