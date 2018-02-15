import json

import os
from shutil import copyfile

from datasets.streusle_v4.settings_data.enrich_autoid import enrich_autoid
from datasets.streusle_v4.settings_data.enrich_govobj import enrich_govobj
from datasets.streusle_v4.settings_data.enrich_ners import enrich_ners
from datasets.streusle_v4.settings_data.run_corenlp_on_conllulex import run_corenlp_on_conllulex
from datasets.streusle_v4.settings_data.subsitute_conllulex_corenlp import substitute_conllulex_corenlp
from datasets.streusle_v4.release import conllulex2json


def run_pipeline(conllulex_train_path, conllulex_dev_path, conllulex_test_path, identification, preprocessing, identify_script_path):
    assert identification in ['goldid', 'autoid']
    assert preprocessing in ['goldsyn', 'autosyn']

    out_files = ['.'.join([f.replace('.conllulex', ''), identification, preprocessing]) + '.json' for f in [conllulex_train_path, conllulex_dev_path, conllulex_test_path]]

    for inf_path, outf_path in zip([conllulex_train_path, conllulex_dev_path, conllulex_test_path], out_files):
        copyfile(inf_path, outf_path)

    out_files = {
        'train': out_files[0],
        'dev': out_files[1],
        'test': out_files[2]
    }

    print('run_pipeline: extracting conlls')

    conlls = {}

    for stype, fpath in out_files.items():
        conlls[stype] = run_corenlp_on_conllulex(fpath, format='conll')

    print('run_pipeline: preprocessing')

    if preprocessing == 'autosyn':
        for f in out_files.values():
            enriched = substitute_conllulex_corenlp(f)
            with open(f, 'w', encoding='utf-8') as f:
                f.write(enriched)

    print('run_pipeline: identification')

    if identification == 'autoid':
        for f in out_files.values():
            enrich_autoid(out_files['train'], f, identify_script_path, f)

    print('run_pipeline: converting to json')

    for fpath in out_files.values():
        with open(fpath, 'r', encoding='utf-8') as f:
            sents = list(conllulex2json.load_sents(f, identification=identification, input_type='conllulex'))
        with open(fpath, 'w', encoding='utf-8') as f:
            try:
                json.dump(sents, f)
            except:
                json.dumps(sents)
                raise

    print('run_pipeline: adding gov/obj')

    for fpath in out_files.values():
        enrich_govobj(fpath)

    print('run_pipeline: adding ner')

    for stype, fpath in out_files.items():
        with open(fpath, 'r', encoding='utf-8') as f:
            recs = json.load(f)
        enriched_recs = enrich_ners(recs, conlls[stype])
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(enriched_recs, f, indent=2)


def build_data(conllulex_train_path, conllulex_dev_path, conllulex_test_path, identify_script_path):
    for syn in ['autosyn', 'goldsyn']:
        for id in ['autoid', 'goldid']:
            run_pipeline(conllulex_train_path, conllulex_dev_path, conllulex_test_path, id, syn, identify_script_path)



if __name__ == '__main__':
    STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/nlp/datasets/streusle_v4/release'
    build_data(
        STREUSLE_BASE + '/train/streusle.ud_train.conllulex',
        STREUSLE_BASE + '/dev/streusle.ud_dev.conllulex',
        STREUSLE_BASE + '/test/streusle.ud_test.conllulex',
        STREUSLE_BASE + '/identify.py'
    )


