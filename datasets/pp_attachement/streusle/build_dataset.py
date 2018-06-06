import os

import glob

# from datasets.pp_attachement.boknilev.build_dataset import collect_pp_annotations
# from datasets.streusle_v4 import StreusleLoader
import json

from datasets.pp_attachement.boknilev.build_dataset import collect_pp_annotations, build_sample, add_preprocessing, \
    preprocess_samples
from datasets.streusle_v4 import StreusleLoader

BASE_PATH = os.path.dirname(__file__)


STREUSLE_BASE = os.environ.get('STREUSLE_BASE') or '/cs/usr/aviramstern/lab/nlp/datasets/streusle_v4/release'

task = 'goldid.autosyn'
loader = StreusleLoader()
train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')

records = train_records + dev_records + test_records
id_to_rec = {rec.id: rec for rec in records}


def build_conllx(base_dir, target_file):
    located = 0
    total = 0
    all_rec_ids = set(id_to_rec)
    with open(target_file, 'w') as out_f:
        for fpath in sorted(glob.glob(base_dir + '/*.conllx')):
            with open(fpath) as in_f:
                print(fpath)
                fname = os.path.basename(fpath).split('.')[0]
                sents = in_f.read().split('\n\n')
                for ind, sent in enumerate(sents):
                    sent_id = 'ewtb.r.%06d.%d' % (int(fname), ind+1)
                    print(fpath, ind, sent_id)
                    out_f.write('#%s\n' % sent_id)
                    out_f.write(sent)
                    out_f.write('\n')
                    if sent != sents[-1]:
                        out_f.write('\n')
                    srec = id_to_rec.get(sent_id)
                    if srec:
                        located += 1
                        all_rec_ids.remove(srec.id)
                        assert len(srec.tokens()) == len([s.split('\t')[1] for s in sent.split('\n')]), '%s != %s' % (repr(srec.tokens()), repr([s.split('\t')[1] for s in sent.split('\n')]))
                    total += 1
    print('located ', located, 'total', total, 'streusle count', len(records))
    print('missing:', all_rec_ids)


def build_dataset(boknilev_input_base_path=BASE_PATH + '/all.dep.pp'):
    annotations = collect_pp_annotations([boknilev_input_base_path], BASE_PATH + '/annotations.json')
    sent_id_to_anns = {}
    for ann in annotations:
        sent_id_to_anns[ann['sentids'][0]] = sent_id_to_anns.get(ann['sentids'][0]) or []
        sent_id_to_anns[ann['sentids'][0]].append(ann)

    for out_path, s_records in [('train.json', train_records), ('dev.json', dev_records), ('test.json', test_records)]:
        samples = []
        for rec in s_records:
            if not sent_id_to_anns.get(rec.id):
                continue
            sent = {
                'sent': [tok.token for tok in rec.tagged_tokens],
                'id': rec.id
            }
            sample = build_sample(
                sent,
                sent_id_to_anns[rec.id]
            )
            sample['preprocessing'] = {
                'gold_pss_role': [
                    tok.supersense_role for tok in rec.tagged_tokens
                ],
                'gold_pss_func': [
                    tok.supersense_func for tok in rec.tagged_tokens
                ],
                'gold_noun_ss': [
                    tok.noun_ss for tok in rec.tagged_tokens
                ],
                'gold_verb_ss': [
                    tok.verb_ss for tok in rec.tagged_tokens
                ],
            }
            samples.append(sample)
        preprocess_samples(samples)

        with open(BASE_PATH + '/' + out_path, 'w') as out_f:
            json.dump(samples, out_f, indent=2)


if __name__ == '__main__':
    build_conllx('/cs/usr/aviramstern/lab/eng_web_tbk/data/reviews/penntree', BASE_PATH + '/all.conllx')
    build_dataset()