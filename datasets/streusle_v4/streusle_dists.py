from datasets.streusle_v4 import StreusleLoader
import os
from collections import Counter
from pprint import pprint

loader = StreusleLoader()
STREUSLE_BASE = os.path.join(os.path.dirname(__file__), 'release')
task ='goldid.goldsyn'
train_records = loader.load(STREUSLE_BASE + '/train/streusle.ud_train.' + task + '.json', input_format='json')
dev_records = loader.load(STREUSLE_BASE + '/dev/streusle.ud_dev.' + task + '.json', input_format='json')
test_records = loader.load(STREUSLE_BASE + '/test/streusle.ud_test.' + task + '.json', input_format='json')
records = train_records + dev_records + test_records

samples = [] # (prep, role, func)

for rec in records:
    for ttok in rec.tagged_tokens:
        if ttok.supersense_role:
            assert ttok.supersense_func
            prep = ' '.join([rec.get_tok_by_ud_id(udid).token for udid in ttok.we_toknums]).lower()
            samples.append((prep, ttok.supersense_role, ttok.supersense_func))

prep_dist = Counter([x[0] for x in samples])
role_dist = Counter([x[1] for x in samples])
func_dist = Counter([x[2] for x in samples])
pair_dist = Counter([(x[1], x[2]) for x in samples])

print('Prepositions distribution:')
pprint(prep_dist)
print('')
print('Role distribution:')
pprint(role_dist)
print('')
print('Func distribution:')
pprint(func_dist)
print('')
print('Pair distribution:')
pprint(pair_dist)
print('')
role_eq_func = len([x for x in samples if x[1] == x[2]])
print('ROLE == FUNC: %d/%d (%%%2.2f)' % (role_eq_func, len(samples), role_eq_func/len(samples)*100))