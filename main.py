import random
import dynet_config
import os, sys

print(os.environ)
print(sys.path)

os.environ['DYNET_RANDOM_SEED'] = str(random.randrange(10000000))
dynet_config.set(random_seed=int(os.environ['DYNET_RANDOM_SEED']))
import dynet

from run import most_frequent_baseline, lstm_mlp_baseline, dataset_statistics
from collections import Counter
from datasets.streusle_v4 import streusle


# streusle_loader = streusle.StreusleLoader()
# train_records, dev_records, test_records = streusle_loader.load()
# print('loaded %d train records with %d tokens (%d unique), %d prepositions' % (len(train_records),
#                                                        sum([len(x.tagged_tokens) for x in train_records]),
#                                                        len(set([t.token for s in train_records for t in s.tagged_tokens])),
#                                                        len([tok for rec in train_records for tok in rec.tagged_tokens if tok.supersense_combined])))
# print('loaded %d dev records with %d tokens (%d unique), %d prepositions' % (len(dev_records),
#                                                        sum([len(x.tagged_tokens) for x in dev_records]),
#                                                        len(set([t.token for s in dev_records for t in s.tagged_tokens])),
#                                                        len([tok for rec in dev_records for tok in rec.tagged_tokens if tok.supersense_combined])))
# print('loaded %d test records with %d tokens (%d unique), %d prepositions' % (len(test_records),
#                                                        sum([len(x.tagged_tokens) for x in test_records]),
#                                                        len(set([t.token for s in test_records for t in s.tagged_tokens])),
#                                                        len([tok for rec in test_records for tok in rec.tagged_tokens if tok.supersense_combined])))
#
#
# all_records = train_records + dev_records + test_records
# all_ignored_ss = [ignored_ss for rec in all_records for ignored_ss in rec.ignored_supersenses]
# unfamiliar_ss = [ss for ss in all_ignored_ss if not supersenses.filter_non_supersense(ss)]
# unfamiliar_ss = [ss for ss in unfamiliar_ss if not(ss.startswith('`') or '_' in ss or '?' in ss)]
# print('Ignored %d supersenses, %d out of them are unfamiliar:' % (len(set(all_ignored_ss)), len(set(unfamiliar_ss))))
# for ss in sorted(set(unfamiliar_ss)):
#     print("%s (%d appearances)" % (ss, unfamiliar_ss.count(ss)))
#
# unfamiliar_ss_after_splitting = [_ss for ss in unfamiliar_ss for __ss in ss.split('|') for _ss in __ss.split(' ') if not supersenses.filter_non_supersense(_ss)]
# print('And after splitting:')
# for ss in sorted(set(unfamiliar_ss_after_splitting)):
#     print("%s (%d appearances)" % (ss, unfamiliar_ss_after_splitting.count(ss)))
#
# all_prepositions = set([t.token.lower() for rec in all_records for t in rec.tagged_tokens if t.supersense_combined])
# print("All prepositions:", len(all_prepositions))
# print("----------------")
# for p in sorted(all_prepositions):
#     print(p)
# print("---")
#
# all_mwe_prepositions = [t.token.lower() for rec in all_records for t in rec.tagged_tokens if t.supersense_combined and t.is_part_of_mwe]
# print("All mwe prepositions:", len(set(all_mwe_prepositions)))
# print("----------------")
# for p in sorted(set(all_mwe_prepositions)):
#     print(p)
# print("---")
#
#
# print("Preposition POSes:", Counter([t.pos for rec in all_records for t in rec.tagged_tokens if t.supersense_combined]))
# print("Preposition POSes (MWEs dropped):", Counter([t.pos for rec in all_records for t in rec.tagged_tokens if t.supersense_combined and not t.is_part_of_mwe]))

# dataset_statistics.run(all_records)
# most_frequent_baseline.run(train_records, dev_records)
lstm_mlp_baseline.run()
