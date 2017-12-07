from collections import Counter

import supersenses
import json
from datasets.streusle import streusle
from evaluators.classifier_evaluator import ClassifierEvaluator
from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from models.general.simple_conditional_multiclass_model.model import SimpleConditionalMulticlassModel
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from models.supersenses.tuner_domains import PS
from vocabulary import Vocabulary


def streusle_record_to_lstm_model_sample(record,
                                         ss_types,
                                         deps_from # 'ud' or 'spacy'
                                         ):
    assert deps_from in ['ud', 'spacy']
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
                token=tagged_token.token,
                pos=tagged_token.pos,
                dep=tagged_token.spacy_dep if deps_from == 'spacy' else tagged_token.ud_dep,
                head_ind=tagged_token.spacy_head_ind if deps_from == 'spacy' else tagged_token.ud_head_ind,
            ) for tagged_token in record.tagged_tokens],
        ys=[LstmMlpSupersensesModel.SampleY(
                supersense_role=tagged_token.supersense_role \
                    if tagged_token.supersense_role is None \
                       or supersenses.get_supersense_type(tagged_token.supersense_role) in ss_types \
                    else None,
                supersense_func=tagged_token.supersense_func \
                    if tagged_token.supersense_func is None \
                       or supersenses.get_supersense_type(tagged_token.supersense_func) in ss_types \
                    else None
            ) for tagged_token in record.tagged_tokens
         ],
    )

def streusle_record_to_conditional_model_sample(record, ss_types):
    return SimpleConditionalMulticlassModel.Sample(
        xs=[{
            "token": tagged_token.token,
            "pos": tagged_token.pos
        } for tagged_token in record.tagged_tokens],
        ys=[
            tagged_token.supersense_role \
                if tagged_token.supersense_role is None \
                   or supersenses.get_supersense_type(tagged_token.supersense_role) in ss_types \
                else None
            for tagged_token in record.tagged_tokens
        ],
    )

streusle_loader = streusle.StreusleLoader()
train_records, dev_records, test_records = streusle_loader.load()
print('loaded %d train records with %d tokens (%d unique), %d prepositions' % (len(train_records),
                                                       sum([len(x.tagged_tokens) for x in train_records]),
                                                       len(set([t.token for s in train_records for t in s.tagged_tokens])),
                                                       len([tok for rec in train_records for tok in rec.tagged_tokens if tok.combined_supersense])))
print('loaded %d dev records with %d tokens (%d unique), %d prepositions' % (len(dev_records),
                                                       sum([len(x.tagged_tokens) for x in dev_records]),
                                                       len(set([t.token for s in dev_records for t in s.tagged_tokens])),
                                                       len([tok for rec in dev_records for tok in rec.tagged_tokens if tok.combined_supersense])))
print('loaded %d test records with %d tokens (%d unique), %d prepositions' % (len(test_records),
                                                       sum([len(x.tagged_tokens) for x in test_records]),
                                                       len(set([t.token for s in test_records for t in s.tagged_tokens])),
                                                       len([tok for rec in test_records for tok in rec.tagged_tokens if tok.combined_supersense])))

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
# all_prepositions = set([t.token.lower() for rec in all_records for t in rec.tagged_tokens if t.combined_supersense])
# print("All prepositions:", len(all_prepositions))
# print("----------------")
# for p in sorted(all_prepositions):
#     print(p)
# print("---")
#
# all_mwe_prepositions = [t.token.lower() for rec in all_records for t in rec.tagged_tokens if t.combined_supersense and t.part_of_mwe]
# print("All mwe prepositions:", len(set(all_mwe_prepositions)))
# print("----------------")
# for p in sorted(set(all_mwe_prepositions)):
#     print(p)
# print("---")
#
#
# print("Preposition POSes:", Counter([t.pos for rec in all_records for t in rec.tagged_tokens if t.combined_supersense]))
# print("Preposition POSes (MWEs dropped):", Counter([t.pos for rec in all_records for t in rec.tagged_tokens if t.combined_supersense and not t.part_of_mwe]))

train_samples = [streusle_record_to_conditional_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in train_records]
train_samples = [s for s in train_samples if any(s.ys)]

dev_samples = [streusle_record_to_conditional_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in dev_records]
dev_samples = [s for s in dev_samples if any(s.ys)]

evaluator = ClassifierEvaluator()

if False:
    print('Simple conditional model evaluation:')
    scm = SimpleConditionalMulticlassModel()
    scm.fit(train_samples, validation_samples=dev_samples, evaluator=evaluator)

deps_from='ud'
train_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE], deps_from=deps_from) for r in train_records]
dev_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE], deps_from=deps_from) for r in dev_records]
test_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE], deps_from=deps_from) for r in test_records]

pp_vocab = Vocabulary('Prepositions')
pp_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys) if any([y.supersense_role, y.supersense_func])]))

dep_vocab = Vocabulary('Dependencies')
dep_vocab.add_words(set([x.dep for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

pos_vocab = Vocabulary('POS')
pos_vocab.add_words(set([x.pos for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

token_vocab = Vocabulary('Tokens')
token_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys)]))

pss_vocab = Vocabulary('PrepositionalSupersenses')
pss_vocab.add_words(supersenses.PREPOSITION_SUPERSENSES_SET)
pss_vocab.add_word(None)

tuner = LstmMlpSupersensesModelHyperparametersTuner(
    token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
    token_vocab=token_vocab,
    token_onehot_vocab=pp_vocab,
    pos_vocab=pos_vocab,
    dep_vocab=dep_vocab,
    supersense_vocab=pss_vocab,

)
tuner.tune(train_samples,
           # '/cs/labs/oabend/aviramstern/results.csv',
           '/tmp/results.csv',
           validation_samples=dev_samples,
           n_executions=1,
           show_progress=True,
           show_epoch_eval=True,
           tuner_domains_override=[
               PS(name='labels_to_predict', values=[
                   ('supersense_role', 'supersense_func'),
               ]),
               # PS(name='mask_by', values=['pos:IN,PRP$,RB,TO']),
               PS(name='mask_by', values=['sample-ys']),
               PS(name='learning_rate', values=[0.1]),
               PS(name='learning_rate_decay', values=[0.01]),
               PS(name='mlp_dropout_p', values=[0.1])
           ])

# print('LSTM-MLP evaluation:')
# lstm_mlp_model = LstmMlpSupersensesModel(
#     token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
#     token_vocab=token_vocab,
#     token_onehot_vocab=pp_vocab,
#     pos_vocab=pos_vocab,
#     dep_vocab=dep_vocab,
#     supersense_vocab=pss_vocab,
#     hyperparameters=LstmMlpSupersensesModel.HyperParameters(**json.loads("""{"use_head": true, "lstm_h_dim": 74, "epochs": 100, "mlp_layer_dim": 62, "mlp_layers": 2, "num_lstm_layers": 2, "use_token_onehot": true, "update_token_embd": false, "learning_rate": 0.1, "use_token": true, "use_pos": false, "token_internal_embd_dim": 27, "use_token_internal": false, "use_dep": true, "validation_split": 0.3, "token_embd_dim": 300, "pos_embd_dim": 80, "mlp_dropout_p": 0.05, "is_bilstm": true, "mlp_activation": "tanh", "update_pos_embd": true, "learning_rate_decay": 0.00031622776601683794}"""))
# )
#
# lstm_mlp_model.fit(train_samples,
#                    dev_samples,
#                    evaluator=evaluator)
#
# evaluator = ClassifierEvaluator(predictor=lstm_mlp_model.model)
# ll_samples = [LstmMlpSupersensesModel.sample_to_lowlevel(x) for x in test_samples]
# evaluator.evaluate(ll_samples, examples_to_show=5)