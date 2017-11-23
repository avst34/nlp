import supersenses
from datasets.streusle import streusle
from evaluators.classifier_evaluator import ClassifierEvaluator
from models.general.lstm_mlp_multiclass_model import LstmMlpMulticlassModel
from models.general.simple_conditional_multiclass_model.model import SimpleConditionalMulticlassModel
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.lstm_mlp_supersenses_model_hyperparameters_tuner import \
    LstmMlpSupersensesModelHyperparametersTuner
from vocabulary import Vocabulary


def streusle_record_to_lstm_model_sample(record, ss_types):
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
                token=tagged_token.token,
                pos=tagged_token.pos,
                dep=tagged_token.dep,
                head_ind=tagged_token.head_ind,
            ) for tagged_token in record.tagged_tokens],
        ys=[LstmMlpSupersensesModel.SampleY(
                supersense=tagged_token.supersense \
                    if tagged_token.supersense is None \
                       or supersenses.get_supersense_type(tagged_token.supersense) in ss_types \
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
            tagged_token.supersense \
                if tagged_token.supersense is None \
                   or supersenses.get_supersense_type(tagged_token.supersense) in ss_types \
                else None
            for tagged_token in record.tagged_tokens
        ],
    )

streusle_loader = streusle.StreusleLoader()
train_records, dev_records, test_records = streusle_loader.load()
print('loaded %d train records with %d tokens (%d unique)' % (len(train_records), sum([len(x.tagged_tokens) for x in train_records]),
                                                        len(set([t.token for s in train_records for t in s.tagged_tokens]))))
print('loaded %d test dev records with %d tokens (%d unique)' % (len(dev_records), sum([len(x.tagged_tokens) for x in dev_records]),
                                                        len(set([t.token for s in dev_records for t in s.tagged_tokens]))))
print('loaded %d test records with %d tokens (%d unique)' % (len(test_records), sum([len(x.tagged_tokens) for x in test_records]),
                                                        len(set([t.token for s in test_records for t in s.tagged_tokens]))))

train_samples = [streusle_record_to_conditional_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in train_records]
train_samples = [s for s in train_samples if any(s.ys)]

dev_samples = [streusle_record_to_conditional_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in dev_records]
dev_samples = [s for s in dev_samples if any(s.ys)]

evaluator = ClassifierEvaluator()

print('Simple conditional model evaluation:')
scm = SimpleConditionalMulticlassModel()
scm.fit(train_samples, validation_samples=dev_samples, evaluator=evaluator)

print('')

train_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in train_records]
train_samples = [s for s in train_samples if any([y.supersense for y in s.ys])]

dev_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in dev_records]
dev_samples = [s for s in dev_samples if any([y.supersense for y in s.ys])]

test_samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in test_records]
test_samples = [s for s in test_samples if any([y.supersense for y in s.ys])]

# print('LSTM-MLP evaluation:')
# lstm_mlp_model = LstmMlpSupersensesModel(
#     token_embd=streusle_loader.get_tokens_word2vec_model().as_dict()
# )
#
# lstm_mlp_model.fit(samples,
#                    epochs=50,
#                    show_progress=True,
#                    show_epoch_eval=True,
#                    validation_split=0.3,
#                    evaluator=evaluator)
#

pp_vocab = Vocabulary('Prepositions')
pp_vocab.add_words(set([x.token for s in train_samples + dev_samples + test_samples for x, y in zip(s.xs, s.ys) if y.supersense]))

tuner = LstmMlpSupersensesModelHyperparametersTuner(
    token_embd=streusle_loader.get_tokens_word2vec_model().as_dict(),
    token_onehot_vocab=pp_vocab
)
tuner.tune(train_samples,
           '/tmp/results.csv',
           validation_samples=dev_samples,
           n_executions=50,
           show_progress=True,
           show_epoch_eval=True)

