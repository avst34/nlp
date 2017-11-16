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
records = streusle_loader.load()
print('loaded %d records with %d tokens (%d unique)' % (len(records), sum([len(x.tagged_tokens) for x in records]),
                                                        len(set([t.token for s in records for t in s.tagged_tokens]))))

samples = [streusle_record_to_conditional_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in records]
samples = [s for s in samples if any(s.ys)]

evaluator = ClassifierEvaluator()

print('Simple conditional model evaluation:')
scm = SimpleConditionalMulticlassModel()
scm.fit(samples, validation_split=0.3, evaluator=evaluator)

print('')

samples = [streusle_record_to_lstm_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in records]
samples = [s for s in samples if any([y.supersense for y in s.ys])]

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

tuner = LstmMlpSupersensesModelHyperparametersTuner(
    token_embd=streusle_loader.get_tokens_word2vec_model().as_dict()
)
tuner.tune(samples,
           '/tmp/results.csv',
           n_executions=50,
           show_progress=True,
           show_epoch_eval=True)

