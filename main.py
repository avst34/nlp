import supersenses
from datasets.streusle import streusle
from evaluators.samples_evaluator import ClassifierEvaluator
from lstm_mlp_multiclass_model.model import LstmMlpMulticlassModel, Sample
from simple_conditional_multiclass_model.model import SimpleConditionalMulticlassModel
from vocabulary import Vocabulary

def streusle_record_to_model_sample(record, ss_types):
    return Sample(
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
print('loaded %d records with %d tokens' % (len(records), sum([len(x.tagged_tokens) for x in records])))

token_vocab = Vocabulary('Tokens')
streusle.wordVocabularyBuilder.feed_records_to_vocab(records, token_vocab)
print(token_vocab)

pos_vocab = Vocabulary('POS')
streusle.posVocabularyBuilder.feed_records_to_vocab(records, pos_vocab)
print(pos_vocab)

ss_vocab = Vocabulary('Supersenses')
streusle.ssVocabularyBuilder.feed_records_to_vocab(records, ss_vocab)
ss_vocab.add_word(None)
print(ss_vocab)

samples = [streusle_record_to_model_sample(r, [supersenses.constants.TYPES.PREPOSITION_SUPERSENSE]) for r in records]
samples = [s for s in samples if any(s.ys)]
l = [token['token'].lower() for sample in samples for (token, y) in zip(sample.xs, sample.ys) if y]

evaluator = ClassifierEvaluator()

lstm_mlp_model = LstmMlpMulticlassModel(
    input_vocabularies={
        'token': token_vocab,
        'pos': pos_vocab
    },
    input_embeddings={
        'token': streusle_loader.get_tokens_word2vec_model().as_dict()
    },
    input_embeddings_default_size=300,
    output_vocabulary=ss_vocab,
    is_bilstm=True
)

print('Simple conditional model evaluation:')
scm = SimpleConditionalMulticlassModel()
scm.fit(samples, validation_split=0.3, evaluator=evaluator)

print('')

print('LSTM-MLP evaluation:')
lstm_mlp_model.fit(samples,
                   epochs=50,
                   show_progress=True,
                   show_epoch_eval=False,
                   validation_split=0.3,
                   evaluator=evaluator)


