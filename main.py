import supersenses
from datasets.steusle import streusle
from ss_classifier_baseline.model import SupersensesClassifierBaselineModel, Sample, XTokenData, YTokenData
from vocabulary import Vocabulary

def streusle_record_to_model_sample(record):
    return Sample(
        xs=[XTokenData(
                token=tagged_token.token,
                pos=tagged_token.pos
            ) for tagged_token in record.tagged_tokens],
        ys=[YTokenData(
            supersense=tagged_token.supersense
        ) for tagged_token in record.tagged_tokens],
    )

loader = streusle.StreusleLoader()
records = loader.load()
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

model = SupersensesClassifierBaselineModel(
    token_vocab=token_vocab,
    pos_vocab=pos_vocab,
    ss_vocab=ss_vocab,
    ss_types_to_predict=[supersenses.constants.TYPES.PREPOSITION_SUPERSENSE],
    is_bilstm=False
)

samples = [streusle_record_to_model_sample(r) for r in records]
predictor = model.fit(samples, epochs=3)