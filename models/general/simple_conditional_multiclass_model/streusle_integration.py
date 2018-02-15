from models.general.simple_conditional_multiclass_model.model import MostFrequentClassModel


def streusle_record_to_most_frequent_class_model_sample(record, class_names=None):
    class_names = class_names or ['supersense_role', 'supersense_func']
    return MostFrequentClassModel.Sample(
        xs=[
            {
                "token": tagged_token.token,
                "lemma": tagged_token.lemma,
                "pos": tagged_token.ud_xpos,
                "identified_for_pss": tagged_token.identified_for_pss
            }
            for tagged_token in record.tagged_tokens
        ],
        ys=[
            tuple([getattr(tagged_token, class_name) for class_name in class_names])
            for tagged_token in record.tagged_tokens
        ],
        mask=None
    )

