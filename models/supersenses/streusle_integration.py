from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel


def streusle_record_to_lstm_model_sample(record):
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
            token=tagged_token.token,
            ind=ind,
            ud_pos=tagged_token.ud_xpos,
            spacy_pos=tagged_token.spacy_pos,
            spacy_dep=tagged_token.spacy_dep,
            spacy_head_ind=tagged_token.spacy_head_ind,
            spacy_ner=tagged_token.spacy_ner,
            spacy_lemma=tagged_token.spacy_lemma,
            ud_lemma=tagged_token.ud_lemma,
            ud_dep=tagged_token.ud_dep,
            ud_head_ind=tagged_token.ud_head_ind,
            is_part_of_mwe=tagged_token.is_part_of_mwe,
        ) for ind, tagged_token in enumerate(record.tagged_tokens)
        ],
        ys=[LstmMlpSupersensesModel.SampleY(
            supersense_role=tagged_token.supersense_role,
            supersense_func=tagged_token.supersense_func
        ) for tagged_token in record.tagged_tokens
        ],
        sample_id=record.id
    )
