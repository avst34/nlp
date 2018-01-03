from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel


def streusle_record_to_lstm_model_sample(record, drop_mwes_pss=True):
    print('WARNING: Dropping MWE pss from streusle records')
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
            token=tagged_token.token,
            ind=ind,
            ud_pos=tagged_token.ud_xpos,
            spacy_pos=tagged_token.spacy_pos,
            spacy_dep=tagged_token.spacy_dep,
            spacy_head_ind=tagged_token.spacy_head_ind,
            spacy_ner=tagged_token.spacy_ner,
            ud_dep=tagged_token.ud_dep,
            ud_head_ind=tagged_token.ud_head_ind,
            is_part_of_mwe=tagged_token.is_part_of_mwe,
        ) for ind, tagged_token in enumerate(record.tagged_tokens)
        ],
        ys=[LstmMlpSupersensesModel.SampleY(
            supersense_role=tagged_token.supersense_role if not tagged_token.is_part_of_mwe or not drop_mwes_pss,
            supersense_func=tagged_token.supersense_func if not tagged_token.is_part_of_mwe or not drop_mwes_pss,
        ) for tagged_token in record.tagged_tokens
        ],
        id=record.id
    )
