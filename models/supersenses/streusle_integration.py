from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel


def streusle_record_to_lstm_model_sample(record):
    return LstmMlpSupersensesModel.Sample(
        xs=[LstmMlpSupersensesModel.SampleX(
            token=tagged_token.token,
            ind=ind,
            ud_xpos=tagged_token.ud_xpos,
            ud_upos=tagged_token.ud_upos,
            ner=tagged_token.ner,
            lemma=tagged_token.lemma,
            ud_dep=tagged_token.ud_dep,
            ud_head_ind=tagged_token.ud_head_ind,
            is_part_of_mwe=tagged_token.is_part_of_mwe,
            gov_ind=tagged_token.gov_ind,
            obj_ind=tagged_token.obj_ind,
            govobj_config=tagged_token.govobj_config,
            identified_for_pss=tagged_token.identified_for_pss,
            lexcat=tagged_token.lexcat,
        ) for ind, tagged_token in enumerate(record.tagged_tokens)
        ],
        ys=[LstmMlpSupersensesModel.SampleY(
            supersense_role=tagged_token.supersense_role,
            supersense_func=tagged_token.supersense_func
        ) for tagged_token in record.tagged_tokens
        ],
        sample_id=record.id
    )
