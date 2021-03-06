from models.supersenses_simple.simple_mlp_supersenses_model import SimpleMlpSupersensesModel


def streusle_record_to_simple_lstm_model_samples(rec):
    samples = []
    for ttok in rec.tagged_tokens:
        if ttok.supersense_role:
            if ttok.is_part_of_wmwe or ttok.gov_ind is None or ttok.obj_ind is None:
                continue
            preps = [(rec.get_tok_by_ud_id(toknum).token, rec.get_tok_by_ud_id(toknum).elmo) for toknum in ttok.we_toknums]
            gov = (rec.tagged_tokens[ttok.gov_ind].token, rec.tagged_tokens[ttok.gov_ind].elmo)
            obj = (rec.tagged_tokens[ttok.obj_ind].token, rec.tagged_tokens[ttok.obj_ind].elmo)
            samples.append(SimpleMlpSupersensesModel.Sample(
                sample_id='id',
                x=SimpleMlpSupersensesModel.SampleX(
                    prep_tokens=[p[0] for p in preps],
                    prep_embds=[p[1] for p in preps],
                    gov_token=gov[0],
                    gov_embd=gov[1],
                    obj_token=obj[0],
                    obj_embd=obj[1],
                    role=ttok.supersense_role,
                    func=ttok.supersense_func
                ),
                y=SimpleMlpSupersensesModel.SampleY(supersense_role=ttok.supersense_role, supersense_func=ttok.supersense_func)
            ))
    return samples

