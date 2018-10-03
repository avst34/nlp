from models.gmm_func_clust.gmm_func_clust_model import GmmFuncClustModel


def streusle_records_to_gmm_func_clust_model_samples(recs):
    samples = []
    for rec in recs:
        for ttok in rec.tagged_tokens:
            if ttok.supersense_role:
                if ttok.is_part_of_wmwe or ttok.gov_ind is None or ttok.obj_ind is None:
                    continue
                preps = [(rec.get_tok_by_ud_id(toknum).token, rec.get_tok_by_ud_id(toknum).ud_xpos) for toknum in ttok.we_toknums]
                gov = (rec.tagged_tokens[ttok.gov_ind].token, rec.tagged_tokens[ttok.gov_ind].ud_xpos)
                obj = (rec.tagged_tokens[ttok.obj_ind].token, rec.tagged_tokens[ttok.obj_ind].ud_xpos)
                samples.append(
                    GmmFuncClustModel.Sample(
                        x=GmmFuncClustModel.SampleX(
                            prep_tokens=[p[0] for p in preps],
                            prep_xpos=preps[0][1],
                            gov_token=gov[0],
                            gov_xpos=gov[1],
                            obj_token=obj[0],
                            obj_xpos=obj[1],
                            govobj_config=ttok.govobj_config,
                            ud_dep=ttok.ud_dep,
                            role=ttok.supersense_role,
                        ),
                        y=GmmFuncClustModel.SampleY(
                            func=ttok.supersense_func,
                            cluster=None
                        ),
                        sample_id=str(rec.id) + '_' + str(ttok.ud_id)
                    )
                )

    return samples

