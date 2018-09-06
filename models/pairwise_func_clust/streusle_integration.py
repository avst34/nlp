import itertools

from models.pairwise_func_clust.pairwise_func_clust_model import PairwiseFuncClustModel


def streusle_records_to_pairwise_func_clust_model_samples(recs):
    nodes = []
    for rec in recs:
        for ttok in rec.tagged_tokens:
            if ttok.supersense_role:
                if ttok.is_part_of_wmwe or ttok.gov_ind is None or ttok.obj_ind is None:
                    continue
                preps = [(rec.get_tok_by_ud_id(toknum).token, rec.get_tok_by_ud_id(toknum).ud_xpos) for toknum in ttok.we_toknums]
                gov = (rec.tagged_tokens[ttok.gov_ind].token, rec.tagged_tokens[ttok.gov_ind].ud_xpos)
                obj = (rec.tagged_tokens[ttok.obj_ind].token, rec.tagged_tokens[ttok.obj_ind].ud_xpos)
                nodes.append({
                    'sample_id': 'id',
                    'prep_tokens': [p[0] for p in preps],
                    'prep_xpos': preps[0][1],
                    'gov_token': gov[0],
                    'gov_xpos': gov[1],
                    'obj_token': obj[0],
                    'obj_xpos': obj[1],
                    'govobj_config': ttok.govobj_config,
                    'ud_dep': ttok.ud_dep,
                    'supersense_role': ttok.supersense_role,
                    'supersense_func': ttok.supersense_func
                })

    samples = []
    for node1, node2 in itertools.combinations(nodes, 2):
        samples.append(PairwiseFuncClustModel.Sample(
            x=PairwiseFuncClustModel.SampleX(
                prep_tokens1=node1['prep_tokens'],
                prep_xpos1=node1['prep_xpos'],
                gov_token1=node1['gov_token'],
                gov_xpos1=node1['gov_xpos'],
                obj_token1=node1['obj_token'],
                obj_xpos1=node1['obj_xpos'],
                govobj_config1=node1['govobj_config'],
                ud_dep1=node1['ud_dep'],
                role1=node1['supersense_role'],
                prep_tokens2=node2['prep_tokens'],
                prep_xpos2=node2['prep_xpos'],
                gov_token2=node2['gov_token'],
                gov_xpos2=node2['gov_xpos'],
                obj_token2=node2['obj_token'],
                obj_xpos2=node2['obj_xpos'],
                govobj_config2=node2['govobj_config'],
                ud_dep2=node2['ud_dep'],
                role2=node2['supersense_role'],
            ),
            y=PairwiseFuncClustModel.SampleY(
                is_same_cluster_prob=1 if node1['supersense_func'] == node2['supersense_func'] else 0
            ),
            sample_id=node1['sample_id'] + ':' + node2['sample_id']
        ))

    return samples

