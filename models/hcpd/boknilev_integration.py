from models.hcpd.hcpd_model import HCPDModel
from models.hcpd import vocabs
from wordnet_verbnet import is_word_in_verb_frames, get_noun_hypernyms


def boknilev_record_to_hcpd_samples(record):
    sample_xs = [
        HCPDModel.SampleX(
            sent_id=record['sent_id'],
            tokens=record['tokens'],
            head_cands=[HCPDModel.HeadCand(
                ind=hc['ind'],
                word=record['tokens'][hc['ind']],
                lemma=record['preprocessing']['lemma'][hc['ind']],
                pp_distance=dist + 1,
                is_verb=hc['gold']['is_verb'],
                is_noun=hc['gold']['is_noun'],
                next_pos=hc['gold']['next_pos'],
                hypernyms=record['preprocessing']['hypernyms'][hc['ind']],
                verb_ss=record['preprocessing']['gold_verb_ss'][hc['ind']] if 'gold_verb_ss' in record['preprocessing'] else None,
                noun_ss=record['preprocessing']['gold_noun_ss'][hc['ind']] if 'gold_noun_ss' in record['preprocessing'] else None,
                is_pp_in_verbnet_frame=is_word_in_verb_frames(record['tokens'][hc['ind']], record['tokens'][pp['ind']])
            ) for dist, hc in enumerate(reversed(pp['head_cands']))],
            pp=HCPDModel.PP(
                word=record['tokens'][pp['ind']],
                pss_role=record['preprocessing']['gold_pss_role'][pp['ind']] if 'gold_pss_role' in record['preprocessing'] else None,
                pss_func=record['preprocessing']['gold_pss_func'][pp['ind']] if 'gold_pss_func' in record['preprocessing'] else None,
                ind=pp['ind']
            ),
            child=HCPDModel.Child(
                word=record['tokens'][pp['child_ind'][0]],
                lemma=record['preprocessing']['lemma'][pp['child_ind'][0]],
                hypernyms=record['preprocessing']['hypernyms'][pp['child_ind'][0]],
                noun_ss=record['preprocessing']['noun_ss'][pp['child_ind']] if 'noun_ss' in record['preprocessing'] else None,
            )
        )
        for pp in record['pps'] if pp.get('child_ind')]
    return [
        HCPDModel.Sample(
            x=sample_x,
            y=HCPDModel.SampleY(
                scored_heads=[(1, hc) for hc in sample_x.head_cands if hc.ind == pp['head_ind']] + \
                             [(0, hc) for hc in sample_x.head_cands if hc.ind != pp['head_ind']]
            )
        )
        for sample_x,  pp in zip(sample_xs, [pp for pp in record['pps'] if pp.get('child_ind')])
    ]
