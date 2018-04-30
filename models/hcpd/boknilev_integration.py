from models.hcpd.hcpd_model import HCPDModel
from models.hcpd import vocabs
from wordnet_verbnet import is_word_in_verb_frames, get_noun_hypernyms


def boknilev_record_to_hcpd_samples(record):
    sample_xs = [
        HCPDModel.SampleX(
            head_cands=[HCPDModel.HeadCand(
                ind=hc['ind'],
                word=record['tokens'][hc['ind']],
                pp_distance=dist + 1,
                is_verb=hc['gold']['is_verb'],
                is_noun=hc['gold']['is_noun'],
                next_pos=hc['gold']['next_pos'],
                hypernyms=record['preprocessing']['hypernyms'][hc['ind']],
                is_pp_in_verbnet_frame=is_word_in_verb_frames(record['tokens'][hc['ind']], record['tokens'][pp['ind']])
            ) for dist, hc in enumerate(reversed(pp['head_cands']))],
            pp=HCPDModel.PP(word=record['tokens'][pp['ind']]),
            child=HCPDModel.Child(
                word=record['tokens'][pp['child_ind'][0]],
                hypernyms=record['preprocessing']['hypernyms'][pp['child_ind'][0]]
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
