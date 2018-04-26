from models.hcpd.hcpd_model import HCPDModel, TOP_NOUN_HYPERNYMS
from wordnet import get_noun_hypernyms

from wordnet_verbnet import is_word_in_verb_frames


def boknilev_record_to_hpcd_sample(record):
    return [HCPDModel.SampleX(
          head_cands=[HCPDModel.HeadCand(
            word=record['tokens'][hc['ind']],
            pp_distance=dist + 1,
            is_verb=hc['gold']['is_verb'],
            is_noun=hc['gold']['is_noun'],
            next_pos=hc['gold']['next_pos'],
            hypernyms=get_noun_hypernyms(record['tokens'][hc['ind']], TOP_NOUN_HYPERNYMS),
            is_pp_in_verbnet_frame=is_word_in_verb_frames(record['tokens'][hc['ind']], record['tokens'][pp['ind']])
          ) for dist, hc in enumerate(reversed(pp['head_cands']))]
    ) for pp in record['pps']]
