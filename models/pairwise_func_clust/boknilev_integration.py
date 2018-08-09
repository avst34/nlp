from models.supersenses.embeddings import TOKENS_WORD2VEC, LEMMAS_WORD2VEC
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from word2vec import Word2VecModel

w2v = Word2VecModel.load_google_model()

def boknilev_record_to_lstm_model_sample_xs(record):
    return [LstmMlpSupersensesModel.SampleX(
                token=record['tokens'][ind],
                ind=ind,
                ud_xpos=record['preprocessing']['ud_xpos'][ind],
                ud_upos=None,
                ner=record['preprocessing']['ner'][ind],
                lemma=record['preprocessing']['lemma'][ind],
                ud_dep=record['preprocessing']['ud_dep'][ind],
                ud_head_ind=record['preprocessing']['ud_head_ind'][ind],
                is_part_of_mwe=False,
                gov_ind=record['preprocessing']['govobj'][ind]['gov'] - 1 if record['preprocessing']['govobj'][ind]['gov'] else None,
                obj_ind=record['preprocessing']['govobj'][ind]['obj'] - 1 if record['preprocessing']['govobj'][ind]['obj'] else None,
                govobj_config=record['preprocessing']['govobj'][ind]['config'],
                identified_for_pss=ind in [pp['ind'] for pp in record['pps']],
                lexcat=None,
                token_word2vec=w2v.get(record['tokens'][ind]) if record['tokens'][ind] not in TOKENS_WORD2VEC else None,
                lemma_word2vec=w2v.get(record['preprocessing']['lemma'][ind]) if record['preprocessing']['lemma'][ind] not in LEMMAS_WORD2VEC else None
            ) for ind in range(len(record['tokens']))
    ]
