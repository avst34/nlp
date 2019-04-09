from models.supersenses.embeddings import TOKENS_WORD2VEC, LEMMAS_WORD2VEC
from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.preprocessing.elmo import run_elmo
from word2vec import Word2VecModel

def boknilev_record_to_lstm_model_sample_xs(record):
    elmo_vecs = run_elmo(record["tokens"])
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
                role=None,
                func=None,
                hidden=None,
                token_embd=elmo_vecs,
                lemma_embd=elmo_vecs,
                ud_grandparent_ind_override=None,
                mwe_start_ind=None
            ) for ind in range(len(record['tokens']))
    ]
