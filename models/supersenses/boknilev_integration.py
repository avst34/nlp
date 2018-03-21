from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel


def boknilev_record_to_lstm_model_sample_xs(record):
    return [LstmMlpSupersensesModel.SampleX(
                token=record['tokens'][ind],
                ind=ind,
                ud_xpos=record['preprocessing']['xpos'][ind],
                ud_upos=None,
                ner=record['preprocessing']['ner'][ind],
                lemma=record['preprocessing']['lemma'][ind],
                ud_dep=record['preprocessing']['ud_dep'][ind],
                ud_head_ind=record['preprocessing']['ud_head_ind'][ind],
                is_part_of_mwe=False,
                gov_ind=record['preprocessing']['govobj'][ind]['gov_ind'],
                obj_ind=record['preprocessing']['govobj'][ind]['obj_ind'],
                govobj_config=record['preprocessing']['govobj'][ind]['config'],
                identified_for_pss=ind in [pp['ind'] for pp in record['pps']],
                lexcat=None,
            ) for ind in range(len(record['tokens']))
    ]
