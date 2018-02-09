import os

from datasets.streusle_v4.settings_data.run_corenlp_on_conllulex import run_corenlp_on_conllulex


def substitute_conllulex_corenlp(conllulex_fpath):
    outs = run_corenlp_on_conllulex(conllulex_fpath, 'conllu')

    with open(conllulex_fpath, 'r', encoding='utf8') as f:
        orig_lines = f.readlines()

    enriched_lines = []
    sent_ind = 0
    corenlp_tok_ind = 0
    for orig_line in orig_lines:
        if orig_line.startswith('#'):
            enriched_lines.append(orig_line)
        elif orig_line.strip() == '':
            sent_ind += 1
            corenlp_tok_ind = 0
            enriched_lines.append(orig_line)
        else:
            orig_cols = orig_line.split('\t')
            if '.' in orig_cols[0]:
                enhanced_cols = orig_cols[:2] + ['_'] * 8 + orig_cols[10:]
            else:
                conllu_cols = outs[sent_ind][corenlp_tok_ind].split('\t')
                assert orig_cols[0] == conllu_cols[0]
                assert orig_cols[1] == conllu_cols[1]
                enhanced_cols = conllu_cols[:10] + orig_cols[10:]
                corenlp_tok_ind += 1
            assert(len(enhanced_cols)) == 19
            enriched_lines.append('\t'.join(enhanced_cols))


    return ''.join(enriched_lines)

