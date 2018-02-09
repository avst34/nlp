import os

from datasets.streusle_v4.settings_data.run_corenlp_on_conllulex import run_corenlp_on_conllulex


def fix_lexlemmas(enriched_lines):
    fixed_lines = []
    for ind, line in enumerate(enriched_lines):
        if line.strip() and not line.startswith('#'):
            cols = line.split('\t')
            orig_lexlemma = cols[12]
            if orig_lexlemma != '_':
                smwe = cols[10]
                if ':' in smwe:
                    assert smwe.split(':')[1] == '1', smwe
                    mweid = smwe.split(':')[0]
                    next_lines = enriched_lines[ind + 1:]
                    next_lines = next_lines[:next_lines.index('\n')]
                    mwe_lines = [x for x in next_lines for mid in [x.split('\t')[10]] if ':' in mid and mid.split(':')[0] == mweid]
                    mwe_lines.sort(key=lambda l: int(l.split('\t')[10].split(':')[1]))
                    lemmas = [cols[2]] + [x.split('\t')[2] for x in mwe_lines]
                    if '_' in lemmas:
                        lexlemma = '_'
                    else:
                        lexlemma = ' '.join(lemmas)
                else:
                    lexlemma = cols[2]
                print('lexlemma: "%s" -> "%s"' % (orig_lexlemma, lexlemma))
                cols[12] = lexlemma
                line = '\t'.join(cols)
        fixed_lines.append(line)
    return fixed_lines


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

    enriched_lines = fix_lexlemmas(enriched_lines)

    return ''.join(enriched_lines)

