from collections import Counter, namedtuple
import random


def pp_pos_stats(records):
    pp_pos_counter = Counter([t.ud_xpos for rec in records for t in rec.tagged_tokens if t.supersense_combined and not t.is_part_of_mwe])
    pos_counter = Counter([t.ud_xpos for rec in records for t in rec.tagged_tokens if not t.is_part_of_mwe])
    poses = sorted(pp_pos_counter.keys(), key=lambda pos: -pp_pos_counter[pos] / pos_counter[pos])
    print("POS stats:")
    for pos in poses:
        print("%s: %d/%d (%2.2f%%)" % (pos, pp_pos_counter[pos], pos_counter[pos], pp_pos_counter[pos] / pos_counter[pos] * 100))

def deps_examples(records):
    records = list(records)
    fields = ['spacy_dep', 'ud_dep']

    for field in fields:
        print(field)
        print('-----')
        all_deps = [getattr(tok, field) for rec in records for tok in rec.tagged_tokens if tok.supersense_combined]
        deps_counter = Counter(all_deps)
        deps_ordered = sorted(deps_counter, key=lambda dep: -deps_counter[dep])
        for dep in deps_ordered:
            print(dep + ' (%2.2f%%):' % (deps_counter[dep] / len(all_deps) * 100))
            print('-----------')
            found = []
            found_mwe = []
            random.shuffle(records)
            for rec in records:
                for tok in rec.tagged_tokens:
                    if tok.supersense_combined and getattr(tok, field) == dep:
                        parent = rec.tagged_tokens[getattr(tok, field.replace('dep', 'head_ind'))]
                        s = '%s: (%s -> %s) [MWE %s]:' % (rec.id, parent.token, tok.token, tok.is_part_of_mwe) + ' ' +  ' '.join([t.token for t in rec.tagged_tokens])
                        if tok.is_part_of_mwe:
                            found_mwe.append(s)
                        else:
                            found.append(s)
                    if len(found) > 3 and len(found_mwe) > 3:
                        break
                if len(found) > 3 and len(found_mwe) > 3:
                    break
            for s in found[:3]:
                print(s)
            for s in found_mwe[:3]:
                print(s)
            print('')
        print('#####')

def deps_stats(records):
    UDBin = namedtuple('UDBin', ['pos', 'parent', 'grandparent'])
    ud_bins = {}
    ud_bins_all_pos = {}
    ud_bins_all_pos_grandparent = {}
    SpacyBin = namedtuple('UDBin', ['pos', 'parent', 'children'])
    spacy_bins = {}
    spacy_bins_all_pos = {}
    spacy_bins_all_pos_parents = {}
    spacy_bins_all_pobj = {}
    spacy_bins_all_pobjless = {}

    pobjless_toks_cnt = 0

    for record in records:
        for tok in record.tagged_tokens:
            if tok.supersense_combined:
                _bin = UDBin(pos=tok.ud_xpos,
                            parent=tok.ud_dep,
                             grandparent='-- ALL --')
                ud_bins[_bin] = ud_bins.get(_bin, 0)
                ud_bins[_bin] += 1

                _bin = UDBin(pos='-- ALL --',
                            parent=tok.ud_dep,
                             grandparent='-- ALL --')
                ud_bins_all_pos[_bin] = ud_bins_all_pos.get(_bin, 0)
                ud_bins_all_pos[_bin] += 1

                _bin = UDBin(pos='-- ALL --',
                            parent='-- ALL --',
                             grandparent=record.tagged_tokens[tok.ud_head_ind].ud_dep)
                ud_bins_all_pos_grandparent[_bin] = ud_bins_all_pos_grandparent.get(_bin, 0)
                ud_bins_all_pos_grandparent[_bin] += 1

                _bin = SpacyBin(parent='-- ALL --', pos=tok.pos,
                               children=tuple(sorted([t.spacy_dep for t in record.tagged_tokens
                                       if record.tagged_tokens[t.spacy_head_ind] == tok
                                        and t != tok]))
                               )
                spacy_bins[_bin] = spacy_bins.get(_bin, 0)
                spacy_bins[_bin] += 1

                _bin = SpacyBin(parent='-- ALL --', pos='-- ALL --',
                               children=tuple(sorted([t.spacy_dep for t in record.tagged_tokens
                                       if record.tagged_tokens[t.spacy_head_ind] == tok
                                        and t != tok]))
                               )
                spacy_bins_all_pos[_bin] = spacy_bins_all_pos.get(_bin, 0)
                spacy_bins_all_pos[_bin] += 1

                _bin = SpacyBin(parent=tok.spacy_dep,
                                pos='-- ALL --',
                                children='-- ALL --')

                spacy_bins_all_pos_parents[_bin] = spacy_bins_all_pos_parents.get(_bin, 0)
                spacy_bins_all_pos_parents[_bin] += 1

                # spaCy
                children = tuple(sorted([t for t in record.tagged_tokens
                                         if record.tagged_tokens[t.spacy_head_ind] == tok
                                         and t != tok]))
                children_labels = [x.spacy_dep for x in children]
                _bin = SpacyBin(parent='-- ALL --', pos='-- ALL --',
                               children='pobj' if 'pobj' in children_labels else 'No' if len(children) == 0 else 'Other'
                               )
                spacy_bins_all_pobj[_bin] = spacy_bins_all_pobj.get(_bin, 0)
                spacy_bins_all_pobj[_bin] += 1

                if 'pobj' not in children_labels and len(children):
                    if 'pcomp' in children_labels:
                        print('pcomp (%s -> %s):' % (tok.token, [c for c in children if c.spacy_dep == 'pcomp'][0].token), ' '.join([t.token for t in record.tagged_tokens]))
                    if 'prep' in children_labels:
                        print('prep (%s -> %s):' % (tok.token, [c for c in children if c.spacy_dep == 'prep'][0].token), ' '.join([t.token for t in record.tagged_tokens]))
                    pobjless_toks_cnt += 1
                    for label in children_labels:
                        spacy_bins_all_pobjless[label] = spacy_bins_all_pobjless.get(label, 0)
                        spacy_bins_all_pobjless[label] += 1
                elif 'pobj' in children_labels:
                    print('pobj (%s -> %s):' % (tok.token, [c for c in children if c.spacy_dep == 'pobj'][0].token), ' '.join([t.token for t in record.tagged_tokens]))
                else:
                    print('no children (%s):' % (tok.token), ' '.join([t.token for t in record.tagged_tokens]))

                # UD
                ud_parent = record.tagged_tokens[tok.ud_head_ind]
                # if tok.ud_dep == 'case':
                #     print('case (%s -> %s):' % (ud_parent.token, tok.token), ' '.join([t.token for t in record.tagged_tokens]))
                # elif tok.ud_dep == 'nmod:poss':
                #     print('nmod:poss (%s -> %s):' % (ud_parent.token, tok.token), ' '.join([t.token for t in record.tagged_tokens]))
                # elif tok.ud_dep == 'mark':
                #     print('mark (%s -> %s):' % (ud_parent.token, tok.token), ' '.join([t.token for t in record.tagged_tokens]))
                # elif tok.ud_dep == 'advmod':
                #     print('advmod (%s -> %s):' % (ud_parent.token, tok.token), ' '.join([t.token for t in record.tagged_tokens]))
                # elif tok.ud_dep == 'obl':
                #     print('obl (%s -> %s):' % (ud_parent.token, tok.token), ' '.join([t.token for t in record.tagged_tokens]))
                if tok.ud_dep == 'root':
                    print('UD root (%s -> %s) [MWE %s]:' % (ud_parent.token, tok.token, tok.is_part_of_mwe), ' '.join([t.token for t in record.tagged_tokens]))

    print("UD Deps stats:")
    print("--------------")
    print("POS\tParent\tCount\tPercent")
    bins = sorted(ud_bins.keys(), key=lambda bin: -ud_bins[bin])
    for _bin in bins:
        print("%s\t%s\t%d\t%2.2f" % (_bin.pos, _bin.parent, ud_bins[_bin], ud_bins[_bin] / sum(ud_bins.values()) * 100))
    print("")

    print("UD Deps stats (aggr):")
    print("--------------")
    print("Parent\tCount\tPercent")
    bins = sorted(ud_bins_all_pos.keys(), key=lambda bin: -ud_bins_all_pos[bin])
    for _bin in bins:
        print("%s\t%d\t%2.2f" % (_bin.parent, ud_bins_all_pos[_bin], ud_bins_all_pos[_bin] / sum(ud_bins_all_pos.values()) * 100))
    print("")

    print("UD Deps stats (aggr-grandparent):")
    print("--------------")
    print("Grandparent\tCount\tPercent")
    bins = sorted(ud_bins_all_pos_grandparent.keys(), key=lambda bin: -ud_bins_all_pos_grandparent[bin])
    for _bin in bins:
        print("%s\t%d\t%2.2f" % (_bin.grandparent, ud_bins_all_pos_grandparent[_bin], ud_bins_all_pos_grandparent[_bin] / sum(ud_bins_all_pos_grandparent.values()) * 100))
    print("")

    print("Spacy Deps stats:")
    print("--------------")
    print("POS\tParent\tChildren\tpobj\tCount\tPercent")
    bins = sorted(spacy_bins.keys(), key=lambda bin: -spacy_bins[bin])
    for _bin in bins:
        print("%s\t%s\t%s\t%d\t%2.2f" % (_bin.pos, str(_bin.children), str('pobj' in _bin.children), spacy_bins[_bin], spacy_bins[_bin] / sum(spacy_bins.values()) * 100))
    print("")

    print("Spacy Deps stats (aggr):")
    print("--------------")
    print("Children\tpobj\tCount\tPercent")
    bins = sorted(spacy_bins_all_pos.keys(), key=lambda bin: -spacy_bins_all_pos[bin])
    for _bin in bins:
        print("%s\t%s\t%d\t%2.2f" % (str(_bin.children), str('pobj' in _bin.children), spacy_bins_all_pos[_bin], spacy_bins_all_pos[_bin] / sum(spacy_bins_all_pos.values()) * 100))
    print("")

    print("Spacy Deps stats (aggr-pobj):")
    print("--------------")
    print("pobj\tCount\tPercent")
    bins = sorted(spacy_bins_all_pobj.keys(), key=lambda bin: -spacy_bins_all_pobj[bin])
    for _bin in bins:
        print("%s\t%d\t%2.2f" % (str(_bin.children), spacy_bins_all_pobj[_bin], spacy_bins_all_pobj[_bin] / sum(spacy_bins_all_pobj.values()) * 100))
    print("")

    print("Spacy Deps stats (aggr-pobjless):")
    print("--------------")
    print("child\tCount\tPercent")
    bins = sorted(spacy_bins_all_pobjless.keys(), key=lambda bin: -spacy_bins_all_pobjless[bin])
    for _bin in bins:
        print("%s\t%d\t%2.2f" % (_bin, spacy_bins_all_pobjless[_bin], spacy_bins_all_pobjless[_bin] / pobjless_toks_cnt * 100))
    print("")

    print("Spacy Deps stats (aggr-parent):")
    print("--------------")
    print("Parent\tCount\tPercent")
    bins = sorted(spacy_bins_all_pos_parents.keys(), key=lambda bin: -spacy_bins_all_pos_parents[bin])
    for _bin in bins:
        print("%s\t%d\t%2.2f" % (_bin.parent, spacy_bins_all_pos_parents[_bin], spacy_bins_all_pos_parents[_bin] / sum(spacy_bins_all_pos_parents.values()) * 100))
    print("")


def run(records):
    # pp_pos_stats(records)
    # deps_stats(records)
    deps_examples(records)

