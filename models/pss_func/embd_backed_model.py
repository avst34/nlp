import copy

from gurobipy import *

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.calc_ss_per_pp_dist import calc_ss_per_pp_dist
from models.pss_func.eval_func_pred import eval_func_pred
from models.supersenses.embeddings import TOKENS_WORD2VEC
from word2vec import Word2VecModel


def load_embeddings(preps, minus_avg=False):
    print('Loading embeddings..')
    embds = {p: TOKENS_WORD2VEC.get(p) for p in preps}
    dim = len(list(TOKENS_WORD2VEC.values())[0])
    print('Embeddings dim:', dim)
    for p, embd in list(embds.items()):
        if embd is None:
            print('WARNING: no embeddings available for "%s", will use zero vector' % p)
            del embds[p]
            # embds[p] = [0] * dim
        else:
            embds[p] = list(embd)
    print('Loading completed.')
    avg_vec = [sum(xs)/len(xs) for xs in zip(*embds.values())]
    for p, embd in embds.items():
        embds[p] = [x - y for x,y in zip(embd, avg_vec)]
    return embds


def build_model(role_dists, dist_w=1, sim_w=1, prep_embds=None):
    labels = set(sum([list(v.keys()) for v in role_dists.values()], []))

    print('preps', len(preps), preps)
    print('labels', len(labels), labels)

    role_dists = copy.deepcopy(role_dists)
    for p in preps:
        for l in labels:
            role_dists[p][l] = role_dists[p].get(l, 0)

    def cos_sim(v1, v2):
        return sum([x * y for x, y in zip(v1, v2)])

    def sim(p1, p2):
        assert p1 in preps
        assert p2 in preps
        assert p1 != p2
        embd1 = prep_embds[p1]
        embd2 = prep_embds[p2]
        assert len(embd1) == len(embd2)
        s = cos_sim(embd1, embd2)
        return s

    for p1 in preps:
        for p2 in preps:
            if p1 != p2:
                print(p1 + ' ~ ' + p2 + ': ' + str(sim(p1, p2)))

    vars = {}

    def var(p1, l1, p2, l2):
        assert p1 != p2
        assert p1 in preps
        assert p2 in preps
        assert l1 in labels
        assert l2 in labels
        (p1, l1), (p2, l2) = sorted([(p1, l1), (p2, l2)])
        varName = '-:-'.join([p1, l1, p2, l2])
        vars[varName] = vars.get(varName) or m.addVar(vtype=GRB.BINARY, name=varName)
        return vars[varName]

    m = Model('model')

    print('constraints', len(m.getConstrs()))

    for p1 in preps:
        for p2 in preps:
            if p1 < p2:
                pairSum = 0
                for l1 in labels:
                    for l2 in labels:
                        pairSum = var(p1, l1, p2, l2) + pairSum
                m.addConstr(pairSum == 1)
        m.update()

    print('constraints 0', len(m.getConstrs()))

    for p in preps:
        for l in labels:
            for p1 in preps:
                if p == p1:
                    continue
                for p2 in preps:
                    if p == p2:
                        continue
                    if p1 < p2:
                        s1 = 0
                        s2 = 0
                        for _l in labels:
                            s1 = var(p, l, p1, _l) + s1
                            s2 = var(p, l, p2, _l) + s2
                        m.addConstr(s1 == s2)
        m.update()

    print('constraints 1', len(m.getConstrs()))

    mean_sim = 1
    n = 0
    for p1 in preps:
        for p2 in preps:
            if p1 < p2 and sim(p1, p2) != 0:
                mean_sim += sim(p1, p2)
                n += 1
    mean_sim /= n
    print("mean sim:", mean_sim)

    objective = 0
    for p1 in preps:
        for p2 in preps:
            if p1 < p2:
                for l1 in labels:
                    for l2 in labels:
                        prob1 = role_dists[p1][l1]
                        prob2 = role_dists[p2][l2]
                        sim_punishment = 0 if l1 == l2 else 1/(sim(p1, p2) or mean_sim)
                        objective += var(p1, l1, p2, l2) * (dist_w * (prob1 + prob2) - sim_w * sim_punishment)

    m.setObjective(objective, GRB.MAXIMIZE)
    return m

def get_predictions(m, preps):
    preds = {}
    for v in m.getVars():
        p1, l1, p2, l2 = v.varName.split('-:-')
        if v.x:
            assert p1 not in preds or preds[p1] == l1
            assert p2 not in preds or preds[p2] == l2
            preds[p1] = l1
            preds[p2] = l2
        else:
            assert (p1 not in preds or preds[p1] != l1) or (p2 not in preds or preds[p2] != l2), "p1: %s, preds[p1]: %s, l1: %s, p2: %s, preds[p2]: %s, l2: %s" % (p1, preds[p1], l1, p2, preds[p2], l2)
    assert sorted(preps) == sorted(preds)
    return preds



if __name__ == '__main__':
    records = StreusleLoader().load()
    role_dists = calc_ss_per_pp_dist(records, top_k=20)
    preps = list(role_dists.keys())
    prep_embds = load_embeddings(preps, minus_avg=True)
    preps = [p for p in preps if prep_embds.get(p) is not None]
    role_dists = {k:v for k, v in role_dists.items() if k in preps}

    print(role_dists)
    print('Building model..')
    m = build_model(role_dists, prep_embds=prep_embds)
    print('Optimizing..')
    m.optimize()
    print('Optimization Complete.')
    # for v in m.getVars():
    #     print(v.varName, v.x)
    print('Obj:', m.objVal)
    preps = list(role_dists.keys())
    predictions = get_predictions(m, preps)
    for p, l in predictions.items():
        print(p, ':', l)
    print(eval_func_pred(predictions, records))