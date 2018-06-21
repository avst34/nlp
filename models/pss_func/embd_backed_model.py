from gurobipy import *

from datasets.streusle_v4 import StreusleLoader
from models.pss_func.calc_ss_per_pp_dist import calc_ss_per_pp_dist
from models.supersenses.embeddings import TOKENS_WORD2VEC
from word2vec import Word2VecModel


def load_embeddings(preps):
    print('Loading embeddings..')
    embds = {p: TOKENS_WORD2VEC.get(p) for p in preps}
    dim = len(list(TOKENS_WORD2VEC.values())[0])
    print('Embeddings dim:', dim)
    for p, embd in embds.items():
        if embd is None:
            print('WARNING: no embeddings available for "%s", will use zero vector' % p)
        embds[p] = [0] * dim
    print('Loading completed.')
    return embds


def build_model(role_dists, dist_w, sim_w, prep_embds=None):
    preps = list(role_dists.keys())
    prep_embds = prep_embds or load_embeddings(preps)
    labels = list(list(role_dists.values())[0].keys())

    def cos_sim(p1, p2):
        assert p1 in preps
        assert p2 in preps
        assert p1 != p2
        embd1 = prep_embds[p1]
        embd2 = prep_embds[p2]
        assert len(embd1) == len(embd2)
        return sum([x * y for x, y in zip(embd1, embd2)])

    vars = {}

    def var(p1, l1, p2, l2):
        assert p1 != p2
        assert p1 in preps
        assert p2 in preps
        assert l1 in labels
        assert l2 in labels
        (p1, l1), (p2, l2) = sorted([(p1, l1), (p2, l2)])
        varName = '-'.join([p1, l1, p2, l2])
        vars[varName] = vars.get(varName, m.addVar(vtype=GRB.BINARY, name=varName))
        return vars[varName]

    m = Model('model')

    for p1 in preps:
        for p2 in preps:
            if p1 < p2:
                pairSum = 0
                for l1 in labels:
                    for l2 in labels:
                        pairSum = var(p1, l1, p2, l2) + pairSum
                m.addConstr(pairSum == 1)

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

    objective = 0
    for p1 in preps:
        for p2 in preps:
            if p1 < p2:
                for l1 in labels:
                    for l2 in labels:
                        prob1 = role_dist[p1][l1]
                        prob2 = role_dist[p2][l2]
                        sim_punishment = 0 if l1 == l2 else 1/cos_sim(p1, p2)
                        objective = objective + var(p1, l1, p2, l2) * (dist_w * (prob1 + prob2) - sim_w * sim_punishment)

    m.setObjective(objective)
    return m

if __name__ == '__main__':
    records = StreusleLoader().load()
    role_dist = calc_ss_per_pp_dist(records)
    print('Building model..')
    m = build_model(role_dist, 1, 1)
    print('Optimizing..')
    m.optimize()
    print('Optimization Complete.')
    for v in m.getVars():
        print(v.varName, v.x)
    print('Obj:', m.objVal)

