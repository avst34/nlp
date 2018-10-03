import json
import os
import pickle
import zipfile
from collections import namedtuple
from glob import glob
from pprint import pprint

import numpy as np
from sklearn.mixture import GaussianMixture

from models.gmm_func_clust import embeddings
from models.gmm_func_clust import vocabs
from models.gmm_func_clust.gmm_func_clust_evaluator import GmmFuncClustEvaluator

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'W',
    'b',
    'mlps'
])

MLPSoftmaxParam = namedtuple('MLPSoftmax', ['mlp', 'softmax'])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

def concat(lists):
    return sum(lists, [])

class GmmFuncClustModel:

    class SampleX:

        def __init__(self,
                     prep_tokens,
                     prep_xpos,
                     gov_token,
                     gov_xpos,
                     obj_token,
                     obj_xpos,
                     govobj_config,
                     ud_dep,
                     role,
                    ):
            self.prep_tokens = prep_tokens
            self.prep_xpos = prep_xpos
            self.gov_token = gov_token
            self.gov_token = gov_token
            self.gov_xpos = gov_xpos
            self.obj_token = obj_token
            self.obj_xpos = obj_xpos
            self.govobj_config = govobj_config
            self.ud_dep = ud_dep
            self.role = role

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print(repr(self))

        @staticmethod
        def from_dict(d):
            return GmmFuncClustModel.SampleX(**d)

        def __repr__(self):
            return '[prep:%s,xpos:%s,gov:%s,gov_xpos:%s,govobj_config:%s,ud_dep:%s,obj:%s,obj_xpos:%s]' % (
                ' '.join(self.prep_tokens), self.prep_xpos, self.gov_token, self.gov_xpos, self.govobj_config, self.ud_dep, self.obj_token, self.obj_xpos
            )

    class SampleY:

        def __init__(self, func, cluster=None):
            self.func = func
            self.cluster = cluster

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print('[func:%s]' % self.func)

        @staticmethod
        def from_dict(d):
            return GmmFuncClustModel.SampleY(**d)

    class Sample:

        def __init__(self, x, y, sample_id):
            self.x = x
            self.y = y
            self.sample_id = sample_id

        def to_dict(self):
            return {
                'x': self.x.to_dist(),
                'y': self.y.to_dist(),
                'sample_id': self.sample_id
            }

        @staticmethod
        def from_dict(d):
            return GmmFuncClustModel.Sample(
                x=GmmFuncClustModel.SampleX.from_dict(d['x']),
                y=GmmFuncClustModel.SampleY.from_dict(d['y']),
                sample_id=d['sample_id']
            )

        def pprint(self):
            self.x.pprint()
            self.y.pprint()

    class HyperParameters:

        def __init__(self,
                     use_prep,
                     use_ud_dep,
                     use_gov,
                     use_obj,
                     use_govobj_config,
                     use_role,
                     token_embd_dim,
                     cov_type,
                     gmm_max_iter,
                     gmm_means_init,
                     ):
            self.gmm_means_init = gmm_means_init
            self.use_ud_dep = use_ud_dep
            self.use_prep = use_prep
            self.use_gov = use_gov
            self.use_obj = use_obj
            self.use_govobj_config = use_govobj_config
            self.use_role = use_role
            self.token_embd_dim = token_embd_dim
            self.cov_type = cov_type
            self.gmm_max_iter = gmm_max_iter

        def clone(self, override=None):
            override = override or {}
            params = self.__dict__
            params.update(override)
            return GmmFuncClustModel.HyperParameters(**params)

    def __init__(self, hyperparameters, gov_vocab=None, obj_vocab=None, prep_vocab=None, pss_vocab=None,
                 govobj_config_vocab=None, ud_xpos_vocab=None, ud_dep_vocab=None, token_embeddings=None):
        self.prep_vocab = prep_vocab or vocabs.PREPS
        self.obj_vocab = obj_vocab or vocabs.OBJ
        self.gov_vocab = gov_vocab or vocabs.GOV
        self.govobj_config_vocab = govobj_config_vocab or vocabs.GOVOBJ_CONFIGS
        self.ud_xpos_vocab = ud_xpos_vocab or vocabs.UD_XPOS
        self.pss_vocab = pss_vocab or vocabs.PSS
        self.ud_dep_vocab = ud_dep_vocab or vocabs.UD_DEPS
        self.embeddings = token_embeddings or embeddings.TOKENS_WORD2VEC
        self.hyperparameters = hyperparameters

        print("GmmFuncClustModel: Building model with the following hyperparameters:")
        pprint(hyperparameters.__dict__)

    def get_vec_dim(self):
        hp = self.hyperparameters
        dim = 0
        if hp.use_prep:
            dim += hp.lstm_h_dim
            dim += self.ud_xpos_vocab.size()
        if hp.use_gov:
            dim += hp.token_embd_dim
            dim += self.ud_xpos_vocab.size()
        if hp.use_obj:
            dim += hp.token_embd_dim
            dim += self.ud_xpos_vocab.size()
        if hp.use_govobj_config:
            dim += self.govobj_config_vocab.size()
        if hp.use_ud_dep:
            dim += self.ud_dep_vocab.size()
        if hp.use_role:
            dim += self.pss_vocab.size()
        return dim

    def _get_vec(self, preps, prep_xpos, gov, gob_xpos, obj, obj_xpos, govobj_config, ud_dep, role):
        hp = self.hyperparameters
        vecs = []
        if hp.use_prep:
            vecs.append(self.get_tokens_embedding(preps))
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, prep_xpos))
        if hp.use_gov:
            vecs.append(self.get_token_embedding(gov))
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, gob_xpos))
        if hp.use_obj:
            vecs.append(self.get_token_embedding(obj))
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, obj_xpos))
        if hp.use_govobj_config:
            vecs.append(self._build_onehot_vec(self.govobj_config_vocab, govobj_config))
        if hp.use_ud_dep:
            vecs.append(self._build_onehot_vec(self.ud_dep_vocab, ud_dep))
        if hp.use_role:
            vecs.append(self._build_onehot_vec(self.pss_vocab, role))
        return concat(vecs)

    def get_token_embedding(self, tok):
        return list(self.embeddings.get(tok, self.embeddings.get(tok.lower(), [0] * self.hyperparameters.token_embd_dim)))

    def get_tokens_embedding(self, toks):
        return [sum(xs)/len(xs) for xs in zip(*[self.get_token_embedding(tok) for tok in toks])]

    def _build_vocab_onehot_embd(self, vocab):
        n_words = vocab.size()
        embeddings = {}
        for word in vocab.all_words():
            word_ind = vocab.get_index(word)
            vec = [0] * n_words
            vec[word_ind] = 1
            embeddings[word] = vec
        return embeddings

    def _build_onehot_vec(self, vocab, word):
        n_words = vocab.size()
        word_ind = vocab.get_index(word)
        vec = [0] * n_words
        vec[word_ind] = 1
        return vec

    def _vectorize(self, x):
        return self._get_vec(
            x.prep_tokens,
            x.prep_xpos,
            x.gov_token,
            x.gov_xpos,
            x.obj_token,
            x.obj_xpos,
            x.govobj_config,
            x.ud_dep,
            x.role,
        )

    def fit(self, samples, validation_samples,
            evaluator=None):
        samples = samples[:500]
        X_train = np.array([self._vectorize(s.x) for s in samples])

        means_init = None
        if self.hyperparameters.use_role and self.hyperparameters.gmm_means_init == 'by_role':
            all_classes = list(set([s.x.role for s in samples]))
            means_init = np.array([
                X_train[[ind for ind, s in enumerate(samples) if s.x.role == label]].mean(axis=0)
                for label in all_classes
            ])
        else:
            all_classes = list(set([s.y.func for s in samples]))

        n_classes = len(all_classes)
        self.model = GaussianMixture(n_components=n_classes,
                                     init_params=self.hyperparameters.gmm_means_init if self.hyperparameters.gmm_means_init != 'by_role' else 'kmeans',
                                     means_init=means_init,
                                     covariance_type=self.hyperparameters.cov_type,
                                     max_iter=self.hyperparameters.gmm_max_iter)


        print('fit: fitting GMM...')
        self.model.fit(X_train)
        print('fit: done, evaluating on train..')
        evaluator = evaluator or GmmFuncClustEvaluator()
        self.train_eval = evaluator.evaluate(samples, predictor=self)
        print('fit: done, evaluating on test..')
        self.test_eval = evaluator.evaluate(validation_samples, predictor=self)
        print('fit: done')
        return self

    def predict(self, sample_x):
        y = self.model.predict(np.array(self._vectorize(sample_x)).reshape(1, -1))[0]
        return GmmFuncClustModel.SampleY(func=None, cluster=y)

    def save(self, base_path):
        def pythonize_embds(embds):
            return {k: [float(x) for x in list(v)] for k, v in embds.items()}

        with open(base_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(base_path + '.hp', 'w') as f:
            json.dump(vars(self.hyperparameters), f, indent=2)
        vocabs = {
            name: vocab.pack() for name, vocab in {
                "gov_vocab": self.gov_vocab,
                "obj_vocab": self.obj_vocab,
                "govobj_config_vocab": self.govobj_config_vocab,
                "prep_vocab": self.prep_vocab,
                "pss_vocab": self.pss_vocab,
                "ud_dep_vocab": self.ud_dep_vocab,
                "ud_xpos_vocab": self.ud_xpos_vocab
            }.items()
        }
        with open(base_path + '.vocabs', 'w') as f:
            json.dump(vocabs, f)
        with open(base_path + '.tok_embds', 'w') as f:
            json.dump(pythonize_embds(self.embeddings), f)

        zip_path = base_path + '.zip'
        if os.path.exists(zip_path):
            os.remove(zip_path)
        files = glob(base_path + ".*") + [base_path]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zh:
            for fname in files:
                print("writing to zip..", fname)
                zh.write(fname, arcname=os.path.basename(fname))
        for fname in files:
            print("removing..", fname)
            os.remove(fname)

    @staticmethod
    def load(base_path):
        with zipfile.ZipFile(base_path + ".zip", "r") as zh:
            zh.extractall(os.path.dirname(base_path))
        try:
            with open(base_path, 'wb') as model_f:
                with open(base_path + '.hp', 'r') as hp_f:
                    with open(base_path + '.vocabs', 'r') as in_vocabs_f:
                            with open(base_path + '.tok_embds', 'r') as embds_f:
                                model = GmmFuncClustModel(
                                    hyperparameters=GmmFuncClustModel.HyperParameters(**json.load(hp_f)),
                                    token_embeddings=json.load(embds_f),
                                    **json.load(in_vocabs_f)
                                )
                                model.model = pickle.load(model_f)
                                return model
        finally:
            files = glob(base_path + ".*") + [base_path]
            for fname in files:
                if os.path.realpath(fname) != os.path.realpath(base_path + ".zip"):
                    print("loading..", fname)
                    os.remove(fname)
