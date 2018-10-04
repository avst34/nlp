import json
import os
import zipfile
from collections import namedtuple
from glob import glob
from pprint import pprint

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC

from models.svm_pairwise_func_clust import embeddings
from models.svm_pairwise_func_clust import vocabs
from models.svm_pairwise_func_clust.svm_pairwise_func_clust_evaluator import SvmPairwiseFuncClustEvaluator

MLPSoftmaxParam = namedtuple('MLPSoftmax', ['mlp', 'softmax'])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

def concat(lists):
    return sum(lists, [])



class SvmPairwiseFuncClustModel:

    class SampleX:

        def __init__(self,
                     prep_tokens1,
                     prep_xpos1,
                     gov_token1,
                     gov_xpos1,
                     obj_token1,
                     obj_xpos1,
                     govobj_config1,
                     ud_dep1,
                     role1,
                     prep_tokens2,
                     prep_xpos2,
                     gov_token2,
                     gov_xpos2,
                     obj_token2,
                     obj_xpos2,
                     govobj_config2,
                     ud_dep2,
                     role2,
                     ):
            self.prep_tokens1 = prep_tokens1
            self.prep_xpos1 = prep_xpos1
            self.gov_token1 = gov_token1
            self.gov_token1 = gov_token1
            self.gov_xpos1 = gov_xpos1
            self.obj_token1 = obj_token1
            self.obj_xpos1 = obj_xpos1
            self.govobj_config1 = govobj_config1
            self.ud_dep1 = ud_dep1
            self.role1 = role1
            self.prep_tokens2 = prep_tokens2
            self.prep_xpos2 = prep_xpos2
            self.gov_token2 = gov_token2
            self.gov_token2 = gov_token2
            self.gov_xpos2 = gov_xpos2
            self.obj_token2 = obj_token2
            self.obj_xpos2 = obj_xpos2
            self.govobj_config2 = govobj_config2
            self.ud_dep2 = ud_dep2
            self.role2 = role2

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print(repr(self))

        @staticmethod
        def from_dict(d):
            return SvmPairwiseFuncClustModel.SampleX(**d)

        def __repr__(self):
            return '[prep1:%s,xpos:%s,gov:%s,gov_xpos:%s,govobj_config:%s,ud_dep:%s,obj:%s,obj_xpos:%s]' % (
                ' '.join(self.prep_tokens1), self.prep_xpos1, self.gov_token1, self.gov_xpos1, self.govobj_config1, self.ud_dep1, self.obj_token1, self.obj_xpos1
            ) + ' <-> [prep2:%s,xpos:%s,gov:%s,gov_xpos:%s,govobj_config:%s,ud_dep:%s,obj:%s,obj_xpos:%s]' % (
                ' '.join(self.prep_tokens2), self.prep_xpos2, self.gov_token2, self.gov_xpos2, self.govobj_config2, self.ud_dep2, self.obj_token2, self.obj_xpos2
            )

    class SampleY:

        def __init__(self, is_same_cluster):
            self.is_same_cluster = is_same_cluster

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print('[same_cluster:%s]' % self.is_same_cluster)

        @staticmethod
        def from_dict(d):
            return SvmPairwiseFuncClustModel.SampleY(**d)

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
            return SvmPairwiseFuncClustModel.Sample(
                x=SvmPairwiseFuncClustModel.SampleX.from_dict(d['x']),
                y=SvmPairwiseFuncClustModel.SampleY.from_dict(d['y']),
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
                     svm_c,
                     svm_gamma,
                     svm_shrinking,
                     svm_tol
                     ):
            self.svm_c = svm_c
            self.svm_tol = svm_tol
            self.shrinking = svm_shrinking
            self.svm_gamma = svm_gamma
            self.use_ud_dep = use_ud_dep
            self.use_prep = use_prep
            self.use_gov = use_gov
            self.use_obj = use_obj
            self.use_govobj_config = use_govobj_config
            self.use_role = use_role
            self.token_embd_dim = token_embd_dim

        def clone(self, override=None):
            override = override or {}
            params = self.__dict__
            params.update(override)
            return SvmPairwiseFuncClustModel.HyperParameters(**params)

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

        print("SvmPairwiseFuncClustModel: Building model with the following hyperparameters:")
        pprint(hyperparameters.__dict__)

    def get_svm_vec_dim(self):
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

        # ## DEBUG-START
        # dim += self.pss_vocab.size()
        # ## DEBUG-END

        return 2*dim

    def _get_svm_half_vec(self, preps, prep_xpos, gov, gob_xpos, obj, obj_xpos, govobj_config, ud_dep, role):
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
        return concat([
            self._get_svm_half_vec(
                x.prep_tokens1,
                x.prep_xpos1,
                x.gov_token1,
                x.gov_xpos1,
                x.obj_token1,
                x.obj_xpos1,
                x.govobj_config1,
                x.ud_dep1,
                x.role1,
            ),
            self._get_svm_half_vec(
                x.prep_tokens2,
                x.prep_xpos2,
                x.gov_token2,
                x.gov_xpos2,
                x.obj_token2,
                x.obj_xpos2,
                x.govobj_config2,
                x.ud_dep2,
                x.role2,
        )])

    def fit(self, samples, validation_samples,
            evaluator=None):
        print('fit: preparing vectors..')
        vectorized = []
        for ind, s in enumerate(samples):
            vectorized.append(self._vectorize(s))
            if ind % 1000 == 0:
                print("%d/%d" % (ind, len(samples)))
        print('fit: converting to numpy..')
        X_train = np.array([vectorized])
        print('fit: scaling vectors')
        self.scaler = preprocessing.StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        Y_train = np.array([1 if s.y.is_same_cluster else -1 for s in samples])

        self.model = SVC(C=self.hyperparameters.svm_c,
                         gamma=self.hyperparameters.svm_gamma,
                         shrinking=self.hyperparameters.svm_shrinking,
                         tol=self.hyperparameters.svm_tol,
                         class_weight='balanced',
                         cache_size=512,
                         verbose=True,
                         kernel='rbf')
        print('fit: training model..')
        self.model.fit(X_train, Y_train)
        evaluator = evaluator or SvmPairwiseFuncClustEvaluator()
        print('fit: evaluating..')
        self.test_eval = evaluator.evaluate(validation_samples, examples_to_show=5, predictor=self)
        print('fit: done')
        return self

    def predict(self, sample_x):
        x_vec = self.scaler.transform(np.array(self._vectorize(sample_x)))
        return SvmPairwiseFuncClustModel.SampleY(self.model.predict(x_vec)[0] == 1)

    def get_token_embedding(self, tok):
        return list(self.embeddings.get(tok, self.embeddings.get(tok.lower(), [0] * self.hyperparameters.token_embd_dim)))

    def get_tokens_embedding(self, toks):
        return [sum(xs)/len(xs) for xs in zip(*[self.get_token_embedding(tok) for tok in toks])]

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
