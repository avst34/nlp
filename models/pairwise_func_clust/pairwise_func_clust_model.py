import json
import os
import random
import zipfile
from collections import namedtuple
from glob import glob
from pprint import pprint

import dynet as dy
import math
import numpy as np

from dynet_utils import get_activation_function
from models.pairwise_func_clust import embeddings
from models.pairwise_func_clust import vocabs
from models.pairwise_func_clust.pairwise_func_clust_evaluator import PairwiseFuncClustEvaluator

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'W',
    'b',
    'mlps'
])

MLPSoftmaxParam = namedtuple('MLPSoftmax', ['mlp', 'softmax'])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class PairwiseFuncClustModel:

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
            return PairwiseFuncClustModel.SampleX(**d)

        def __repr__(self):
            return '[prep1:%s,xpos:%s,gov:%s,gov_xpos:%s,govobj_config:%s,ud_dep:%s,obj:%s,obj_xpos:%s]' % (
                ' '.join(self.prep_tokens1), self.prep_xpos1, self.gov_token1, self.gov_xpos1, self.govobj_config1, self.ud_dep1, self.obj_token1, self.obj_xpos1
            ) + ' <-> [prep2:%s,xpos:%s,gov:%s,gov_xpos:%s,govobj_config:%s,ud_dep:%s,obj:%s,obj_xpos:%s]' % (
                ' '.join(self.prep_tokens2), self.prep_xpos2, self.gov_token2, self.gov_xpos2, self.govobj_config2, self.ud_dep2, self.obj_token2, self.obj_xpos2
            )

    class SampleY:

        def __init__(self, is_same_cluster_prob):
            self.is_same_cluster_prob = is_same_cluster_prob
            self.is_same_cluster = is_same_cluster_prob > 0.5

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print('[same_cluster:%s (%2.2f)]' % (self.is_same_cluster, self.is_same_cluster_prob))

        @staticmethod
        def from_dict(d):
            return PairwiseFuncClustModel.SampleY(**d)

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
            return PairwiseFuncClustModel.Sample(
                x=PairwiseFuncClustModel.SampleX.from_dict(d['x']),
                y=PairwiseFuncClustModel.SampleY.from_dict(d['y']),
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
                     update_prep_embd,
                     token_embd_dim,
                     internal_token_embd_dim,
                     # use_instance_embd,
                     num_mlp_layers,
                     mlp_layer_dim,
                     mlp_activation,
                     lstm_h_dim,
                     num_lstm_layers,
                     is_bilstm,
                     # mlp_dropout_p,
                     lstm_dropout_p,
                     epochs,
                     learning_rate,
                     learning_rate_decay,
                     dynet_random_seed
                     ):
            self.mlp_activation = mlp_activation
            self.lstm_dropout_p = lstm_dropout_p
            self.internal_token_embd_dim = internal_token_embd_dim
            self.use_ud_dep = use_ud_dep
            self.use_prep = use_prep
            self.use_gov = use_gov
            self.use_obj = use_obj
            self.use_govobj_config = use_govobj_config
            self.use_role = use_role
            self.update_prep_embd = update_prep_embd
            self.token_embd_dim = token_embd_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_layer_dim = mlp_layer_dim
            self.num_mlp_layers = num_mlp_layers
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.learning_rate_decay = learning_rate_decay
            self.dynet_random_seed = dynet_random_seed

        def clone(self, override=None):
            override = override or {}
            params = self.__dict__
            params.update(override)
            return PairwiseFuncClustModel.HyperParameters(**params)

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

        print("PairwiseFuncClustModel: Building model with the following hyperparameters:")
        pprint(hyperparameters.__dict__)
        self.pc = self._build_network_params()

    def get_softmax_vec_dim(self):
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

    def _get_softmax_half_vec(self, preps, prep_xpos, gov, gob_xpos, obj, obj_xpos, govobj_config, ud_dep, role):
        hp = self.hyperparameters
        vecs = []
        if hp.use_prep:
            if not self.hyperparameters.is_bilstm:
                cur_lstm_state = self.lstm_builder.initial_state()
            else:
                cur_lstm_state = self.lstm_builder
            embeddings = [
                dy.concatenate([
                    dy.lookup(
                        self.params.input_lookups['prep'],
                        self.prep_vocab.get_index(tok),
                        update=self.hyperparameters.update_prep_embd
                        ),
                    dy.lookup(
                        self.params.input_lookups['prep_internal'],
                        self.prep_vocab.get_index(tok),
                        update=True
                    )
                ])
                for tok in preps
            ]
            lstm_output = cur_lstm_state.transduce(embeddings)[-1]
            vecs.append(lstm_output)
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, prep_xpos))
        if hp.use_gov:
            vecs.append(dy.lookup(self.params.input_lookups['gov'], self.gov_vocab.get_index(gov)))
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, gob_xpos))
        if hp.use_obj:
            vecs.append(dy.lookup(self.params.input_lookups['obj'], self.obj_vocab.get_index(obj)))
            vecs.append(self._build_onehot_vec(self.ud_xpos_vocab, obj_xpos))
        if hp.use_govobj_config:
            vecs.append(self._build_onehot_vec(self.govobj_config_vocab, govobj_config))
        if hp.use_ud_dep:
            vecs.append(self._build_onehot_vec(self.ud_dep_vocab, ud_dep))
        if hp.use_role:
            vecs.append(self._build_onehot_vec(self.pss_vocab, role))
        return dy.concatenate(vecs)

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters
        self.params = ModelOptimizedParams(
            input_lookups={
                "gov": pc.add_lookup_parameters((self.gov_vocab.size(), hp.token_embd_dim)),
                "obj": pc.add_lookup_parameters((self.obj_vocab.size(), hp.token_embd_dim)),
                "prep": pc.add_lookup_parameters((self.prep_vocab.size(), hp.token_embd_dim)),
                "prep_internal": pc.add_lookup_parameters((self.prep_vocab.size(), hp.internal_token_embd_dim))
            },
            W=pc.add_parameters((2, self.get_softmax_vec_dim() if hp.num_mlp_layers == 0 else hp.mlp_layer_dim)),
            b=pc.add_parameters((2,)),
            mlps=[MLPLayerParams(
                pc.add_parameters((hp.mlp_layer_dim, self.get_softmax_vec_dim() if i == 0 else hp.mlp_layer_dim)),
                pc.add_parameters((hp.mlp_layer_dim,))
            ) for i in range(hp.num_mlp_layers)]
        )

        input_vocabs = {
            'gov': self.gov_vocab,
            'obj': self.obj_vocab,
            'prep': self.prep_vocab,
        }
        for field, lookup_param in sorted(self.params.input_lookups.items()):
            vocab = input_vocabs.get(field)
            if vocab:
                miss = 0
                for word in vocab.all_words():
                    word_index = input_vocabs[field].get_index(word)
                    vector = self.embeddings.get(word, self.embeddings.get(word.lower()))
                    if vector is not None:
                        lookup_param.init_row(word_index, vector)
                    else:
                        # if field not in self.hyperparameters.input_embeddings_to_allow_partial:
                        #     raise Exception('Missing embedding vector for field: %s, word %s' % (field, word))
                        lookup_param.init_row(word_index, [0] * hp.token_embd_dim)
                        miss += 1
                print('%s: %d/%d embeddings missing (%2.2f%%)' % (field, miss, len(vocab.all_words()), miss / len(vocab.all_words()) * 100))

        embedded_input_dim = hp.token_embd_dim + hp.internal_token_embd_dim
        if self.hyperparameters.is_bilstm:
            self.lstm_builder = dy.BiRNNBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc, dy.LSTMBuilder)
        else:
            self.lstm_builder = dy.LSTMBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc)

        return pc

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
        return dy.inputTensor(vec)

    def _build_network_for_input(self, x, apply_dropout):
        if apply_dropout:
            self.lstm_builder.set_dropout(self.hyperparameters.lstm_dropout_p)

        softmax_vec = dy.concatenate([
            self._get_softmax_half_vec(
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
            self._get_softmax_half_vec(
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

        mlp_vec = softmax_vec
        mlp_activation = get_activation_function(self.hyperparameters.mlp_activation)
        for mlp in self.params.mlps:
            mlp_vec = mlp_activation(dy.parameter(mlp.W) * mlp_vec + dy.parameter(mlp.b))

        out = dy.log_softmax(dy.parameter(self.params.W) * mlp_vec + dy.parameter(self.params.b))
        return {
            False: out[0],
            True: out[1]
        }

    def _build_loss(self, output, y):
        # print("loss-out True:", output[True], math.exp(output[True].npvalue()))
        # print("loss-out False:", output[False], math.exp(output[False].npvalue()))
        # print("True?:", y.is_same_cluster)
        return -output[y.is_same_cluster]

    def fit(self, samples, validation_samples, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        evaluator = evaluator or PairwiseFuncClustEvaluator()
        self.pc = self._build_network_params()

        self.test_set_evaluation = []
        self.train_set_evaluation = []

        best_test_f1 = None
        train_f1 = None
        best_epoch = None
        model_file_path = '/tmp/_m_' + str(random.randrange(10000))

        true_samples = [s for s in samples if s.y.is_same_cluster]
        false_samples = [s for s in samples if not s.y.is_same_cluster]

        # true_val_samples = [s for s in validation_samples if s.y.is_same_cluster]
        # false_val_samples = [s for s in validation_samples if not s.y.is_same_cluster]

        try:
            trainer = dy.SimpleSGDTrainer(self.pc, learning_rate=self.hyperparameters.learning_rate)
            for epoch in range(self.hyperparameters.epochs):
                if np.isinf(trainer.learning_rate):
                    break

                # test = true_val_samples + random.sample(false_val_samples, len(true_val_samples))

                loss_sum = 0

                BATCH_SIZE = 20
                BATCHES_PER_EPOCH = 2000
                EPOCH_SAMPLES_COUNT = BATCH_SIZE * BATCHES_PER_EPOCH

                batches = [
                    random.sample(true_samples, BATCH_SIZE // 2)
                           +
                    random.sample(false_samples, BATCH_SIZE // 2)
                for _ in range(BATCHES_PER_EPOCH)]

                for batch_ind, batch in enumerate(batches):
                    random.shuffle(batch)
                    dy.renew_cg(immediate_compute=True, check_validity=True)
                    losses = []
                    for sample in batch:
                        output = self._build_network_for_input(sample.x, apply_dropout=True)
                        sample_loss = self._build_loss(output, sample.y)
                        if sample_loss is not None:
                            losses.append(sample_loss)
                    if len(losses):
                        batch_loss = dy.esum(losses)
                        batch_loss.forward()
                        # print('Loss before update:', batch_loss.value())
                        batch_loss.backward()
                        v = batch_loss.value()
                        loss_sum += v
                        trainer.update()
                        # print('Loss after update:', batch_loss.value(recalculate=True))

                    if show_progress:
                        if int((batch_ind + 1) / len(batches) * 100) > int(batch_ind / len(batches) * 100):
                            per = int((batch_ind + 1) / len(batches) * 100)
                            print('\r\rEpoch %3d (%d%%): |' % (epoch, per) + '#' * per + '-' * (100 - per) + '|',)
                if self.hyperparameters.learning_rate_decay:
                    trainer.learning_rate /= (1 - self.hyperparameters.learning_rate_decay)

                if evaluator and show_epoch_eval:
                    print('--------------------------------------------')
                    print('Epoch %d complete, avg loss: %1.4f' % (epoch, loss_sum/EPOCH_SAMPLES_COUNT))
                    print('Validation data evaluation:')
                    epoch_test_eval = evaluator.evaluate(validation_samples, examples_to_show=5, predictor=self)
                    self.test_set_evaluation.append(epoch_test_eval)
                    # print('Training data evaluation:')
                    # epoch_train_eval = evaluator.evaluate(train, examples_to_show=5, predictor=self)
                    # self.train_set_evaluation.append(epoch_train_eval)
                    print('--------------------------------------------')

                    get_f1 = lambda ev: ev['f1']

                    test_f1 = get_f1(epoch_test_eval)
                    if best_test_f1 is None or test_f1 > best_test_f1:
                        print("Best epoch so far! with f1 of: %1.2f" % test_f1)
                        best_test_f1 = test_f1
                        best_test_eval = epoch_test_eval
                        best_epoch = epoch
                        self.pc.save(model_file_path)

            self.test_f1 = best_test_f1
            self.test_eval = best_test_eval
            self.best_epoch = best_epoch
            print('--------------------------------------------')
            print('Training is complete (%d samples, %d epochs)' % (len(samples), self.hyperparameters.epochs))
            print('--------------------------------------------')
            self.pc.populate(model_file_path)
        finally:
            if os.path.exists(model_file_path):
                os.remove(model_file_path)

        return self

    def predict(self, sample_x):
        dy.renew_cg()
        outputs = self._build_network_for_input(sample_x, apply_dropout=False)
        output = outputs[True]
        # print('True-Output log:', output.npvalue())
        prob = math.exp(output.npvalue())
        # print('True-Output prob:', prob)
        # print('False-Output log:', outputs[False].npvalue())
        # print('False-Output prob:', math.exp(outputs[False].npvalue()))
        # print('Sum:', math.exp(outputs[False].npvalue()) + prob)

        return PairwiseFuncClustModel.SampleY(prob)

    # def predict_dist(self, x):
    #     dy.renew_cg()
    #     output = self._build_network_for_input(x, apply_dropout=False)
    #     prob = math.exp(output)
    #     predictions = {}
    #     for label, scores in output.items():
    #         logprobs = list(scores.npvalue())
    #         probs = [math.exp(lp) for lp in logprobs]
    #         assert sum(probs) > 0.99 and sum(probs) < 1.01, 'bad probs: ' + str(sum(probs))
    #         dist = {self.pss_vocab.get_word(ind): p for ind, p in enumerate(probs)}
    #         predictions[label] = dist
    #     return predictions
    #
    def save(self, base_path):
        def pythonize_embds(embds):
            return {k: [float(x) for x in list(v)] for k, v in embds.items()}

        self.pc.save(base_path)
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
            with open(base_path + '.hp', 'r') as hp_f:
                with open(base_path + '.vocabs', 'r') as in_vocabs_f:
                        with open(base_path + '.tok_embds', 'r') as embds_f:
                            model = PairwiseFuncClustModel(
                                hyperparameters=PairwiseFuncClustModel.HyperParameters(**json.load(hp_f)),
                                token_embeddings=json.load(embds_f),
                                **json.load(in_vocabs_f)
                            )
                            model.pc.populate(base_path)
                            return model
        finally:
            files = glob(base_path + ".*") + [base_path]
            for fname in files:
                if os.path.realpath(fname) != os.path.realpath(base_path + ".zip"):
                    print("loading..", fname)
                    os.remove(fname)
