import json
import math
import os
import random
import zipfile
from collections import namedtuple
from glob import glob
from pprint import pprint

import dynet as dy
import numpy as np

from dynet_utils import get_activation_function
from evaluators.simple_pss_classifier_evaluator import SimplePSSClassifierEvaluator
from models.supersenses_simple import embeddings
from models.supersenses_simple import vocabs

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'mlp_softmaxes',
])

MLPSoftmaxParam = namedtuple('MLPSoftmax', ['mlp', 'softmax'])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class SimpleMlpSupersensesModel:

    class SampleX:

        def __init__(self,
                     prep_tokens,
                     gov_token,
                     obj_token):
            self.prep_tokens = prep_tokens
            self.gov_token = gov_token
            self.obj_token = obj_token

        def to_dict(self):
            return self.__dict__

        def pprint(self):
            print('[prep:%s,gov:%s,obj:%s]' % (' '.join(self.prep_tokens), self.gov_token, self.obj_token))

        @staticmethod
        def from_dict(d):
            return SimpleMlpSupersensesModel.SampleX(**d)

        def __repr__(self):
            return repr(self.prep_tokens)

    class SampleY:

        def __init__(self, supersense_role=None, supersense_func=None):
            self.supersense_role = supersense_role
            self.supersense_func = supersense_func
            assert supersense_role or supersense_func

        def to_dict(self):
            return self.__dict__

        def is_empty(self):
            return not self.supersense_func and not self.supersense_role

        def pprint(self):
            print('[role:%s,func:%s]' % (self.supersense_role, self.supersense_func))

        @staticmethod
        def from_dict(d):
            return SimpleMlpSupersensesModel.SampleY(**d)

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
            return SimpleMlpSupersensesModel.Sample(
                x=SimpleMlpSupersensesModel.SampleX.from_dict(d['x']),
                y=SimpleMlpSupersensesModel.SampleY.from_dict(d['y']),
                sample_id=d['sample_id']
            )

        def pprint(self):
            print('---')
            self.x.pprint()
            self.y.pprint()

    SUPERSENSE_ROLE = "supersense_role"
    SUPERSENSE_FUNC = "supersense_func"

    class HyperParameters:

        def __init__(self,
                     labels_to_predict,
                     use_prep,
                     use_gov,
                     use_obj,
                     update_prep_embd,
                     update_gov_embd,
                     update_obj_embd,
                     token_embd_dim,
                     internal_token_embd_dim,
                     mlp_layers,
                     mlp_layer_dim,
                     mlp_activation,
                     lstm_h_dim,
                     num_lstm_layers,
                     is_bilstm,
                     mlp_dropout_p,
                     lstm_dropout_p,
                     epochs,
                     learning_rate,
                     learning_rate_decay,
                     dynet_random_seed
                     ):
            self.internal_token_embd_dim = internal_token_embd_dim
            self.labels_to_predict = labels_to_predict
            self.use_prep = use_prep
            self.use_gov = use_gov
            self.use_obj = use_obj
            self.update_prep_embd = update_prep_embd
            self.update_gov_embd = update_gov_embd
            self.update_obj_embd = update_obj_embd
            self.token_embd_dim = token_embd_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.mlp_activation = mlp_activation
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_dropout_p = mlp_dropout_p
            self.lstm_dropout_p = lstm_dropout_p
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.learning_rate_decay = learning_rate_decay
            self.dynet_random_seed = dynet_random_seed
            assert all([label in [SimpleMlpSupersensesModel.SUPERSENSE_FUNC, SimpleMlpSupersensesModel.SUPERSENSE_ROLE] for label in (labels_to_predict or [])])

        def clone(self, override=None):
            override = override or {}
            params = self.__dict__
            params.update(override)
            return SimpleMlpSupersensesModel.HyperParameters(**params)

    def __init__(self, hyperparameters, gov_vocab=None, obj_vocab=None, prep_vocab=None, pss_vocab=None, token_embeddings=None):
        self.prep_vocab = prep_vocab or vocabs.PREPS
        self.obj_vocab = obj_vocab or vocabs.OBJ
        self.gov_vocab = gov_vocab or vocabs.GOV
        self.pss_vocab = pss_vocab or vocabs.PSS
        self.embeddings = token_embeddings or embeddings.TOKENS_WORD2VEC
        self.hyperparameters = hyperparameters

        print("SimpleMlpSupersensesModel: Building model with the following hyperparameters:")
        pprint(hyperparameters.__dict__)
        self.pc = self._build_network_params()

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters
        mlp_input_dim = (self.hyperparameters.lstm_h_dim  if hp.use_prep else 0) + \
                        (hp.token_embd_dim if hp.use_gov else 0) + \
                        (hp.token_embd_dim if hp.use_obj else 0)
        print('mlp input dim', mlp_input_dim)
        self.params = ModelOptimizedParams(
            input_lookups={
                "gov": pc.add_lookup_parameters((self.gov_vocab.size(), hp.token_embd_dim)),
                "obj": pc.add_lookup_parameters((self.obj_vocab.size(), hp.token_embd_dim)),
                "prep": pc.add_lookup_parameters((self.prep_vocab.size(), hp.token_embd_dim)),
                "prep_internal": pc.add_lookup_parameters((self.prep_vocab.size(), hp.internal_token_embd_dim))
            },
            mlp_softmaxes={
              pss_type: MLPSoftmaxParam(
                  mlp=[MLPLayerParams(
                      W=pc.add_parameters((self.hyperparameters.mlp_layer_dim, self.hyperparameters.mlp_layer_dim if i > 0 else mlp_input_dim)),
                      b=pc.add_parameters((self.hyperparameters.mlp_layer_dim,))
                  ) for i in range(self.hyperparameters.mlp_layers)],
                  softmax=MLPLayerParams(
                      W=pc.add_parameters((self.pss_vocab.size(), self.hyperparameters.mlp_layer_dim)),
                      b=pc.add_parameters((self.pss_vocab.size(),))
                  )
              ) for pss_type in self.hyperparameters.labels_to_predict
            }
        )

        input_vocabs = {
            'gov': self.gov_vocab,
            'obj': self.obj_vocab,
            'prep': self.prep_vocab
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

    def _build_network_for_input(self, x, apply_dropout):
        if apply_dropout:
            mlp_dropout_p = self.hyperparameters.mlp_dropout_p
            lstm_dropout_p = self.hyperparameters.lstm_dropout_p
        else:
            mlp_dropout_p = 0
            lstm_dropout_p = 0

        self.lstm_builder.set_dropout(lstm_dropout_p)

        if self.hyperparameters.use_prep:
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
                for tok in x.prep_tokens
            ]
            lstm_output = cur_lstm_state.transduce(embeddings)[-1]
            vecs = [lstm_output]
        else:
            vecs = []
        if self.hyperparameters.use_gov:
            cur_out = dy.concatenate(vecs + [dy.lookup(self.params.input_lookups['gov'], self.gov_vocab.get_index(x.gov_token))])
        if self.hyperparameters.use_obj:
            cur_out = dy.concatenate(vecs + [dy.lookup(self.params.input_lookups['obj'], self.obj_vocab.get_index(x.obj_token))])
        if not self.hyperparameters.use_gov and not self.hyperparameters.use_obj:
            cur_out = dy.concatenate(vecs)
        mlp_activation = get_activation_function(self.hyperparameters.mlp_activation)
        output = {}
        for label, mlp_softmax in self.params.mlp_softmaxes.items():
            mlp_cur_out = cur_out
            for mlp_layer_params in mlp_softmax.mlp:
                mlp_cur_out = dy.dropout(mlp_cur_out, mlp_dropout_p)
                mlp_cur_out = mlp_activation(dy.parameter(mlp_layer_params.W) * mlp_cur_out + dy.parameter(mlp_layer_params.b))
            mlp_cur_out = dy.log_softmax(dy.parameter(mlp_softmax.softmax.W) * mlp_cur_out + dy.parameter(mlp_softmax.softmax.b))
            output[label] = mlp_cur_out
        return output

    def _build_loss(self, output, y):
        losses = []
        for label, scores in output.items():
            ss_ind = self.pss_vocab.get_index(getattr(y, label))
            loss = -dy.pick(scores, ss_ind)
            losses.append(loss)
        return dy.esum(losses)

    def fit(self, samples, validation_samples, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        evaluator = evaluator or SimplePSSClassifierEvaluator()
        self.pc = self._build_network_params()
        test = validation_samples
        train = samples

        self.test_set_evaluation = []
        self.train_set_evaluation = []

        best_test_acc = None
        train_acc = None
        best_epoch = None
        model_file_path = '/tmp/_m_' + str(random.randrange(10000))
        try:
            trainer = dy.SimpleSGDTrainer(self.pc, learning_rate=self.hyperparameters.learning_rate)
            for epoch in range(1, self.hyperparameters.epochs + 1):
                if np.isinf(trainer.learning_rate):
                    break

                train = list(train)
                random.shuffle(train)
                loss_sum = 0

                BATCH_SIZE = 20
                batches = [train[batch_ind::int(math.ceil(len(train)/BATCH_SIZE))] for batch_ind in range(int(math.ceil(len(train)/BATCH_SIZE)))]
                for batch_ind, batch in enumerate(batches):
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
                        batch_loss.backward()
                        loss_sum += batch_loss.value()
                        trainer.update()
                    if show_progress:
                        if int((batch_ind + 1) / len(batches) * 100) > int(batch_ind / len(batches) * 100):
                            per = int((batch_ind + 1) / len(batches) * 100)
                            print('\r\rEpoch %3d (%d%%): |' % (epoch, per) + '#' * per + '-' * (100 - per) + '|',)
                if self.hyperparameters.learning_rate_decay:
                    trainer.learning_rate /= (1 - self.hyperparameters.learning_rate_decay)

                if evaluator and show_epoch_eval:
                    print('--------------------------------------------')
                    print('Epoch %d complete, avg loss: %1.4f' % (epoch, loss_sum/len(train)))
                    print('Validation data evaluation:')
                    epoch_test_eval = evaluator.evaluate(test, examples_to_show=5, predictor=self)
                    self.test_set_evaluation.append(epoch_test_eval)
                    print('Training data evaluation:')
                    epoch_train_eval = evaluator.evaluate(train, examples_to_show=5, predictor=self)
                    self.train_set_evaluation.append(epoch_train_eval)
                    print('--------------------------------------------')

                    get_acc = lambda ev: sum([ev[label] for label in self.hyperparameters.labels_to_predict]) / len(self.hyperparameters.labels_to_predict)

                    test_acc = get_acc(epoch_test_eval)
                    if best_test_acc is None or test_acc > best_test_acc:
                        print("Best epoch so far! with f1 of: %1.2f" % test_acc)
                        best_test_acc = test_acc
                        train_acc = get_acc(epoch_train_eval)
                        best_epoch = epoch
                        self.pc.save(model_file_path)

            self.train_acc = train_acc
            self.test_acc = best_test_acc
            self.best_epoch = best_epoch
            print('--------------------------------------------')
            print('Training is complete (%d samples, %d epochs)' % (len(train), self.hyperparameters.epochs))
            print('--------------------------------------------')
            self.pc.populate(model_file_path)
        finally:
            if os.path.exists(model_file_path):
                os.remove(model_file_path)

        return self

    def predict(self, sample_x):
        dy.renew_cg()
        output = self._build_network_for_input(sample_x, apply_dropout=False)
        prediction = {}
        for label, scores in output.items():
            ind = np.argmax(scores.npvalue())
            predicted = self.pss_vocab.get_word(ind)
            prediction[label] = predicted
        return SimpleMlpSupersensesModel.SampleY(**prediction)

    def predict_dist(self, x):
        dy.renew_cg()
        outputs = self._build_network_for_input(x, apply_dropout=False)
        predictions = {}
        for label, scores in output.items():
            logprobs = list(scores.npvalue())
            probs = [math.exp(lp) for lp in logprobs]
            assert sum(probs) > 0.99 and sum(probs) < 1.01, 'bad probs: ' + str(sum(probs))
            dist = {self.pss_vocab.get_word(ind): p for ind, p in enumerate(probs)}
            predictions[label] = dist
        return predictions

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
                "prep_vocab": self.prep_vocab,
                "pss_vocab": self.pss_vocab
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
                            model = SimpleMlpSupersensesModel(
                                hyperparameters=SimpleMlpSupersensesModel.HyperParameters(**json.load(hp_f)),
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
