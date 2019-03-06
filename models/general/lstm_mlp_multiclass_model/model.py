import json
import math
import os
import random
import zipfile
from collections import namedtuple
from glob import glob

import dynet as dy
import numpy as np

from dynet_utils import get_activation_function
from vocabulary import Vocabulary

# There are more hidden parameters coming from the LSTMs

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'mlps',
    'softmaxes',
    'unknown_local',
    'unknown_mlp_inputs'
])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class LstmMlpMulticlassModel(object):

    Sample = namedtuple('Sample', ['xs', 'ys', 'mask'])

    class SampleX:
        def __init__(self, fields, hidden, neighbors=None, embeddings_override=None, attrs=None):
            self.hidden = hidden
            self.fields = fields
            self.neighbors = neighbors or {}
            self.embeddings_override = embeddings_override or {}
            self.attrs = attrs or {}

        def __getitem__(self, field):
            return self.fields[field]

        def items(self):
            return self.fields.items()

        def __iter__(self):
            return self.fields.__iter__()

        def keys(self):
            return self.fields.keys()

    # SampleY - List of labels

    class HyperParameters:
        def __init__(self,
                     lstm_input_fields,
                     token_neighbour_types,
                     input_embeddings_to_allow_partial,
                     input_embeddings_to_update,
                     input_embedding_dims,
                     input_embeddings_default_dim,
                     mlp_input_fields,
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
                     n_labels_to_learn,
                     label_inds_to_predict,
                     dynet_random_seed,
                     use_local,
                     local_dropout_p,
                     mlp_input_dropouts):
            self.mlp_input_dropouts = mlp_input_dropouts
            self.local_dropout_p = local_dropout_p
            self.use_local = use_local
            self.label_inds_to_predict = label_inds_to_predict
            self.dynet_random_seed = dynet_random_seed
            self.input_embeddings_to_allow_partial = input_embeddings_to_allow_partial
            self.lstm_dropout_p = lstm_dropout_p
            self.token_neighbour_types = token_neighbour_types
            self.n_labels_to_learn = n_labels_to_learn
            self.mlp_input_fields = mlp_input_fields
            self.learning_rate_decay = learning_rate_decay
            self.learning_rate = learning_rate
            self.lstm_input_fields = lstm_input_fields
            self.input_embeddings_to_update = input_embeddings_to_update
            self.input_embedding_dims = input_embedding_dims
            self.input_embeddings_default_dim = input_embeddings_default_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.mlp_activation = mlp_activation
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_dropout_p = mlp_dropout_p
            self.epochs = epochs

    def __init__(self,
                 input_vocabularies=None,
                 output_vocabulary=None,
                 input_embeddings=None,
                 hyperparameters=None):

        self.input_vocabularies = input_vocabularies
        self.output_vocabulary = output_vocabulary
        self.input_embeddings = input_embeddings or {}

        # input_embeddings_dims = {field: hyperparameters.input_embeddings_default_dim for field in hyperparameters.lstm_input_fields}
        # input_embeddings_dims.update({k: v for (k,v) in (hyperparameters.input_embedding_dims or {}).items() if k in hyperparameters.lstm_input_fields})
        # input_embeddings_dims.update({
        #     field: len(list(self.input_embeddings[field].values())[0])
        #     for field in hyperparameters.lstm_input_fields
        #     if self.input_embeddings.get(field)
        # })
        # self.hyperparameters = LstmMlpMulticlassModel.HyperParameters(**update_dict(hyperparameters.__dict__, {'input_embedding_dims': input_embeddings_dims}))
        self.hyperparameters = hyperparameters
        self.test_set_evaluation = None
        self.dev_set_evaluation = None
        self.train_set_evaluation = None
        self.embds_to_randomize = []
        self.missing_embd_count = 0
        self.existing_embd_count = 0
        self.reset_embd_counts()
        self.pc = self._build_network_params()

        self.validate_params()

    def reset_embd_counts(self):
        self.missing_embd_count = 0
        self.existing_embd_count = 0

    def report_embd_counts(self):
        print("Missing embeddings: %d, existing embeddings: %d, missing percentage: %d" % (self.missing_embd_count, self.existing_embd_count, self.missing_embd_count/(self.missing_embd_count + self.existing_embd_count)*100))

    @property
    def all_input_fields(self):
        return self.hyperparameters.lstm_input_fields + self.hyperparameters.mlp_input_fields

    def validate_params(self):
        # Make sure input embedding dimensions fit embedding vectors size (if given)
        for field in self.all_input_fields:
            if self.input_embeddings.get(field):
                embd_vec_dim = self.input_embeddings[field].dim() if "dim" in dir(self.input_embeddings[field]) else len(list(self.input_embeddings[field].values())[0])
                given_dim = self.hyperparameters.input_embedding_dims[field]
                if embd_vec_dim != given_dim:
                    raise Exception("Input field '%s': Mismatch between given embedding vector size (%d) and given embedding size (%d)" % (field, embd_vec_dim, given_dim))

    def get_field_dim(self, field):
        if field in self.hyperparameters.token_neighbour_types:
            return self.hyperparameters.lstm_h_dim
        else:
            return self.get_embd_dim(field)

    def get_embd_dim(self, field):
        dim = None
        if field in self.hyperparameters.input_embedding_dims:
            dim = self.hyperparameters.input_embedding_dims[field]
        elif self.input_embeddings.get(field):
            for vec in self.input_embeddings[field].values():
                dim = len(vec)
                break
        else:
            dim = self.hyperparameters.input_embeddings_default_dim
        assert self.hyperparameters.input_embedding_dims.get(field, dim) == dim
        assert dim is not None, 'Unable to resolve embeddings dimensions for field: ' + field
        return dim

    def _build_network_params(self):
        pc = dy.ParameterCollection()

        embedded_input_dim = sum([self.get_embd_dim(field) for field in self.hyperparameters.lstm_input_fields])

        if self.hyperparameters.num_lstm_layers > 0:
            ctx_input_dim = self.hyperparameters.lstm_h_dim
        else:
            ctx_input_dim = embedded_input_dim

        mlp_input_dim = 0
        if self.hyperparameters.use_local:
            mlp_input_dim += ctx_input_dim
        mlp_input_dim += sum([self.get_embd_dim(field) for field in self.hyperparameters.mlp_input_fields])
        mlp_input_dim += ctx_input_dim * len(self.hyperparameters.token_neighbour_types)

        if self.hyperparameters.mlp_layers > 0:
            mlp_output_dim = self.hyperparameters.mlp_layer_dim
        else:
            mlp_output_dim = mlp_input_dim

        self.params = ModelOptimizedParams(
            input_lookups={
                field: pc.add_lookup_parameters((self.input_vocabularies[field].size(), self.get_embd_dim(field)))
                for field in (self.hyperparameters.lstm_input_fields + self.hyperparameters.mlp_input_fields)
            },
            mlps=[
                [
                    MLPLayerParams(
                        W=pc.add_parameters((self.hyperparameters.mlp_layer_dim, self.hyperparameters.mlp_layer_dim if i > 0 else mlp_input_dim)),
                        b=pc.add_parameters((self.hyperparameters.mlp_layer_dim,))
                    )
                    for i in range(self.hyperparameters.mlp_layers)
                ]
                for _ in range(self.hyperparameters.n_labels_to_learn)
            ],
            softmaxes=[
                MLPLayerParams(
                        W=pc.add_parameters((self.output_vocabulary.size(), mlp_output_dim)),
                        b=pc.add_parameters((self.output_vocabulary.size(),))
                )
                for _ in range(self.hyperparameters.n_labels_to_learn)
            ],
            unknown_local=pc.add_parameters((ctx_input_dim,)),
            unknown_mlp_inputs={
                field: pc.add_parameters((self.get_field_dim(field),))
                for field in (self.hyperparameters.mlp_input_fields + self.hyperparameters.token_neighbour_types)
            }
        )

        for field, lookup_param in sorted(self.params.input_lookups.items()):
            embeddings = (self.input_embeddings or {}).get(field)
            if embeddings:
                vocab = self.input_vocabularies[field]
                for word in vocab.all_words():
                    if word is not None:
                        word_index = self.input_vocabularies[field].get_index(word)
                        vector = embeddings.get(word)
                        if vector is None:
                            vector = embeddings.get(word.lower())
                        if vector is not None:
                            lookup_param.init_row(word_index, vector)
                        else:
                            if field not in self.hyperparameters.input_embeddings_to_allow_partial:
                                raise Exception('Missing embedding vector for field: %s, word %s' % (field, word))
                            lookup_param.init_row(word_index, [0] * self.get_embd_dim(field))

        if self.hyperparameters.num_lstm_layers > 0:
            if self.hyperparameters.is_bilstm:
                self.lstm_builder = dy.BiRNNBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc, dy.LSTMBuilder)
            else:
                self.lstm_builder = dy.LSTMBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc)
        else:
            self.lstm_builder = None

        return pc

    def build_mask(self, xs, external_mask=None):
        external_mask = external_mask or [True for _ in xs]
        mask = list(external_mask)
        # for ind, x in enumerate(xs):
        #     if mask[ind]:
        #         for field in self.hyperparameters.mlp_input_fields:
        #             if not self.input_vocabularies[field].has_word(x[field]):
        #                 mask[ind] = False
        return mask

    def _validate_xs(self, xs, mask):
        for x, x_mask in zip(xs, mask):
            for neighbor_type in x.neighbors:
                if neighbor_type not in self.hyperparameters.token_neighbour_types:
                    raise Exception("Unknown dep type:" + neighbor_type)
            if x_mask:
                for neighbor_type in self.hyperparameters.token_neighbour_types:
                    if neighbor_type not in x.neighbors:
                        raise Exception("X without a dep:" + neighbor_type)

    def get_embd(self, token_data, field):
        if field in self.embds_to_randomize and token_data[field]:
            tmp_lp = self.pc.add_lookup_parameters(
                (1, self.get_embd_dim(field))
            )
            self.existing_embd_count += 1
            return dy.inputTensor(dy.lookup(tmp_lp, 0).npvalue())

        embd = None
        _embd = token_data.embeddings_override.get(field)
        if _embd:
            embd = dy.inputTensor(_embd)
        else:
            if self.input_vocabularies[field].has_word(token_data[field]):
                embd = dy.lookup(
                    self.params.input_lookups[field],
                    self.input_vocabularies[field].get_index(token_data[field]),
                    update=not self.input_embeddings.get(field) or self.hyperparameters.input_embeddings_to_update.get(field)
                )
            elif self.input_embeddings.get(field, {}).get(token_data[field]):
                embd = dy.inputTensor(self.input_embeddings.get(field, {}).get(token_data[field]))

            if embd is not None and not any(list(embd.npvalue())):
                embd = None

            if embd is None:
                if field not in self.hyperparameters.input_embeddings_to_allow_partial:
                    raise Exception('Missing embedding vector for field: %s, word %s' % (field, token_data[field]))
                if self.input_vocabularies[field].has_word(None):
                    self.missing_embd_count += 1
                    self.existing_embd_count -= 1
                    embd = dy.lookup(
                        self.params.input_lookups[field],
                        self.input_vocabularies[field].get_index(None),
                        update=True
                    )

        if embd is None:
            self.missing_embd_count += 1
            embd = dy.inputTensor([0] * self.get_embd_dim(field))
        else:
            assert len(embd.npvalue()) == self.get_embd_dim(field)
            self.existing_embd_count += 1

        return embd


    def _build_network_for_input(self, xs, mask, apply_dropout):
        self._validate_xs(xs, mask)

        if apply_dropout:
            mlp_dropout_p = self.hyperparameters.mlp_dropout_p
            lstm_dropout_p = self.hyperparameters.lstm_dropout_p
        else:
            mlp_dropout_p = 0
            lstm_dropout_p = 0

        mask = self.build_mask(xs, external_mask=mask)
        embeddings = [
            dy.concatenate([
                self.get_embd(token_data, field)
                for field in self.hyperparameters.lstm_input_fields
            ]) if not token_data.hidden and
                  not(
                          mask and mask[ind] and (
                          not self.hyperparameters.use_local
                          or (
                                  apply_dropout and self.hyperparameters.local_dropout_p is not None
                                  and random.random() < self.hyperparameters.local_dropout_p
                          )
                  )
                  )
            else None
            for ind, token_data in enumerate(xs)
        ]
        assert max([ind for ind, x in enumerate(xs) if not x.hidden]) < min([ind for ind, x in enumerate(xs) if x.hidden] + [len(xs)])

        if self.lstm_builder:
            self.lstm_builder.set_dropout(lstm_dropout_p)
            if not self.hyperparameters.is_bilstm:
                cur_lstm_state = self.lstm_builder.initial_state()
            else:
                cur_lstm_state = self.lstm_builder

            lstm_outputs = cur_lstm_state.transduce([e for e in embeddings if e is not None])
            lstm_outputs_and_hidden = []
            oind = 0
            for e in embeddings:
                if e is None:
                    lstm_outputs_and_hidden.append(None)
                else:
                    lstm_outputs_and_hidden.append(lstm_outputs[oind])
                    oind += 1
            assert oind == len(lstm_outputs)
            ctx_inputs = lstm_outputs_and_hidden
        else:
            ctx_inputs = embeddings

        mlp_activation = get_activation_function(self.hyperparameters.mlp_activation)
        outputs = []
        for ind, ctx_input in enumerate(ctx_inputs):
            if mask and not mask[ind]:
                output = None
            else:
                vecs = []
                if self.hyperparameters.use_local:
                    v = ctx_input if ctx_input is not None else dy.parameter(self.params.unknown_local)
                    vecs.append(v)
                inp_token = xs[ind]
                for neighbour_type in self.hyperparameters.token_neighbour_types:
                    neighbour_ind = xs[ind].neighbors.get(neighbour_type)
                    if neighbour_ind is None or ctx_inputs[neighbour_ind] is None or (apply_dropout and random.random() < self.hyperparameters.mlp_input_dropouts.get(neighbour_type, -1)):
                        vecs.append(dy.parameter(self.params.unknown_mlp_inputs[neighbour_type]))
                    else:
                        vecs.append(ctx_inputs[neighbour_ind])

                vecs.extend([dy.lookup(
                                 self.params.input_lookups[field],
                                 self.input_vocabularies[field].get_index(inp_token[field]),
                                 update=self.hyperparameters.input_embeddings_to_update.get(field) or False
                             ) if inp_token[field] is not None and self.input_vocabularies[field].has_word(inp_token[field]) and (not apply_dropout or random.random() > self.hyperparameters.mlp_input_dropouts.get(field, -1)) else dy.parameter(self.params.unknown_mlp_inputs[field]) for field in self.hyperparameters.mlp_input_fields])
                cur_out = dy.concatenate(vecs)
                output = []
                for mlp_params, softmax_params in zip(self.params.mlps, self.params.softmaxes):
                    mlp_cur_out = cur_out
                    for mlp_layer_params in mlp_params:
                        mlp_cur_out = dy.dropout(mlp_cur_out, mlp_dropout_p)
                        try:
                            mlp_cur_out = mlp_activation(dy.parameter(mlp_layer_params.W) * mlp_cur_out + dy.parameter(mlp_layer_params.b))
                        except:
                            raise
                    mlp_cur_out = dy.log_softmax(dy.parameter(softmax_params.W) * mlp_cur_out + dy.parameter(softmax_params.b))
                    output.append(mlp_cur_out)
            outputs.append(output)
        return outputs

    def _build_loss(self, outputs, ys):
        losses = []
        for out, y in zip(outputs, ys):
            if out is not None:
                if y is None:
                    y = [None] * self.hyperparameters.n_labels_to_learn
                assert len([label_y is None for label_y in y]) in [0, len(y)], "Got a sample with partial None labels"
                for label_out, label_y in zip(out, y):
                    if self.output_vocabulary.has_word(label_y):
                        ss_ind = self.output_vocabulary.get_index(label_y)
                        loss = -dy.pick(label_out, ss_ind)
                        losses.append(loss)
                    else:
                        assert label_y is None
        if len(losses):
            loss = dy.esum(losses)
        else:
            loss = None
        return loss

    # def _build_vocabularies(self, samples):
    #     if not self.input_vocabularies:
    #         self.input_vocabularies = {}
    #     for field in self.all_input_fields:
    #         if not self.input_vocabularies.get(field):
    #             vocab = Vocabulary(field)
    #             vocab.add_words([x.fields.get(field) for s in samples for x in s.xs])
    #             self.input_vocabularies[field] = vocab
    #
    #     if not self.output_vocabulary:
    #         vocab = Vocabulary('output')
    #         vocab.add_words([y for s in samples for y in s.ys])
    #         self.output_vocabulary = vocab
    #
    def fit(self, samples, validation_samples=None, test_samples=None, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        self.pc = self._build_network_params()
        # self._build_vocabularies(samples + validation_samples or [])

        test = test_samples
        dev = validation_samples
        train = samples

        self.test_set_evaluation = []
        self.dev_set_evaluation = []
        self.train_set_evaluation = []

        best_dev_acc = None
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
                        outputs = self._build_network_for_input(sample.xs, sample.mask, apply_dropout=True)
                        sample_loss = self._build_loss(outputs, sample.ys)
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
                    epoch_dev_eval = evaluator.evaluate(dev, examples_to_show=0, predictor=self, inds_to_predict=self.hyperparameters.label_inds_to_predict)
                    self.dev_set_evaluation.append(epoch_dev_eval)
                    print('Training data evaluation:')
                    epoch_train_eval = evaluator.evaluate(train, examples_to_show=0, predictor=self, inds_to_predict=self.hyperparameters.label_inds_to_predict)
                    self.train_set_evaluation.append(epoch_train_eval)
                    print('Testing data evaluation:')
                    self.embds_to_randomize = []
                    # self.embds_to_randomize = []
                    epoch_test_eval = evaluator.evaluate(test, examples_to_show=0, predictor=self, inds_to_predict=self.hyperparameters.label_inds_to_predict)
                    self.test_set_evaluation.append(epoch_test_eval)
                    self.embds_to_randomize = []
                    print('--------------------------------------------')

                    dev_acc = epoch_dev_eval['f1']
                    if dev_acc is not None and (best_dev_acc is None or dev_acc > best_dev_acc):
                        print("Best epoch so far! with f1 of: %1.2f" % dev_acc)
                        best_dev_acc = dev_acc
                        # train_acc = epoch_train_eval['f1']
                        best_epoch = epoch
                        self.pc.save(model_file_path)

            print('--------------------------------------------')
            print('Training is complete (%d samples, %d epochs)' % (len(train), self.hyperparameters.epochs))
            print('--------------------------------------------')
            self.pc.populate(model_file_path)
        finally:
            if os.path.exists(model_file_path):
                os.remove(model_file_path)

        return self

    def predict(self, sample_xs, mask=None):
        dy.renew_cg()
        if mask is None:
            mask = [True] * len(sample_xs)
        outputs = self._build_network_for_input(sample_xs, mask, apply_dropout=False)
        ys = []
        for token_ind, out in enumerate(outputs):
            if not mask[token_ind] or out is None:
                predictions = [None] * len(self.hyperparameters.label_inds_to_predict)
            else:
                predictions = []
                for label_ind, klass_out in enumerate(out):
                    if label_ind in self.hyperparameters.label_inds_to_predict:
                        ind = np.argmax(klass_out.npvalue())
                        predicted = self.output_vocabulary.get_word(ind) if mask[token_ind] else None
                        predictions.append(predicted)
            predictions = tuple(predictions)
            ys.append(predictions)
        assert all([y is None or type(y) is tuple and len(y) == len(self.hyperparameters.label_inds_to_predict) for y in ys])
        assert len(ys) == len(sample_xs)
        return ys

    def predict_dist(self, sample_xs, mask=None):
        dy.renew_cg()
        if mask is None:
            mask = [True] * len(sample_xs)
        outputs = self._build_network_for_input(sample_xs, mask, apply_dropout=False)
        ys = []
        for token_ind, out in enumerate(outputs):
            if not mask[token_ind] or out is None:
                predictions = tuple([None] * self.hyperparameters.n_labels_to_learn)
            else:
                predictions = []
                for klass_out in out:
                    logprobs = list(klass_out.npvalue())
                    probs = [math.exp(lp) for lp in logprobs]
                    assert sum(probs) > 0.99 and sum(probs) < 1.01, 'bad probs: ' + str(sum(probs))
                    dist = {self.output_vocabulary.get_word(ind): p for ind, p in enumerate(probs)}
                    predictions.append(dist)
                predictions = tuple(predictions)
            ys.append(predictions)
        assert all([y is None or type(y) is tuple and len(y) == self.hyperparameters.n_labels_to_learn for y in ys])
        return ys

    def save(self, base_path):
        def pythonize_embds(embds):
            return {k: [float(x) for x in list(v)] for k, v in embds.items()}

        self.pc.save(base_path)
        with open(base_path + '.hp', 'w') as f:
            json.dump(vars(self.hyperparameters), f, indent=2)
        input_vocabularies = {
            name: vocab.pack() for name, vocab in self.input_vocabularies.items()
        }
        with open(base_path + '.in_vocabs', 'w') as f:
            json.dump(input_vocabularies, f)
        output_vocabulary = self.output_vocabulary.pack()
        with open(base_path + '.out_vocab', 'w') as f:
            json.dump(output_vocabulary, f)
        # with open(base_path + '.embds', 'w') as f:
        #     json.dump({name: pythonize_embds(embds) for name, embds in self.input_embeddings.items()}, f)

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
    def load(base_path, embds):
        with zipfile.ZipFile(base_path + ".zip", "r") as zh:
            zh.extractall(os.path.dirname(base_path))
        try:
            with open(base_path + '.hp', 'r') as hp_f:
                with open(base_path + '.in_vocabs', 'r') as in_vocabs_f:
                    with open(base_path + '.out_vocab', 'r') as out_vocabs_f:
                        # with open(base_path + '.embds', 'r') as embds_f:
                        model = LstmMlpMulticlassModel(
                            input_vocabularies={name: Vocabulary.unpack(packed) for name, packed in json.load(in_vocabs_f).items()},
                            output_vocabulary=Vocabulary.unpack(json.load(out_vocabs_f)),
                            input_embeddings=embds,
                            hyperparameters=LstmMlpMulticlassModel.HyperParameters(**json.load(hp_f))
                        )
                        model.pc.populate(base_path)
                        return model
        finally:
            files = glob(base_path + ".*") + [base_path]
            for fname in files:
                if os.path.realpath(fname) != os.path.realpath(base_path + ".zip"):
                    print("loading..", fname)
                    os.remove(fname)


    def randomize_embeddings(self, fields, include_none=False):
        for field in fields:
            tmp_lp = self.pc.add_lookup_parameters(
                self.input_vocabularies[field].size(),
                self.get_embd_dim(field)
            )
            lookup = self.params.input_lookups[field]
            vocab = self.input_vocabularies[field]
            for w in vocab:
                if w is None and not include_none:
                    continue
                w_ind = vocab.get_index(w)
                lookup.init_row(w_ind,
                                dy.lookup(tmp_lp, w_ind).npvalue())
