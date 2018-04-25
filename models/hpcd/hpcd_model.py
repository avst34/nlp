import zipfile
import os
from collections import namedtuple

import random
from glob import glob

import dynet as dy
import json
import numpy as np
import math

# There are more hidden parameters coming from the LSTMs
import zlib

from dynet_utils import get_activation_function
from utils import update_dict
from vocabulary import Vocabulary

SingleLayerParams = namedtuple('SingleLayerParams', [
    'W',
    'b'
])

class ModelOptimizedParams:

    def __init__(self, word_embeddings, p1_layer, p2_layers, w):
        self.p1_layer = p1_layer
        self.p2_layers = p2_layers
        self.w = w
        self.word_embeddings = word_embeddings


MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class HCPDModel(object):

    Sample = namedtuple('Sample', ['xs', 'ys', 'mask'])

    class HeadCand:
        def __init__(self, word, pp_distance, is_verb, is_noun, next_pos, hypernyms, is_pp_in_verbnet_frame):
            self.next_pos = next_pos
            self.hypernyms = hypernyms
            self.is_pp_in_verbnet_frame = is_pp_in_verbnet_frame
            self.pp_distance = pp_distance
            self.word = word
            self.is_verb = is_verb
            self.is_noun = is_noun
            assert self.is_verb != self.is_noun
    PP = namedtuple('PP', ['word'])
    Child = namedtuple('PP', ['word', 'hypernyms'])

    class SampleX:
        def __init__(self, heand_cands, pp, child):
            self.heand_cands = heand_cands
            self.pp = pp
            self.child = child

    class SampleY:
        def __init__(self, correct_head_cand, scored_heads):
            self.correct_head_cand = correct_head_cand
            self.scored_heads = scored_heads

    class HyperParameters:
        def __init__(self, 
                     max_head_distance=5,
                     dropout_p=0.5
                     ):
            self.max_head_distance = max_head_distance
            self.dropout_p = dropout_p

    def __init__(self,
                 words_vocab=None,
                 pos_vocab=None,
                 word_embeddings=None,
                 wordnet_hypernyms_vocab=None,
                 hyperparameters=None):

        self.wordnet_hypernyms_vocab = wordnet_hypernyms_vocab
        self.words_vocab = words_vocab
        self.pos_vocab = pos_vocab
        self.word_embeddings = word_embeddings

        self.hyperparameters = hyperparameters
        self.test_set_evaluation = None
        self.train_set_evaluation = None
        self.pc = self._build_network_params()

        self.validate_params()


    @property
    def all_input_fields(self):
        return self.hyperparameters.lstm_input_fields + self.hyperparameters.mlp_input_fields

    def validate_params(self):
        # Make sure input embedding dimensions fit embedding vectors size (if given)
        for field in self.all_input_fields:
            if self.input_embeddings.get(field):
                embd_vec_dim = len(list(self.input_embeddings[field].values())[0])
                given_dim = self.hyperparameters.input_embedding_dims[field]
                if embd_vec_dim != given_dim:
                    raise Exception("Input field '%s': Mismatch between given embedding vector size (%d) and given embedding size (%d)" % (field, embd_vec_dim, given_dim))

    def get_embd_dim(self, field):
        dim = None
        if self.input_embeddings.get(field):
            for vec in self.input_embeddings[field].values():
                dim = len(vec)
                break
        elif field in self.hyperparameters.input_embedding_dims:
            dim = self.hyperparameters.input_embedding_dims[field]
        else:
            dim = self.hyperparameters.input_embeddings_default_dim
        assert dim is not None, 'Unable to resolve embeddings dimensions for field: ' + field
        return dim

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters
        mlp_input_dim = self.hyperparameters.lstm_h_dim
        mlp_input_dim += sum([self.get_embd_dim(field) for field in self.hyperparameters.mlp_input_fields])
        mlp_input_dim += (1 + self.hyperparameters.lstm_h_dim) * len(self.hyperparameters.token_neighbour_types)

        word_vec_dim = self.get_word_embd_dim() \
                      + 2 \
                      + self.pos_vocab.size() \
                      + 1 \
                      + self.wordnet_hypernyms_vocab.size()

        p1_vec_input_dim = 2 * word_vec_dim
        p2_vec_input_dim = hp.layer_dim + word_vec_dim

        self.params = ModelOptimizedParams(
            word_embeddings=pc.add_lookup_parameters(self.words_vocab.size(), self.get_word_embd_dim()),
            p1_layer=SingleLayerParams(
                pc.add_parameters((hp.layer_dim, p1_vec_input_dim)),
                pc.add_parameters((hp.layer_dim,))
            ),
            p2_layers=[
                SingleLayerParams(
                    pc.add_parameters((hp.layer_dim, p2_vec_input_dim)),
                    pc.add_parameters((hp.layer_dim,))
                ) for _ in range(hp.max_head_distance)
            ],
            w=pc.add_parameters((hp.layer_dim, hp.layer_dim))
        )

        for word in self.words_vocab.all_words():
            word_index = self.words_vocab.get_index(word)
            vector = self.word_embeddings.get(word)
            if vector is not None:
                self.params.word_embeddings.init_row(word_index, vector)

        return pc

    def _build_network_for_input(self, sample_x, apply_dropout):
        dropout_p = self.hyperparameters.dropout_p

        embeddings = [
            dy.concatenate([
                self.get_embd(token_data, field)
                for field in self.hyperparameters.lstm_input_fields
            ])
            for token_data in xs
        ]

        lstm_outputs = cur_lstm_state.transduce(embeddings)
        mlp_activation = get_activation_function(self.hyperparameters.mlp_activation)
        outputs = []
        for ind, lstm_out in enumerate(lstm_outputs):
            if mask and not mask[ind]:
                output = None
            else:
                cur_out = lstm_out
                inp_token = xs[ind]
                for neighbour_type in self.hyperparameters.token_neighbour_types:
                    neighbour_ind = xs[ind].neighbors.get(neighbour_type)
                    if neighbour_ind is None:
                        cur_out = dy.concatenate([cur_out, dy.inputTensor([-1]), dy.inputTensor([0] * self.hyperparameters.lstm_h_dim)])
                    else:
                        cur_out = dy.concatenate([cur_out, dy.inputTensor([1]), lstm_outputs[neighbour_ind]])
                cur_out = dy.concatenate([cur_out] +
                                         [dy.lookup(
                                             self.params.input_lookups[field],
                                             self.input_vocabularies[field].get_index(inp_token[field]),
                                             update=self.hyperparameters.input_embeddings_to_update.get(field) or False
                                         ) for field in self.hyperparameters.mlp_input_fields])
                output = []
                for mlp_params, softmax_params in zip(self.params.mlps, self.params.softmaxes):
                    mlp_cur_out = cur_out
                    for mlp_layer_params in mlp_params:
                        mlp_cur_out = dy.dropout(mlp_cur_out, mlp_dropout_p)
                        mlp_cur_out = mlp_activation(dy.parameter(mlp_layer_params.W) * mlp_cur_out + dy.parameter(mlp_layer_params.b))
                    mlp_cur_out = dy.log_softmax(dy.parameter(softmax_params.W) * mlp_cur_out + dy.parameter(softmax_params.b))
                    output.append(mlp_cur_out)
            outputs.append(output)
        return outputs

    def _build_loss(self, outputs, ys):
        losses = []
        for out, y in zip(outputs, ys):
            if out is not None:
                if y is None:
                    y = [None] * self.hyperparameters.n_labels_to_predict
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
    def fit(self, samples, validation_samples=None, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        self.pc = self._build_network_params()
        # self._build_vocabularies(samples + validation_samples or [])

        if validation_samples:
            test = validation_samples
            train = samples
        else:
            test = samples[:int(len(samples) * self.hyperparameters.validation_split)]
            train = samples[int(len(samples) * self.hyperparameters.validation_split):]

        self.test_set_evaluation = []
        self.train_set_evaluation = []

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
                epoch_test_eval = evaluator.evaluate(test, examples_to_show=5, predictor=self)
                self.test_set_evaluation.append(epoch_test_eval)
                print('Training data evaluation:')
                epoch_train_eval = evaluator.evaluate(train, examples_to_show=5, predictor=self)
                self.train_set_evaluation.append(epoch_train_eval)
                print('--------------------------------------------')

        print('--------------------------------------------')
        print('Training is complete (%d samples, %d epochs)' % (len(train), self.hyperparameters.epochs))
        print('--------------------------------------------')

        return self

    def predict(self, sample_xs, mask=None):
        dy.renew_cg()
        if mask is None:
            mask = [True] * len(sample_xs)
        outputs = self._build_network_for_input(sample_xs, mask, apply_dropout=False)
        ys = []
        for token_ind, out in enumerate(outputs):
            if not mask[token_ind] or out is None:
                predictions = [None] * self.hyperparameters.n_labels_to_predict
            else:
                predictions = []
                for klass_out in out:
                    ind = np.argmax(klass_out.npvalue())
                    predicted = self.output_vocabulary.get_word(ind) if mask[token_ind] else None
                    predictions.append(predicted)
            predictions = tuple(predictions)
            ys.append(predictions)
        assert all([y is None or type(y) is tuple and len(y) == self.hyperparameters.n_labels_to_predict for y in ys])
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
        with open(base_path + '.embds', 'w') as f:
            json.dump({name: pythonize_embds(embds) for name, embds in self.input_embeddings.items()}, f)

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
                with open(base_path + '.in_vocabs', 'r') as in_vocabs_f:
                    with open(base_path + '.out_vocab', 'r') as out_vocabs_f:
                        with open(base_path + '.embds', 'r') as embds_f:
                            model = LstmMlpMulticlassModel(
                                input_vocabularies={name: Vocabulary.unpack(packed) for name, packed in json.load(in_vocabs_f).items()},
                                output_vocabulary=Vocabulary.unpack(json.load(out_vocabs_f)),
                                input_embeddings=json.load(embds_f),
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

    def get_word_embd_dim(self):
        for word in self.word_embeddings:
            return len(self.word_embeddings[word])


