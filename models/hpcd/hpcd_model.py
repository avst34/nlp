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
                     dropout_p=0.5,
                     update_embeddings=True,
                     layer_dim=??
                     ):
            self.max_head_distance = max_head_distance
            self.dropout_p = dropout_p
            self.layer_dim = layer_dim
            self.update_embeddings = update_embeddings

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

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters

        self.word_vec_dim = self.get_word_embd_dim() \
                      + 2 \
                      + self.pos_vocab.size() \
                      + 1 \
                      + self.wordnet_hypernyms_vocab.size()

        p1_vec_input_dim = 2 * self.word_vec_dim
        p2_vec_input_dim = hp.layer_dim + self.word_vec_dim

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
            w=pc.add_parameters((1, hp.layer_dim))
        )

        for word in self.words_vocab.all_words():
            word_index = self.words_vocab.get_index(word)
            vector = self.word_embeddings.get(word)
            if vector is not None:
                self.params.word_embeddings.init_row(word_index, vector)

        return pc

    def _get_binary_vec(self, vocab, words):
        vec = [0] * vocab.size()
        for word in words:
            vec[vocab.get_index(word)] = 1
        return vec


    def _build_head_vec(self, head_cand, pp):
        return dy.concatenate([
            dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(head_cand.word), update=self.hyperparameters.update_embeddings),
            dy.constant([1, 0] if head_cand.is_noun else [0, 1]),
            dy.constant(self._get_binary_vec(self.pos_vocab, head_cand.next_pos)),
            dy.constant([1] if head_cand.is_pp_in_verbnet_frame else [0]),
            dy.constant(self._get_binary_vec(self.wordnet_hypernyms_vocab, head_cand.hypernyms))
        ])

    def _build_prep_vec(self, pp):
        return dy.zero_pad(
            dy.concatenate([
                dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(pp.word), update=self.hyperparameters.update_embeddings),
            ]),
            self.word_vec_dim
        )

    def _build_child_vec(self, child):
        return dy.zero_pad(
            dy.concatenate([
                dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(pp.word), update=self.hyperparameters.update_embeddings),
                dy.constant(self._get_binary_vec(self.wordnet_hypernyms_vocab, child.hypernyms))
            ]),
            self.word_vec_dim
        )

    def _build_network_for_input(self, sample_x, apply_dropout):
        dropout_p = self.hyperparameters.dropout_p

        scores = []
        for head_cand in sample_x.head_cands:
            p1_inp_vec = dy.concatenate([
                self._build_child_vec(sample_x.child),
                self._build_prep_vec(sample_x.pp)
            ])
            p1_vec = p1_activation(dy.parameter(self.params.p1_layer.W) * p1_inp_vec + dy.parameter(self.params.p1_layer.b))
            p2_inp_vec = dy.concatenate([self._build_head_vec(head_cand, sample_x.pp), p1_vec])
            head_dist = min(head_cand.pp_distance, self.hyperparameters.max_head_distance)
            layer_params = self.params.p2_layers[head_dist - 1]
            p2_vec = p2_activation(dy.parameter(layer_params.W) * p2_inp_vec + dy.parameter(layer_params.b))
            score = p2_vec * self.params.w
            scores.append(score)

        return scores

    def _build_loss(self, sample, out_scores):
        raise Exception('not implemented')
        # losses = []
        # for out, y in zip(outputs, ys):
        #     if out is not None:
        #         if y is None:
        #             y = [None] * self.hyperparameters.n_labels_to_predict
        #         assert len([label_y is None for label_y in y]) in [0, len(y)], "Got a sample with partial None labels"
        #         for label_out, label_y in zip(out, y):
        #             if self.output_vocabulary.has_word(label_y):
        #                 ss_ind = self.output_vocabulary.get_index(label_y)
        #                 loss = -dy.pick(label_out, ss_ind)
        #                 losses.append(loss)
        #             else:
        #                 assert label_y is None
        # if len(losses):
        #     loss = dy.esum(losses)
        # else:
        #     loss = None
        # return loss

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
                    out_scores = self._build_network_for_input(sample.xs, sample.mask, apply_dropout=True)
                    sample_loss = self._build_loss(sample, out_scores)
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


