import zipfile
import os
from collections import namedtuple

import random
from glob import glob

import dynet as dy
import json
import numpy as np
import math

from vocabulary import Vocabulary
import vocabs
import embeddings

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

    class HeadCand:
        def __init__(self, ind, word, pp_distance, is_verb, is_noun, next_pos, hypernyms, is_pp_in_verbnet_frame):
            self.ind = ind
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
        def __init__(self, head_cands, pp, child):
            self.head_cands = head_cands
            self.pp = pp
            self.child = child

    class SampleY:
        def __init__(self, scored_heads):
            self.scored_heads = scored_heads
            self.correct_head_cand = max([sh for sh in self.scored_heads], key=lambda x: x[0])[1]

        def pprint(self, only_best=False):
            for score, head in sorted(self.scored_heads):
                print("[%s: %1.2f]" % (head.word, score), end='\t')
            print("")

    class Sample:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            assert self.y.correct_head_cand in self.x.head_cands
        def get_correct_head_ind(self):
            return self.x.head_cands.index(self.y.correct_head_cand)
        def pprint(self):
            print("[PP: %s]" % self.x.pp.word)
            print("Correct Head:", end=' ')
            self.y.pprint()

    class HyperParameters:
        def __init__(self, 
                     max_head_distance=5,
                     dropout_p=0.5,
                     update_embeddings=True,
                     epochs=100
                     ):
            self.max_head_distance = max_head_distance
            self.dropout_p = dropout_p
            self.update_embeddings = update_embeddings
            self.epochs = epochs

    def __init__(self,
                 words_vocab=vocabs.WORDS,
                 pos_vocab=vocabs.GOLD_POS,
                 wordnet_hypernyms_vocab=vocabs.HYPERNYMS,
                 word_embeddings=embeddings.SYNTAX_WORD_VECTORS,
                 hyperparameters=HCPDModel.HyperParameters()):

        self.wordnet_hypernyms_vocab = wordnet_hypernyms_vocab
        self.words_vocab = words_vocab
        self.pos_vocab = pos_vocab
        self.word_embeddings = word_embeddings

        self.hyperparameters = hyperparameters
        self.test_set_evaluation = None
        self.train_set_evaluation = None
        self.pc = self._build_network_params()

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters

        self.word_vec_dim = self.get_word_embd_dim() \
                      + 2 \
                      + self.pos_vocab.size() \
                      + 1 \
                      + self.wordnet_hypernyms_vocab.size()

        layer_dim = self.word_vec_dim

        p1_vec_input_dim = 2 * self.word_vec_dim
        p2_vec_input_dim = layer_dim + self.word_vec_dim

        self.params = ModelOptimizedParams(
            word_embeddings=pc.add_lookup_parameters(self.words_vocab.size(), self.get_word_embd_dim()),
            p1_layer=SingleLayerParams(
                pc.add_parameters((layer_dim, p1_vec_input_dim)),
                pc.add_parameters((layer_dim,))
            ),
            p2_layers=[
                SingleLayerParams(
                    pc.add_parameters((layer_dim, p2_vec_input_dim)),
                    pc.add_parameters((layer_dim,))
                ) for _ in range(hp.max_head_distance)
            ],
            w=pc.add_parameters((1, layer_dim))
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

    def _build_network_for_input(self, sample_x):
        dropout_p = self.hyperparameters.dropout_p

        scores = []
        for head_cand in sample_x.head_cands:
            p1_inp_vec = dy.dropout(
                dy.concatenate([
                    self._build_child_vec(sample_x.child),
                    self._build_prep_vec(sample_x.pp)
                ]),
                self.hyperparameters.dropout_p
            )
            p1_vec = dy.tanh(
                dy.dropout(
                    dy.parameter(self.params.p1_layer.W) * p1_inp_vec + dy.parameter(self.params.p1_layer.b),
                    dropout_p
                )
            )
            p2_inp_vec = dy.concatenate([self._build_head_vec(head_cand, sample_x.pp), p1_vec])
            head_dist = min(head_cand.pp_distance, self.hyperparameters.max_head_distance)
            layer_params = self.params.p2_layers[head_dist - 1]
            p2_vec = dy.tanh(dy.parameter(layer_params.W) * p2_inp_vec + dy.parameter(layer_params.b))
            score = p2_vec * self.params.w
            scores.append(score)

        return scores

    def _build_loss(self, sample, out_scores):
        correct_head_score = out_scores[sample.get_correct_head_ind()]
        return dy.max(
            [score - correct_head_score + (1 if ind == sample.get_correct_head_ind() else 0) for ind, score in enumerate(out_scores)]
        )

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

            BATCH_SIZE = 500
            batches = [train[batch_ind::int(math.ceil(len(train)/BATCH_SIZE))] for batch_ind in range(int(math.ceil(len(train)/BATCH_SIZE)))]
            for batch_ind, batch in enumerate(batches):
                dy.renew_cg(immediate_compute=True, check_validity=True)
                losses = []
                for sample in batch:
                    out_scores = self._build_network_for_input(sample.x)
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

    def predict(self, sample_x):
        dy.renew_cg()
        out_scores = [s.value() for s in self._build_network_for_input(sample_x)]
        return HCPDModel.SampleY(
            list(zip(out_scores, sample_x.head_cands))
        )

    def save(self, base_path):
        def pythonize_embds(embds):
            return {k: [float(x) for x in list(v)] for k, v in embds.items()}

        self.pc.save(base_path)
        with open(base_path + '.hp', 'w') as f:
            json.dump(vars(self.hyperparameters), f, indent=2)
        vocabs = {
            name: vocab.pack() for name, vocab in {'pos': self.pos_vocab, 'words': self.words_vocab, 'hypernyms': self.wordnet_hypernyms_vocab}
        }
        with open(base_path + '.vocabs', 'w') as f:
            json.dump(vocabs, f)
        with open(base_path + '.embds', 'w') as f:
            json.dump(pythonize_embds(self.word_embeddings), f)

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
                with open(base_path + '.vocabs', 'r') as vocabs_f:
                    with open(base_path + '.embds', 'r') as embds_f:
                        vocabs_data = json.load(vocabs_f)
                        model = HCPDModel(
                            hyperparameters=HCPDModel.HyperParameters(**json.load(hp_f)),
                            words_vocab=Vocabulary.unpack(vocabs_data['words']),
                            pos_vocab=Vocabulary.unpack(vocabs_data['pos']),
                            wordnet_hypernyms_vocab=Vocabulary.unpack(vocabs_data['hypernym']),
                            word_embeddings=json.load(embds_f)
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


