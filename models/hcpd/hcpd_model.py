import zipfile
import os
from collections import namedtuple

import random
from glob import glob
from tempfile import NamedTemporaryFile

import dynet as dy
import json
import numpy as np
import math

from evaluators.ppatt_evaluator import PPAttEvaluator
from models.hcpd import vocabs
from models.hcpd import word_embeddings
from models.hcpd.pss_debug_evaluator import PSSDebugEvaluator
from supersense_repo import get_pss_hierarchy
from vocabulary import Vocabulary

MLPLayerParams = namedtuple('MLPLayerParams', [
    'W',
    'b'
])

MLPParams = namedtuple('MLPParams', [
    'layers'
])

DebugParams = namedtuple('DebugParams', [
    'W_debug_vec',
    'W_pss_role',
    'W_pss_func',
])

class ModelOptimizedParams:

    def __init__(self, word_embeddings, p1_mlp, p2_mlps, w, verb_ss_embeddings, noun_ss_embeddings, pss_lookup, debug):
        self.p1_mlp = p1_mlp
        self.p2_mlps = p2_mlps
        self.w = w
        self.word_embeddings = word_embeddings
        self.debug = debug
        self.verb_ss_embeddings = verb_ss_embeddings
        self.noun_ss_embeddings = noun_ss_embeddings
        self.pss_lookup = pss_lookup


MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])


class HCPDModel(object):

    class HeadCand:
        def __init__(self, ind, word, lemma, pp_distance, is_verb, is_noun, next_pos, hypernyms, is_pp_in_verbnet_frame, verb_ss, noun_ss):
            self.ind = ind
            self.next_pos = next_pos
            self.hypernyms = hypernyms
            self.is_pp_in_verbnet_frame = is_pp_in_verbnet_frame
            self.pp_distance = pp_distance
            self.word = word
            self.lemma = lemma
            self.is_verb = is_verb
            self.is_noun = is_noun
            self.verb_ss = verb_ss
            self.noun_ss = noun_ss
            assert self.is_verb != self.is_noun
    PP = namedtuple('PP', ['word', 'ind', 'pss_role', 'pss_func'])
    Child = namedtuple('PP', ['word', 'lemma', 'hypernyms', 'noun_ss'])

    class SampleX:
        def __init__(self, head_cands, pp, child, sent_id, tokens, pss_role=None, pss_func=None):
            self.head_cands = head_cands
            self.pp = pp
            self.pss_role = pss_role
            self.pss_func = pss_func
            self.child = child
            self.tokens = tokens
            self.sent_id = sent_id

    class SampleY:
        def __init__(self, scored_heads):
            self.scored_heads = scored_heads
            self.correct_head_cand = max([sh for sh in self.scored_heads], key=lambda x: x[0])[1]

        def pprint(self, only_best=False):
            for score, head in sorted(self.scored_heads, key=lambda x: -x[0]):
                print("[%s: %1.2f]" % (head.word, score), end='\t')
            print("")

    class Sample:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            assert self.y.correct_head_cand in self.x.head_cands
        def get_correct_head_ind(self):
            return self.x.head_cands.index(self.y.correct_head_cand)
        def get_closest_head_to_pp(self):
            return min(self.x.head_cands, key=lambda hc: hc.pp_distance)
        def pprint(self):
            print("[PP: %s]" % self.x.pp.word)
            print("Correct Head: " + self.y.correct_head_cand.word)

    class HyperParameters:
        def __init__(self,
                     max_head_distance=5,
                     p1_vec_dim=150,
                     p1_mlp_layers=1,
                     p2_vec_dim=150,
                     p2_mlp_layers=1,
                     verb_ss_embedding_dim=10,
                     noun_ss_embedding_dim=10,
                     activation='tanh',
                     dropout_p=0.01,
                     learning_rate=1,
                     learning_rate_decay=0,
                     update_embeddings=True,
                     trainer="SimpleSGDTrainer",
                     use_pss=False,
                     pss_embd_dim=5,
                     pss_embd_type='lookup',
                     use_verb_noun_ss=False,
                     fallback_to_lemmas=False,
                     epochs=100
                     ):
            self.pss_embd_type = pss_embd_type
            self.pss_embd_dim = pss_embd_dim
            self.p1_mlp_layers = p1_mlp_layers
            self.p2_mlp_layers = p2_mlp_layers
            self.p1_vec_dim = p1_vec_dim
            self.p2_vec_dim = p2_vec_dim
            self.activation = activation
            self.trainer = trainer
            self.max_head_distance = max_head_distance
            self.dropout_p = dropout_p
            self.update_embeddings = update_embeddings
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.learning_rate_decay = learning_rate_decay
            self.use_pss = use_pss
            self.use_verb_noun_ss = use_verb_noun_ss
            self.fallback_to_lemmas = fallback_to_lemmas
            self.verb_ss_embedding_dim = verb_ss_embedding_dim
            self.noun_ss_embedding_dim = noun_ss_embedding_dim

            assert self.pss_embd_type in ['lookup', 'binary']

    def __init__(self,
                 words_vocab=vocabs.WORDS,
                 words_to_lemmas=vocabs.WORDS_TO_LEMMAS,
                 pos_vocab=vocabs.GOLD_POS,
                 wordnet_hypernyms_vocab=vocabs.HYPERNYMS,
                 pss_vocab=vocabs.PSS,
                 verb_ss_vocab=vocabs.VERB_SS,
                 noun_ss_vocab=vocabs.NOUN_SS,
                 word_embeddings=word_embeddings.SYNTAX_WORD_VECTORS,
                 hyperparameters=HyperParameters(),
                 debug_feature=None):

        self.debug_feature = debug_feature
        self.wordnet_hypernyms_vocab = wordnet_hypernyms_vocab
        self.words_vocab = words_vocab
        self.words_to_lemmas = words_to_lemmas
        self.pos_vocab = pos_vocab
        self.pss_vocab = pss_vocab
        self.verb_ss_vocab = verb_ss_vocab
        self.noun_ss_vocab = noun_ss_vocab
        self.word_embeddings = word_embeddings

        self.hyperparameters = hyperparameters
        self.dev_set_evaluation = None
        self.train_set_evaluation = None
        self.pc = self._build_network_params()

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        hp = self.hyperparameters

        self.p1_vec_inp_dim = self.get_word_embd_dim() * 2 \
                                   + self.wordnet_hypernyms_vocab.size()
        if hp.use_pss:
            if hp.pss_embd_type == 'lookup':
                pss_dim = hp.pss_embd_dim
            else:
                pss_dim = self.pss_vocab.size()
            self.p1_vec_inp_dim += pss_dim * 2
        if hp.use_verb_noun_ss:
            self.p1_vec_inp_dim += hp.noun_ss_embedding_dim

        self.p2_vec_inp_dim = hp.p1_vec_dim \
                                   + self.get_word_embd_dim() \
                                   + 2 \
                                   + self.pos_vocab.size() \
                                   + 1 \
                                   + self.wordnet_hypernyms_vocab.size()
        if hp.use_verb_noun_ss:
            self.p2_vec_inp_dim += hp.verb_ss_embedding_dim + hp.noun_ss_embedding_dim

        self.params = ModelOptimizedParams(
            word_embeddings=pc.add_lookup_parameters((self.words_vocab.size(), self.get_word_embd_dim())),
            p1_mlp=MLPParams(layers=[
                MLPLayerParams(
                    W=pc.add_parameters((hp.p1_vec_dim, self.p1_vec_inp_dim if layer_ind == 0 else hp.p1_vec_dim)),
                    b=pc.add_parameters((hp.p1_vec_dim,))
                ) for layer_ind in range(hp.p1_mlp_layers)
            ]),
            p2_mlps=[
                MLPParams(
                    layers=[
                        MLPLayerParams(
                            W=pc.add_parameters((hp.p2_vec_dim, self.p2_vec_inp_dim if layer_ind == 0 else hp.p2_vec_dim)),
                            b=pc.add_parameters((hp.p2_vec_dim,))
                        )
                    for layer_ind in range(hp.p2_mlp_layers)]
                ) for _ in range(hp.max_head_distance)
            ],
            w=pc.add_parameters((1, hp.p2_vec_dim)),
            verb_ss_embeddings=pc.add_lookup_parameters((self.verb_ss_vocab.size(), hp.verb_ss_embedding_dim)),
            noun_ss_embeddings=pc.add_lookup_parameters((self.noun_ss_vocab.size(), hp.noun_ss_embedding_dim)),

            pss_lookup=pc.add_lookup_parameters((self.pss_vocab.size(), hp.pss_embd_dim)),

            debug=DebugParams(
                W_debug_vec=pc.add_parameters((hp.p2_vec_dim, hp.p2_vec_dim)),
                W_pss_role=pc.add_parameters((self.pss_vocab.size(), hp.p2_vec_dim)),
                W_pss_func=pc.add_parameters((self.pss_vocab.size(), hp.p2_vec_dim)),
            )
        )

        n_words = self.words_vocab.size()
        n_words_with_embeddings = 0
        for word in self.words_vocab.all_words():
            word_index = self.words_vocab.get_index(word)
            vector = self.word_embeddings.get(word)
            if vector is None and self.hyperparameters.fallback_to_lemmas:
                lemma = self.words_to_lemmas.get(word)
                if lemma is not None:
                    vector = self.word_embeddings.get(lemma)
            if vector is not None:
                n_words_with_embeddings += 1
                self.params.word_embeddings.init_row(word_index, vector)

        print("Embedding for %d/%d words (%d%%)" % (n_words_with_embeddings, n_words, int(n_words_with_embeddings/n_words*100)))

        # if hp.update_embd_for_missing:
        #     self.params.verb_ss_embeddings.init_row(self.verb_ss_vocab.get_index(None), [0] * hp.verb_ss_embedding_dim, update=True)
        #     self.params.noun_ss_embeddings.init_row(self.noun_ss_vocab.get_index(None), [0] * hp.noun_ss_embedding_dim, update=True)
        # else:
        #     self.params.verb_ss_embeddings.init_row(self.verb_ss_vocab.get_index(None), [0] * hp.verb_ss_embedding_dim, update=False)
        #     self.params.noun_ss_embeddings.init_row(self.noun_ss_vocab.get_index(None), [0] * hp.noun_ss_embedding_dim, update=False)
        #
        return pc

    def _get_binary_vec(self, vocab, words):
        vec = [0] * vocab.size()
        for word in words or []:
            vec[vocab.get_index(word)] = 1
        return vec

    def _build_head_vec(self, head_cand):
        pv = dy.concatenate([
            dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(head_cand.word), update=self.hyperparameters.update_embeddings),
            dy.inputTensor([1, 0] if head_cand.is_noun else [0, 1]),
            dy.inputTensor(self._get_binary_vec(self.pos_vocab, [head_cand.next_pos])),
            dy.inputTensor([1] if head_cand.is_pp_in_verbnet_frame else [0]),
            dy.inputTensor(self._get_binary_vec(self.wordnet_hypernyms_vocab, head_cand.hypernyms))
        ])
        if self.hyperparameters.use_verb_noun_ss:
            pv = dy.concatenate([
                pv,
                dy.lookup(self.params.noun_ss_embeddings, self.noun_ss_vocab.get_index(head_cand.noun_ss)),
                dy.lookup(self.params.verb_ss_embeddings, self.verb_ss_vocab.get_index(head_cand.verb_ss)),
            ])
        return pv

    def get_pss_vec(self, pss):
        if self.hyperparameters.pss_embd_type == 'lookup':
            return dy.lookup(self.params.pss_lookup, self.pss_vocab.get_index(pss))
        elif self.hyperparameters.pss_embd_type == 'binary':
            return dy.inputTensor(self._get_binary_vec(self.pss_vocab, get_pss_hierarchy(pss)))
        else:
            raise Exception("Unknown PSS embedding type: " + self.hyperparameters.pss_embd_type)

    def _build_prep_vec(self, pp):
        pv = dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(pp.word), update=self.hyperparameters.update_embeddings)
        if self.hyperparameters.use_pss:
            pv = dy.concatenate([
                pv,
                self.get_pss_vec(pp.pss_role),
                self.get_pss_vec(pp.pss_func)
            ])
        return pv

    def _build_child_vec(self, child):
        pv = dy.concatenate([
            dy.lookup(self.params.word_embeddings, self.words_vocab.get_index(child.word), update=self.hyperparameters.update_embeddings),
            dy.inputTensor(self._get_binary_vec(self.wordnet_hypernyms_vocab, child.hypernyms)),
        ])
        if self.hyperparameters.use_verb_noun_ss:
            pv = dy.concatenate([
                pv,
                dy.lookup(self.params.noun_ss_embeddings, self.noun_ss_vocab.get_index(child.noun_ss)),
            ])
        return pv

    def _build_network_for_input(self, sample_x):
        dropout_p = self.hyperparameters.dropout_p

        _debug_vec = None

        scores = []
        for head_cand in sample_x.head_cands:
            p1_inp_vec = dy.dropout(
                dy.concatenate([
                    self._build_child_vec(sample_x.child),
                    self._build_prep_vec(sample_x.pp)
                ]),
                self.hyperparameters.dropout_p
            )
            p1_mlp_vec = p1_inp_vec
            for ind, layer in enumerate(self.params.p1_mlp.layers):
                p1_mlp_vec = getattr(dy, self.hyperparameters.activation)(
                    dy.dropout(
                        dy.parameter(layer.W) * p1_mlp_vec + dy.parameter(layer.b),
                        dropout_p
                    )
                )
            p1_vec = p1_mlp_vec

            p2_inp_vec = dy.concatenate([self._build_head_vec(head_cand), p1_vec])
            head_dist = min(head_cand.pp_distance, self.hyperparameters.max_head_distance)
            p2_mlp_layers = self.params.p2_mlps[head_dist - 1].layers

            p2_mlp_vec = p2_inp_vec
            for ind, layer in enumerate(p2_mlp_layers):
                p2_mlp_vec = getattr(dy, self.hyperparameters.activation)(
                    dy.dropout(
                        dy.parameter(layer.W) * p2_mlp_vec + dy.parameter(layer.b),
                        dropout_p
                    )
                )
            p2_vec = p2_mlp_vec

            _debug_vec = _debug_vec or p2_vec
            _debug_vec = dy.tanh(dy.parameter(self.params.debug.W_debug_vec) * _debug_vec)

            score = dy.parameter(self.params.w) * p2_vec
            scores.append(score)

        _debug_pss_role_probs = dy.log_softmax(dy.parameter(self.params.debug.W_pss_role) * _debug_vec)
        _debug_pss_func_probs = dy.log_softmax(dy.parameter(self.params.debug.W_pss_func) * _debug_vec)

        return {
            'scores': scores,
            'debug': [_debug_pss_role_probs, _debug_pss_func_probs]
        }

    def _build_loss(self, sample, network_out):
        out_scores = network_out['scores']
        correct_head_score = out_scores[sample.get_correct_head_ind()]
        return dy.emax(
            [score - correct_head_score + (0 if ind == sample.get_correct_head_ind() else 1) for ind, score in enumerate(out_scores)]
        )

    def _build_debug_loss_pss(self, sample, network_out):
        _debug_pss_role_probs, _debug_pss_func_probs = network_out['debug']
        loss_role = -dy.pick(_debug_pss_role_probs, self.pss_vocab.get_index(sample.x.pp.pss_role))
        loss_func = -dy.pick(_debug_pss_func_probs, self.pss_vocab.get_index(sample.x.pp.pss_func))
        return loss_role + loss_func

    def fit(self, samples, validation_samples, additional_validation_sets=None, show_progress=True, show_sample_predictions=False, resume=False):
        print("Training size: %d, validation size: %d" % (len(samples), len(validation_samples)))
        additional_validation_sets = additional_validation_sets or {}
        if not resume:
            self.pc = self._build_network_params()
        # self._build_vocabularies(samples + validation_samples or [])

        if self.debug_feature == 'pss':
            build_loss = self._build_debug_loss_pss
            evaluator = PSSDebugEvaluator()
        elif not self.debug_feature:
            build_loss = self._build_loss
            evaluator = PPAttEvaluator()
        else:
            raise Exception('Unsupported feature for debugging:', self.debug_feature)


        test = validation_samples
        train = samples

        best_test_acc = None
        train_acc = None
        best_epoch = None
        model_file_path = '/tmp/_m_' + str(random.randrange(10000))
        try:
            # trainer = dy.SimpleSGDTrainer(self.pc, learning_rate=self.hyperparameters.learning_rate)
            trainer = getattr(dy, self.hyperparameters.trainer)(self.pc, self.hyperparameters.learning_rate)
            for epoch in range(1, self.hyperparameters.epochs + 1):
                if np.isinf(trainer.learning_rate):
                    break

                train = list(train)
                random.shuffle(train)
                loss_sum = 0

                BATCH_SIZE = 500 if len(train) > 10000 else 20
                batches = [train[batch_ind::int(math.ceil(len(train)/BATCH_SIZE))] for batch_ind in range(int(math.ceil(len(train)/BATCH_SIZE)))]
                for batch_ind, batch in enumerate(batches):
                    _build_loss = self._build_loss
                    # _build_loss = random.choice([self._build_loss, build_loss])
                    dy.renew_cg(immediate_compute=True, check_validity=True)
                    losses = []
                    for sample in batch:
                        out = self._build_network_for_input(sample.x)
                        sample_loss = _build_loss(sample, out)
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

                print('--------------------------------------------')
                print('Epoch %d complete, avg loss: %1.4f' % (epoch, loss_sum/len(train)))
                print('Validation data evaluation:')
                epoch_test_eval = evaluator.evaluate(test, examples_to_show=5 if show_sample_predictions else 0, predictor=self)
                for name, vset in additional_validation_sets.items():
                    print('Validation data evaluation (%s):' % name)
                    evaluator.evaluate(vset, examples_to_show=5, predictor=self)
                print('Training data evaluation:')
                epoch_train_eval = evaluator.evaluate(train, examples_to_show=5 if show_sample_predictions else 0, predictor=self)
                print('--------------------------------------------')

                test_acc = epoch_test_eval['acc']
                if best_test_acc is None or test_acc > best_test_acc:
                    print("Best epoch so far! with accuracy of: %1.2f" % test_acc)
                    best_test_acc = test_acc
                    train_acc = epoch_train_eval['acc']
                    best_epoch = epoch
                    self.pc.save(model_file_path)

            self.train_set_evaluation = {'acc': train_acc, 'best_epoch': best_epoch}
            self.dev_set_evaluation = {'acc': best_test_acc, 'best_epoch': best_epoch}

            print('--------------------------------------------')
            print('Training is complete (%d samples, %d epochs, best epoch - %d acc %1.2f)' % (len(train), self.hyperparameters.epochs, best_epoch, best_test_acc))
            print('--------------------------------------------')
            self.pc.populate(model_file_path)
        finally:
            os.remove(model_file_path)

        return self

    def predict(self, sample_x):
        dy.renew_cg()
        out_scores = [s.value() for s in self._build_network_for_input(sample_x)['scores']]
        return HCPDModel.SampleY(
            list(zip(out_scores, sample_x.head_cands))
        )

    def debug_predict_pss(self, sample_x):
        dy.renew_cg()
        _debug_pss_role_probs, _debug_pss_func_probs = self._build_network_for_input(sample_x)['debug']
        role_ind = np.argmax(_debug_pss_role_probs.npvalue())
        role = self.pss_vocab.get_word(role_ind)
        func_ind = np.argmax(_debug_pss_func_probs.npvalue())
        func = self.pss_vocab.get_word(func_ind)
        return [role, func]

    def save(self, base_path):
        def pythonize_embds(embds):
            return {k: [float(x) for x in list(v)] for k, v in embds.items()}

        self.pc.save(base_path)
        with open(base_path + '.hp', 'w') as f:
            json.dump(vars(self.hyperparameters), f, indent=2)
        vocabs = {
            name: vocab.pack() for name, vocab in {'pos': self.pos_vocab, 'words': self.words_vocab, 'hypernyms': self.wordnet_hypernyms_vocab}.items()
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


