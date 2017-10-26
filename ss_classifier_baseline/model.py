from collections import namedtuple

import supersenses
import dynet as dy
import numpy as np

# There are more hidden parameters coming from the LSTMs
from utils import clear_nones, f1_score

XTokenData = namedtuple('XTokenData', ['token', 'pos'])
YTokenData = namedtuple('Y', ['supersense'])
Sample = namedtuple('Sample', ['xs', 'ys'])

ModelOptimizedParams = namedtuple('ModelOptimizedParams',[
    'token_lookup',
    'pos_lookup',
    'mlp',
    'softmax'
])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class SupersensesClassifierBaselineModel(object):

    def __init__(self,
                 token_vocab,
                 pos_vocab,
                 ss_vocab,
                 ss_types_to_predict,
                 token_embedding_vec_size=100,
                 pos_embedding_vec_size=100,
                 mlp_layers=2,
                 mlp_layer_size=50,
                 lstm_h_vec_size=50,
                 num_lstm_layers=2,
                 is_bilstm=True,
                 use_pos_tags=True):
        if ss_types_to_predict is None:
            ss_types_to_predict = list(supersenses.constants.TYPES)
        self.ss_types_to_predict = ss_types_to_predict

        self.token_vocab = token_vocab
        self.ss_vocab = ss_vocab
        self.pos_vocab = pos_vocab

        self.use_pos_tags = use_pos_tags
        self.token_embedding_vec_size = token_embedding_vec_size
        self.pos_embedding_vec_size = pos_embedding_vec_size
        self.mlp_layers = mlp_layers
        self.mlp_layer_size = mlp_layer_size
        self.lstm_h_vec_size = lstm_h_vec_size
        self.num_lstm_layers = num_lstm_layers
        self.is_bilstm = is_bilstm

        self.lstm_cell_output_size = self.lstm_h_vec_size * (2 if self.is_bilstm else 1)
        self.embedded_input_vec_size = sum([
            self.token_embedding_vec_size,
            self.pos_embedding_vec_size if self.use_pos_tags else 0
        ])

    def _build_network_params(self):
        pc = dy.ParameterCollection()

        self.params = ModelOptimizedParams(
            token_lookup=pc.add_lookup_parameters((self.token_vocab.get_vocabulary_size(), self.token_embedding_vec_size)),
            pos_lookup=pc.add_lookup_parameters((self.pos_vocab.get_vocabulary_size(), self.pos_embedding_vec_size)),
            mlp=[
                MLPLayerParams(
                    W=pc.add_parameters((self.mlp_layer_size, self.mlp_layer_size if i > 0 else self.lstm_cell_output_size)),
                    b=pc.add_parameters((self.mlp_layer_size,))
                )
                for i in range(self.mlp_layers)
            ],
            softmax=MLPLayerParams(
                W=pc.add_parameters((self.ss_vocab.get_vocabulary_size(), self.mlp_layer_size)),
                b=pc.add_parameters((self.ss_vocab.get_vocabulary_size(),))
            )
        )
        if self.is_bilstm:
            self.lstm_builder = dy.BiRNNBuilder(self.num_lstm_layers, self.embedded_input_vec_size, self.lstm_h_vec_size, pc, dy.LSTMBuilder)
        else:
            self.lstm_builder = dy.LSTMBuilder(self.num_lstm_layers, self.embedded_input_vec_size, self.lstm_h_vec_size, pc)

        return pc

    def _build_network_for_input(self, inp):
        dy.renew_cg()
        cur_lstm_state = self.lstm_builder.initial_state()
        embeddings = [
            dy.concatenate(clear_nones([
                dy.lookup(self.params.token_lookup, self.token_vocab.get_index(token_data.token)),
                dy.lookup(self.params.pos_lookup, self.pos_vocab.get_index(token_data.pos)) if self.use_pos_tags else None,
            ]))
            for token_data in inp
        ]
        lstm_outputs = cur_lstm_state.transduce(embeddings)
        outputs = []
        for lstm_out in lstm_outputs:
            cur_out = lstm_out
            for mlp_layer_params in self.params.mlp:
                cur_out = dy.tanh(dy.parameter(mlp_layer_params.W) * cur_out + dy.parameter(mlp_layer_params.b))
            cur_out = dy.softmax(dy.parameter(self.params.softmax.W) * cur_out + dy.parameter(self.params.softmax.b))
            outputs.append(cur_out)
        return outputs

    def _build_loss(self, outputs, ys):
        losses = []
        for out, y in zip(outputs, ys):
            ss_ind = self.ss_vocab.get_index(y.supersense)
            loss = -dy.log(dy.pick(out, ss_ind))
            losses.append(loss)
        return dy.esum(losses)

    def fit(self, samples, epochs=5, validation_split=0.2):
        pc = self._build_network_params()
        trainer = dy.SimpleSGDTrainer(pc)
        train = samples[:int(len(samples)*validation_split)]
        test = samples[int(len(samples)*validation_split):]

        for epoch in range(1, epochs + 1):
            loss_sum = 0
            for ind, sample in enumerate(train):
                outputs = self._build_network_for_input(sample.xs)
                loss = self._build_loss(outputs, sample.ys)
                loss.forward()
                loss_sum += loss.value()
                loss.backward()
                trainer.update()

                if int((ind + 1) / len(train) * 100) > int(ind / len(train) * 100):
                    per = int((ind + 1) / len(train) * 100)
                    print('\r\rEpoch %3d (%d%%): |' % (epoch, per) + '#' * per + '-' * (100 - per) + '|',)

            print('Epoch %d complete, avg loss: %1.4f' % (epoch, loss_sum/len(train)))
            self.evaluate(test)
        print('--------------------------------------------')
        print('Training is complete (%d samples, %d epochs)' % (len(samples), epochs))
        print('Test data evaluation:')
        self.evaluate(test)
        print('Training data evaluation:')
        self.evaluate(train)
        print('--------------------------------------------')

        return self

    def predict(self, sample_x):
        outputs = self._build_network_for_input(sample_x)
        ys = []
        for out in outputs:
            ind = np.argmax(out.npvalue())
            ss = self.ss_vocab.get_word(ind)
            ys.append(YTokenData(supersense=ss))
        return ys

    def evaluate(self, samples):
        p_none_a_none = 0
        p_none_a_value = 0
        p_value_a_none = 0
        p_value_a_value_eq = 0
        p_value_a_value_neq = 0
        total = 0
        for sample in samples:
            predicted_ys = self.predict(sample.xs)
            actual_ys = sample.ys
            for p, a in zip(predicted_ys, actual_ys):
                if p.supersense is None and a.supersense is None:
                    p_none_a_none += 1
                elif p.supersense is None and a.supersense is not None:
                    p_none_a_value += 1
                elif p.supersense is not None and a.supersense is None:
                    p_value_a_none += 1
                elif p.supersense == a.supersense:
                    p_value_a_value_eq += 1
                else:
                    p_value_a_value_neq += 1
                total += 1

        precision = (p_value_a_value_eq / (p_value_a_value_eq + p_value_a_value_neq + p_value_a_none))
        recall = (p_value_a_value_eq / (p_value_a_value_eq + p_value_a_value_neq + p_none_a_value))
        f1 = f1_score(precision, recall)
        print('Evaluation on %d samples:' % len(samples))
        print(' - precision: %1.4f' % precision)
        print(' - recall:    %1.4f' % recall)
        print(' - f1 score:  %1.4f' % f1)