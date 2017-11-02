from collections import namedtuple

import random
import dynet as dy
import numpy as np

# There are more hidden parameters coming from the LSTMs
Sample = namedtuple('Sample', ['xs', 'ys'])

ModelOptimizedParams = namedtuple('ModelOptimizedParams',[
    'input_lookups',
    'mlp',
    'softmax'
])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class LstmMlpMulticlassModel(object):

    def __init__(self,
                 input_vocabularies,
                 output_vocabulary,
                 input_embedding_sizes=None,
                 input_embeddings_default_size=10,
                 mlp_layers=2,
                 mlp_layer_size=10,
                 lstm_h_vec_size=10,
                 num_lstm_layers=2,
                 is_bilstm=True):
        self.input_vocabularies = input_vocabularies
        self.output_vocabulary = output_vocabulary
        if not input_embedding_sizes:
            input_embedding_sizes = {field: input_embeddings_default_size for field in self.input_vocabularies}
        self.input_embedding_sizes = input_embedding_sizes

        self.mlp_layers = mlp_layers
        self.mlp_layer_size = mlp_layer_size
        self.lstm_h_vec_size = lstm_h_vec_size
        self.num_lstm_layers = num_lstm_layers
        self.is_bilstm = is_bilstm
        self.mlp_dropout_p = 0

        self.lstm_cell_output_size = self.lstm_h_vec_size * (2 if self.is_bilstm else 1)
        self.embedded_input_vec_size = sum(input_embedding_sizes.values())

        self.validate_config()

    def validate_config(self):
        if set(self.input_vocabularies.keys()) != set(self.input_embedding_sizes.keys()):
            raise Exception("Mismatch between input vocabularies and input_embedding_sizes")

    def _build_network_params(self):
        pc = dy.ParameterCollection()

        self.params = ModelOptimizedParams(
            input_lookups={
                field: pc.add_lookup_parameters((self.input_vocabularies[field].size(), self.input_embedding_sizes[field]))
                for field in self.input_vocabularies
            },
            mlp=[
                MLPLayerParams(
                    W=pc.add_parameters((self.mlp_layer_size, self.mlp_layer_size if i > 0 else self.lstm_cell_output_size)),
                    b=pc.add_parameters((self.mlp_layer_size,))
                )
                for i in range(self.mlp_layers)
            ],
            softmax=MLPLayerParams(
                W=pc.add_parameters((self.output_vocabulary.size(), self.mlp_layer_size)),
                b=pc.add_parameters((self.output_vocabulary.size(),))
            )
        )
        if self.is_bilstm:
            self.lstm_builder = dy.BiRNNBuilder(self.num_lstm_layers, self.embedded_input_vec_size, self.lstm_h_vec_size * 2, pc, dy.LSTMBuilder)
        else:
            self.lstm_builder = dy.LSTMBuilder(self.num_lstm_layers, self.embedded_input_vec_size, self.lstm_h_vec_size, pc)

        return pc

    def _build_network_for_input(self, inp):
        dy.renew_cg()
        if not self.is_bilstm:
            cur_lstm_state = self.lstm_builder.initial_state()
        else:
            cur_lstm_state = self.lstm_builder
        embeddings = [
            dy.concatenate([
                dy.lookup(self.params.input_lookups[field], self.input_vocabularies[field].get_index(token_data[field]))
                for field in self.input_vocabularies
            ])
            for token_data in inp
        ]
        lstm_outputs = cur_lstm_state.transduce(embeddings)
        outputs = []
        for lstm_out in lstm_outputs:
            cur_out = lstm_out
            for mlp_layer_params in self.params.mlp:
                cur_out = dy.dropout(cur_out, self.mlp_dropout_p)
                cur_out = dy.tanh(dy.parameter(mlp_layer_params.W) * cur_out + dy.parameter(mlp_layer_params.b))
            cur_out = dy.softmax(dy.parameter(self.params.softmax.W) * cur_out + dy.parameter(self.params.softmax.b))
            outputs.append(cur_out)
        return outputs

    def _build_loss(self, outputs, ys):
        losses = []
        for out, y in zip(outputs, ys):
            if y is not None:
                ss_ind = self.output_vocabulary.get_index(y)
                loss = -dy.log(dy.pick(out, ss_ind))
                losses.append(loss)
        return dy.esum(losses)

    def fit(self, samples, epochs=5, validation_split=0.2, show_progress=True, show_epoch_eval=True,
            mlp_dropout_p=0, evaluator=None):
        self.mlp_dropout_p = mlp_dropout_p

        test = samples[:int(len(samples)*validation_split)]
        train = samples[int(len(samples)*validation_split):]

        pc = self._build_network_params()
        trainer = dy.SimpleSGDTrainer(pc)
        for epoch in range(1, epochs + 1):
            train = list(train)
            random.shuffle(train)
            loss_sum = 0
            for ind, sample in enumerate(train):
                outputs = self._build_network_for_input(sample.xs)
                loss = self._build_loss(outputs, sample.ys)
                loss.forward()
                loss_sum += loss.value()
                loss.backward()
                trainer.update()
                if show_progress:
                    if int((ind + 1) / len(train) * 100) > int(ind / len(train) * 100):
                        per = int((ind + 1) / len(train) * 100)
                        print('\r\rEpoch %3d (%d%%): |' % (epoch, per) + '#' * per + '-' * (100 - per) + '|',)

            if evaluator and show_epoch_eval:
                print('Epoch %d complete, avg loss: %1.4f' % (epoch, loss_sum/len(train)))
                evaluator.evaluate(test, examples_to_show=0, predictor=self)

        print('--------------------------------------------')
        print('Training is complete (%d samples, %d epochs)' % (len(train), epochs))
        if evaluator:
            print('Test data evaluation:')
            evaluator.evaluate(test, examples_to_show=0, predictor=self)
            print('Training data evaluation:')
            evaluator.evaluate(train, examples_to_show=0, predictor=self)
        print('--------------------------------------------')

        return self

    def predict(self, sample_x, mask=None):
        if mask is None:
            mask = [True] * len(sample_x)
        outputs = self._build_network_for_input(sample_x)
        ys = []
        for token_ind, out in enumerate(outputs):
            ind = np.argmax(out.npvalue())
            predicted = self.output_vocabulary.get_word(ind) if mask[token_ind] else None
            ys.append(predicted)
        return ys
