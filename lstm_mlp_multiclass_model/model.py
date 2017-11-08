from collections import namedtuple

import random
import dynet as dy
import numpy as np

# There are more hidden parameters coming from the LSTMs
Sample = namedtuple('Sample', ['xs', 'ys'])

class HyperParameters:

    def __init__(self,
            input_embedding_sizes=None,
            input_embeddings_default_size=10,
            mlp_layers=2,
            mlp_layer_size=10,
            lstm_h_vec_size=10,
            num_lstm_layers=2,
            is_bilstm=True,
            mlp_dropout_p=0
    ):
        super().__setattr__('input_embedding_sizes', input_embedding_sizes)
        super().__setattr__('input_embeddings_default_size', input_embeddings_default_size)
        super().__setattr__('mlp_layers', mlp_layers)
        super().__setattr__('mlp_layer_size', mlp_layer_size)
        super().__setattr__('lstm_h_vec_size', lstm_h_vec_size)
        super().__setattr__('num_lstm_layers', num_lstm_layers)
        super().__setattr__('is_bilstm', is_bilstm)
        super().__setattr__('mlp_dropout_p', mlp_dropout_p)

    def __setattr__(self, *args):
        raise TypeError

    def __delattr__(self, *args):
        raise TypeError

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'mlp',
    'softmax'
])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class LstmMlpMulticlassModel(object):

    def __init__(self,
                 input_vocabularies,
                 output_vocabulary,
                 input_embeddings=None,
                 hyperparameters=HyperParameters()):
        self.input_vocabularies = input_vocabularies
        self.output_vocabulary = output_vocabulary
        self.input_embeddings = input_embeddings or {}

        input_embeddings_sizes = {field: hyperparameters.input_embeddings_default_size for field in input_vocabularies}
        input_embeddings_sizes.update(hyperparameters.input_embedding_sizes or {})
        input_embeddings_sizes.update({
            field: len(list(self.input_embeddings[field].values())[0])
            for field in self.input_vocabularies
            if self.input_embeddings.get(field)
        })
        self.hyperparameters = HyperParameters(
            input_embedding_sizes=input_embeddings_sizes,
            mlp_layers=hyperparameters.mlp_layers,
            mlp_layer_size=hyperparameters.mlp_layer_size,
            lstm_h_vec_size=hyperparameters.lstm_h_vec_size,
            num_lstm_layers=hyperparameters.num_lstm_layers,
            is_bilstm=hyperparameters.is_bilstm,
            mlp_dropout_p=hyperparameters.mlp_dropout_p            
        )

        self.validate_config()

    def validate_config(self):
        if set(self.input_vocabularies.keys()) != set(self.hyperparameters.input_embedding_sizes.keys()):
            raise Exception("Mismatch between input vocabularies and input_embedding_sizes")

    def _build_network_params(self):
        pc = dy.ParameterCollection()
        
        lstm_cell_output_size = self.hyperparameters.lstm_h_vec_size * (2 if self.hyperparameters.is_bilstm else 1)
        embedded_input_vec_size = sum(self.hyperparameters.input_embedding_sizes.values())
        
        self.params = ModelOptimizedParams(
            input_lookups={
                field: pc.add_lookup_parameters((self.input_vocabularies[field].size(), self.hyperparameters.input_embedding_sizes[field]))
                for field in self.input_vocabularies
            },
            mlp=[
                MLPLayerParams(
                    W=pc.add_parameters((self.hyperparameters.mlp_layer_size, self.hyperparameters.mlp_layer_size if i > 0 else lstm_cell_output_size)),
                    b=pc.add_parameters((self.hyperparameters.mlp_layer_size,))
                )
                for i in range(self.hyperparameters.mlp_layers)
            ],
            softmax=MLPLayerParams(
                W=pc.add_parameters((self.output_vocabulary.size(), self.hyperparameters.mlp_layer_size)),
                b=pc.add_parameters((self.output_vocabulary.size(),))
            )
        )

        for field, lookup_param in self.params.input_lookups.items():
            embeddings = (self.input_embeddings or {}).get(field) or {}
            for word, vector in embeddings.items():
                lookup_param.init_row(self.input_vocabularies[field].get_index(word), vector)

        if self.hyperparameters.is_bilstm:
            self.lstm_builder = dy.BiRNNBuilder(self.hyperparameters.num_lstm_layers, embedded_input_vec_size, self.hyperparameters.lstm_h_vec_size * 2, pc, dy.LSTMBuilder)
        else:
            self.lstm_builder = dy.LSTMBuilder(self.hyperparameters.num_lstm_layers, embedded_input_vec_size, self.hyperparameters.lstm_h_vec_size, pc)

        return pc

    def _build_network_for_input(self, inp):
        dy.renew_cg()
        if not self.hyperparameters.is_bilstm:
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
                cur_out = dy.dropout(cur_out, self.hyperparameters.mlp_dropout_p)
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
            evaluator=None):

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

    def predict(self, sample_xs, mask=None):
        if mask is None:
            mask = [True] * len(sample_xs)
        outputs = self._build_network_for_input(sample_xs)
        ys = []
        for token_ind, out in enumerate(outputs):
            ind = np.argmax(out.npvalue())
            predicted = self.output_vocabulary.get_word(ind) if mask[token_ind] else None
            ys.append(predicted)
        return ys
