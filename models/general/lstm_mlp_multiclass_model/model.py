from collections import namedtuple

import random
import dynet as dy
import numpy as np

# There are more hidden parameters coming from the LSTMs
from vocabulary import Vocabulary

ModelOptimizedParams = namedtuple('ModelOptimizedParams', [
    'input_lookups',
    'mlp',
    'softmax'
])

MLPLayerParams = namedtuple('MLPParams', ['W', 'b'])

class LstmMlpMulticlassModel(object):

    Sample = namedtuple('Sample', ['xs', 'ys'])

    class SampleX:
        def __init__(self, fields, head_ind=None):
            self.fields = fields
            self.head_ind = head_ind

        def __getitem__(self, field):
            return self.fields[field]

        def items(self):
            return self.fields.items()

        def __iter__(self):
            return self.fields.__iter__()

        def keys(self):
            return self.fields.keys()


    class HyperParameters:
        def __init__(self,
                     input_fields,
                     input_embedding_dims,
                     input_embeddings_default_dim,
                     mlp_layers,
                     mlp_layer_dim,
                     lstm_h_dim,
                     num_lstm_layers,
                     is_bilstm,
                     use_head,
                     mlp_dropout_p
                     ):
            self.input_fields = input_fields
            self.input_embedding_dims = input_embedding_dims
            self.input_embeddings_default_dim = input_embeddings_default_dim
            self.mlp_layers = mlp_layers
            self.mlp_layer_dim = mlp_layer_dim
            self.lstm_h_dim = lstm_h_dim
            self.num_lstm_layers = num_lstm_layers
            self.is_bilstm = is_bilstm
            self.mlp_dropout_p = mlp_dropout_p
            self.use_head = use_head

    def __init__(self,
                 input_vocabularies=None,
                 output_vocabulary=None,
                 input_embeddings=None,
                 hyperparameters=None):
        hyperparameters = hyperparameters or LstmMlpMulticlassModel.HyperParameters(
            input_embedding_dims=None,
            input_embeddings_default_dim=10,
            mlp_layers=2,
            mlp_layer_dim=10,
            lstm_h_dim=40,
            num_lstm_layers=2,
            is_bilstm=True,
            use_head=True,
            mlp_dropout_p=0
        )

        self.input_vocabularies = input_vocabularies
        self.output_vocabulary = output_vocabulary
        self.input_embeddings = input_embeddings or {}

        input_embeddings_dims = {field: hyperparameters.input_embeddings_default_dim for field in hyperparameters.input_fields}
        input_embeddings_dims.update(hyperparameters.input_embedding_dims or {})
        input_embeddings_dims.update({
            field: len(list(self.input_embeddings[field].values())[0])
            for field in hyperparameters.input_fields
            if self.input_embeddings.get(field)
        })
        self.hyperparameters = LstmMlpMulticlassModel.HyperParameters(
            input_fields=hyperparameters.input_fields,
            input_embeddings_default_dim=None,
            input_embedding_dims=input_embeddings_dims,
            mlp_layers=hyperparameters.mlp_layers,
            mlp_layer_dim=hyperparameters.mlp_layer_dim,
            lstm_h_dim=hyperparameters.lstm_h_dim,
            num_lstm_layers=hyperparameters.num_lstm_layers,
            is_bilstm=hyperparameters.is_bilstm,
            mlp_dropout_p=hyperparameters.mlp_dropout_p,
            use_head=hyperparameters.use_head
        )

    def _build_network_params(self):
        pc = dy.ParameterCollection()

        if self.hyperparameters.use_head:
            mlp_input_dim = 2 * self.hyperparameters.lstm_h_dim + 1
        else:
            mlp_input_dim = self.hyperparameters.lstm_h_dim

        embedded_input_dim = sum(self.hyperparameters.input_embedding_dims.values())
        
        self.params = ModelOptimizedParams(
            input_lookups={
                field: pc.add_lookup_parameters((self.input_vocabularies[field].size(), self.hyperparameters.input_embedding_dims[field]))
                for field in self.input_vocabularies
            },
            mlp=[
                MLPLayerParams(
                    W=pc.add_parameters((self.hyperparameters.mlp_layer_dim, self.hyperparameters.mlp_layer_dim if i > 0 else mlp_input_dim)),
                    b=pc.add_parameters((self.hyperparameters.mlp_layer_dim,))
                )
                for i in range(self.hyperparameters.mlp_layers)
            ],
            softmax=MLPLayerParams(
                W=pc.add_parameters((self.output_vocabulary.size(), self.hyperparameters.mlp_layer_dim)),
                b=pc.add_parameters((self.output_vocabulary.size(),))
            )
        )

        for field, lookup_param in self.params.input_lookups.items():
            embeddings = (self.input_embeddings or {}).get(field) or {}
            for word, vector in embeddings.items():
                if self.input_vocabularies[field].has_word(word):
                    lookup_param.init_row(self.input_vocabularies[field].get_index(word), vector)

        if self.hyperparameters.is_bilstm:
            self.lstm_builder = dy.BiRNNBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc, dy.LSTMBuilder)
        else:
            self.lstm_builder = dy.LSTMBuilder(self.hyperparameters.num_lstm_layers, embedded_input_dim, self.hyperparameters.lstm_h_dim, pc)

        return pc

    def _build_network_for_input(self, inp):
        dy.renew_cg()
        if not self.hyperparameters.is_bilstm:
            cur_lstm_state = self.lstm_builder.initial_state()
        else:
            cur_lstm_state = self.lstm_builder
        embeddings = [
            dy.concatenate([
                dy.lookup(self.params.input_lookups[field], self.input_vocabularies[field].get_index(token_data[field]), update=not self.input_embeddings.get(field))
                for field in self.input_vocabularies
            ])
            for token_data in inp
        ]
        lstm_outputs = cur_lstm_state.transduce(embeddings)
        outputs = []
        for ind, lstm_out in enumerate(lstm_outputs):
            if self.hyperparameters.use_head:
                if inp[ind].head_ind is not None:
                    cur_out = dy.concatenate([lstm_out, dy.inputTensor([1]), lstm_outputs[inp[ind].head_ind]])
                else:
                    cur_out = dy.concatenate([lstm_out, dy.inputTensor([0]), dy.inputTensor([0] * self.hyperparameters.lstm_h_dim)])
            else:
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

    def _build_vocabularies(self, samples):
        if not self.input_vocabularies:
            self.input_vocabularies = {}
        for field in self.hyperparameters.input_fields:
            if not self.input_vocabularies.get(field):
                vocab = Vocabulary(field)
                vocab.add_words([x.fields.get(field) for s in samples for x in s.xs])
                self.input_vocabularies[field] = vocab

        if not self.output_vocabulary:
            vocab = Vocabulary('output')
            vocab.add_words([y for s in samples for y in s.ys])
            self.output_vocabulary = vocab

    def fit(self, samples, epochs=5, validation_split=0.2, show_progress=True, show_epoch_eval=True,
            evaluator=None):
        self._build_vocabularies(samples)
        pc = self._build_network_params()

        test = samples[:int(len(samples) * validation_split)]
        train = samples[int(len(samples) * validation_split):]

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
