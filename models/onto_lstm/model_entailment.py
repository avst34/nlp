import sys
import argparse
import pickle
import random
import codecs
import numpy
from overrides import overrides

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, Input, merge

from embedding import OntoAwareEmbedding
from encoders import LSTMEncoder, OntoLSTMEncoder
from pooling import AveragePooling, IntraAttention
from index_data import DataProcessor
from onto_attention import OntoAttentionLSTM, OntoAttentionNSE
from nse import NSE, MultipleMemoryAccessNSE, InputMemoryMerger, OutputSplitter

class EntailmentModel(object):
    def __init__(self, bidirectional=False, intra_attention=False, tune_embedding=False, **kwargs):
        self.data_processor = DataProcessor()
        if "embed_dim" in kwargs:
            self.embed_dim = kwargs["embed_dim"]
        else:
            self.embed_dim = 50
        self.bidirectional = bidirectional
        self.intra_attention = intra_attention
        self.tune_embedding = tune_embedding
        self.numpy_rng = numpy.random.RandomState(12345)
        self.label_map = {}  # Maps labels to integers.
        self.model = None
        self.best_epoch = 0  # index of the best epoch
        self.model_name_prefix = None
        self.custom_objects = None
        self.encoder_model = None

    def train(self, train_inputs, train_labels, num_epochs=20, mlp_size=1024, mlp_activation='relu',
              dropout=None, embedding_file=None, tune_embedding=True, num_mlp_layers=2,
              patience=5):
        '''
        train_inputs (list(numpy_array)): The two sentence inputs
        train_labels (numpy_array): One-hot matrix indicating labels
        num_epochs (int): Maximum number of epochs to run
        mlp_size (int): Dimensionality of each layer in the MLP
        dropout (dict(str->float)): Probabilities in Dropout layers after "embedding" and "encoder" (lstm)
        embedding (numpy): Optional pretrained embedding
        tune_embedding (bool): If pretrained embedding is given, tune it.
        patience (int): Early stopping patience
        '''
        if dropout is None:
            dropout = {}
        num_label_types = train_labels.shape[1]  # train_labels is of shape (num_samples, num_label_types)
        sent1_input_layer = Input(name='sent1', shape=train_inputs[0].shape[1:], dtype='int32')
        sent2_input_layer = Input(name='sent2', shape=train_inputs[1].shape[1:], dtype='int32')
        encoded_sent1, encoded_sent2 = self._get_encoded_sentence_variables(sent1_input_layer,
                                                                            sent2_input_layer, dropout,
                                                                            embedding_file, tune_embedding)
        concat_sent_rep = merge([encoded_sent1, encoded_sent2], mode='concat')
        mul_sent_rep = merge([encoded_sent1, encoded_sent2], mode='mul')
        diff_sent_rep = merge([encoded_sent1, encoded_sent2], mode=lambda l: l[0]-l[1],
                              output_shape=lambda l: l[0])
        # Use heuristic from Mou et al. (2015) to get final merged representation
        merged_sent_rep = merge([concat_sent_rep, mul_sent_rep, diff_sent_rep], mode='concat')
        current_projection = merged_sent_rep
        for i in range(num_mlp_layers):
            mlp_layer_i = Dense(output_dim=mlp_size, activation=mlp_activation,
                                name="%s_layer_%d" % (mlp_activation, i))
            current_projection = mlp_layer_i(current_projection)
        if dropout is not None:
            if "output" in dropout:
                current_projection = Dropout(dropout["output"])(current_projection)
        softmax = Dense(output_dim=num_label_types, activation='softmax', name='softmax_layer')
        label_probs = softmax(current_projection)
        model = Model(input=[sent1_input_layer, sent2_input_layer], output=label_probs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print("Entailment model summary:", file=sys.stderr)
        model.summary()
        best_accuracy = 0.0
        num_worse_epochs = 0
        for epoch_id in range(num_epochs):
            print("Epoch: %d" % epoch_id, file=sys.stderr)
            history = model.fit(train_inputs, train_labels, validation_split=0.1, nb_epoch=1)
            validation_accuracy = history.history['val_acc'][0]  # history['val_acc'] is a list of size nb_epoch
            if validation_accuracy > best_accuracy:
                self.save_model(epoch_id)
                self.best_epoch = epoch_id
                num_worse_epochs = 0
                best_accuracy = validation_accuracy
            elif validation_accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= patience:
                    print("Stopping training.", file=sys.stderr)
                    break
        self.save_best_model()

    def _get_encoded_sentence_variables(self, sent1_input_layer, sent2_input_layer, dropout,
                                        embedding_file, tune_embedding):
        if self.bidirectional:
            encoded_sent1_seq = self.encoder_model.get_encoded_phrase(sent1_input_layer, dropout=dropout,
                                                                      embedding=embedding_file)
            encoded_sent2_seq = self.encoder_model.get_encoded_phrase(sent2_input_layer, dropout=dropout,
                                                                      embedding=embedding_file)
            if self.intra_attention:
                pooling_layer = IntraAttention(name='intra_attention')
            else:
                pooling_layer = AveragePooling(name='average_pooling')
            encoded_sent1 = pooling_layer(encoded_sent1_seq)
            encoded_sent2 = pooling_layer(encoded_sent2_seq)
        else:
            encoded_sent1 = self.encoder_model.get_encoded_phrase(sent1_input_layer, dropout=dropout,
                                                                  embedding=embedding_file)
            encoded_sent2 = self.encoder_model.get_encoded_phrase(sent2_input_layer, dropout=dropout,
                                                                  embedding=embedding_file)
        return encoded_sent1, encoded_sent2

    def process_train_data(self, input_file, onto_aware):
        print("Reading training data", file=sys.stderr)
        label_ind = []
        tagged_sentences = []
        for line in open(input_file):
            lnstrp = line.strip()
            label, tagged_sentence = lnstrp.split("\t")
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)
            label_ind.append(self.label_map[label])
            tagged_sentences.append(tagged_sentence)
        # Shuffling so that when Keras does validation split, it is not always at the end.
        sentences_and_labels = zip(tagged_sentences, label_ind)
        random.shuffle(sentences_and_labels)
        tagged_sentences, label_ind = zip(*sentences_and_labels)
        print("Indexing training data", file=sys.stderr)
        train_inputs = self.data_processor.prepare_paired_input(tagged_sentences, onto_aware=onto_aware,
                                                                for_test=False, remove_singletons=True)
        train_labels = self.data_processor.make_one_hot(label_ind)
        return train_inputs, train_labels

    def process_test_data(self, input_file, onto_aware, is_labeled=True):
        if not self.model:
            raise RuntimeError, "Model not trained yet!"
        print("Reading test data", file=sys.stderr)
        label_ind = []
        tagged_sentences = []
        for line in open(input_file):
            lnstrp = line.strip()
            if is_labeled:
                label, tagged_sentence = lnstrp.split("\t")
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                label_ind.append(self.label_map[label])
            else:
                tagged_sentence = lnstrp
            tagged_sentences.append(tagged_sentence)
        print("Indexing test data", file=sys.stderr)
        # Infer max sentence length if the model is trained
        input_shape = self.model.get_input_shape_at(0)[0]  # take the shape of the first of two inputs at 0.
        sentlenlimit = input_shape[1]  # (num_sentences, num_words, num_senses, num_hyps)
        test_inputs = self.data_processor.prepare_paired_input(tagged_sentences, onto_aware=onto_aware,
                                                               sentlenlimit=sentlenlimit, for_test=True)
        test_labels = self.data_processor.make_one_hot(label_ind)
        return test_inputs, test_labels

    def test(self, inputs, targets):
        if not self.model:
            raise RuntimeError, "Model not trained!"
        metrics = self.model.evaluate(inputs, targets)
        print("Test accuracy: %.4f" % (metrics[1])  # The first metric is loss., file=sys.stderr)
        predictions = numpy.argmax(self.model.predict(inputs), axis=1)
        rev_label_map = {ind: label for label, ind in self.label_map.items()}
        predicted_labels = [rev_label_map[pred] for pred in predictions]
        return predicted_labels

    def save_model(self, epoch):
        '''
        Saves the current model using the epoch id to identify the file.
        '''
        self.model.save("%s_%d.model" % (self.model_name_prefix, epoch))
        pickle.dump(self.data_processor, open("%s.dataproc" % self.model_name_prefix, "wb"))
        pickle.dump(self.label_map, open("%s.labelmap" % self.model_name_prefix, "wb"))

    def save_best_model(self):
        '''
        Copies the model corresponding to the best epoch as the final model file.
        '''
        from shutil import copyfile
        best_model_file = "%s_%d.model" % (self.model_name_prefix, self.best_epoch)
        final_model_file = "%s.model" % self.model_name_prefix
        copyfile(best_model_file, final_model_file)

    def load_model(self, epoch=None):
        '''
        Loads a saved model. If epoch id is provided, will load the corresponding model. Or else,
        will load the best model.
        '''
        if not epoch:
            self.model = load_model("%s.model" % self.model_name_prefix,
                                    custom_objects=self.custom_objects)
        else:
            self.model = load_model("%s_%d.model" % (self.model_name_prefix, epoch),
                                    custom_objects=self.custom_objects)
        self.data_processor = pickle.load(open("%s.dataproc" % self.model_name_prefix, "rb"))
        self.label_map = pickle.load(open("%s.labelmap" % self.model_name_prefix, "rb"))


class LSTMEntailmentModel(EntailmentModel):
    def __init__(self, **kwargs):
        super(LSTMEntailmentModel, self).__init__(**kwargs)
        # If bidirectional, we'll do pooling here. So make the encoder return sequences iff bidirectional.
        self.encoder_model = LSTMEncoder(data_processor=self.data_processor, embed_dim=self.embed_dim,
                                         bidirectional=self.bidirectional,
                                         tune_embedding=self.tune_embedding,
                                         return_sequences=self.bidirectional)
        self.model_name_prefix = "lstm_ent_bi=%s_pool-att=%s" % (self.bidirectional, self.intra_attention)
        self.custom_objects = {}
        if self.bidirectional:
            if self.intra_attention:
                self.custom_objects["IntraAttention"] = IntraAttention
            else: 
                self.custom_objects["AveragePooling"] = AveragePooling


class OntoLSTMEntailmentModel(EntailmentModel):
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, **kwargs):
        super(OntoLSTMEntailmentModel, self).__init__(**kwargs)
        # Set self.data_processor again, now with the right arguments.
        self.data_processor = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps)
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.attention_model = None  # Keras model with just embedding and encoder to output attention.
        self.set_sense_priors = set_sense_priors
        self.use_attention = use_attention
        # If bidirectional, we'll do pooling here. So make the encoder return sequences iff bidirectional.
        self.encoder_model = OntoLSTMEncoder(num_senses=num_senses, num_hyps=num_hyps,
                                             use_attention=use_attention, set_sense_priors=set_sense_priors,
                                             data_processor=self.data_processor, embed_dim=self.embed_dim,
                                             bidirectional=self.bidirectional,
                                             tune_embedding=self.tune_embedding,
                                             return_sequences=self.bidirectional)
        self.model_name_prefix = "ontolstm_ent_att=%s_senses=%d_hyps=%d_sense-priors=%s_tune-embedding=%s_bi=%s_pool-att=%s" % (
            str(self.use_attention), self.num_senses, self.num_hyps, str(set_sense_priors), str(self.tune_embedding),
            str(self.bidirectional), str(self.intra_attention))
        self.custom_objects = {"OntoAttentionLSTM": OntoAttentionLSTM, "OntoAwareEmbedding": OntoAwareEmbedding}
        if self.bidirectional:
            if self.intra_attention:
                self.custom_objects["IntraAttention"] = IntraAttention
            else: 
                self.custom_objects["AveragePooling"] = AveragePooling

    def get_encoder(self, return_sequences=False):
        lstm = OntoAttentionLSTM(input_dim=self.embed_dim, output_dim=self.embed_dim, num_senses=self.num_senses,
                                 num_hyps=self.num_hyps, use_attention=self.use_attention, consume_less="gpu",
                                 return_sequences=return_sequences, name="onto_lstm")
        return lstm

    def get_attention(self, inputs):
        # Takes inputs and returns pairs of synsets and corresponding attention values.
        if not self.attention_model:
            self.define_attention_model()
        attention_outputs = self.attention_model.predict(inputs)
        sent_attention_values = []
        for sentence_input, sentence_attention in zip(inputs, attention_outputs):
            word_attention_values = []
            for word_input, word_attention in zip(sentence_input, sentence_attention):
                if word_input.sum() == 0:
                    # This is just padding
                    continue
                sense_attention_values = []
                for sense_input, sense_attention in zip(word_input, word_attention):
                    if sense_input.sum() == 0:
                        continue
                    hyp_attention_values = []
                    for hyp_input, hyp_attention in zip(sense_input, sense_attention):
                        if hyp_input == 0:
                            continue
                        hyp_attention_values.append((self.data_processor.get_token_from_index(hyp_input,
                                                        onto_aware=True), hyp_attention))
                    sense_attention_values.append(hyp_attention_values)
                word_attention_values.append(sense_attention_values)
            sent_attention_values.append(word_attention_values)
        return sent_attention_values

    def define_attention_model(self):
        # Take necessary parts out of the entailment model to get OntoLSTM attention.
        if not self.model:
            raise RuntimeError, "Model not trained yet!"
        # We need just one input to get attention. input_shape_at(0) gives a list with two shapes.
        input_shape = self.model.get_input_shape_at(0)[0]
        input_layer = Input(input_shape[1:], dtype='int32')  # removing batch size
        embedding_layer = None
        encoder_layer = None
        for layer in self.model.layers:
            if layer.name == "embedding":
                embedding_layer = layer
            elif layer.name == "encoder":
                # We need to redefine the OntoLSTM layer with the learned weights and set return attention to True.
                # Assuming we'll want attention values for all words (return_sequences = True)
                encoder_layer = OntoAttentionLSTM(input_dim=self.embed_dim,
                                                  output_dim=self.embed_dim, num_senses=self.num_senses,
                                                  num_hyps=self.num_hyps, use_attention=True,
                                                  return_attention=True, return_sequences=True,
                                                  weights=layer.get_weights())
        if not embedding_layer or not encoder_layer:
            raise RuntimeError, "Required layers not found!"
        attention_output = encoder_layer(embedding_layer(input_layer))
        self.attention_model = Model(input=input_layer, output=attention_output)
        self.attention_model.compile(loss="mse", optimizer="sgd")  # Loss and optimizer do not matter!

    def print_attention_values(self, input_file, test_inputs, output_file):
        onto_aware = True
        sent1_attention_outputs = self.get_attention(test_inputs[0])
        sent2_attention_outputs = self.get_attention(test_inputs[1])
        tagged_sentences = [x.strip().split("\t")[1] for x in codecs.open(input_file).readlines()]
        outfile = codecs.open(output_file, "w", "utf-8")
        for sent1_attention, sent2_attention, tagged_sentence in zip(sent1_attention_outputs, sent2_attention_outputs, tagged_sentences):
            print(tagged_sentence, file=outfile)
            print("Sentence 1:", file=outfile)
            for word_attention in sent1_attention:
                for sense_attention in word_attention:
                    print(" ".join(["%s:%f" % (hyp, hyp_att) for hyp, hyp_att in sense_attention]), file=outfile)
                print("", file=outfile))
            print("\nSentence 2:", file=outfile)
            for word_attention in sent2_attention:
                for sense_attention in word_attention:
                    print(" ".join(["%s:%f" % (hyp, hyp_att) for hyp, hyp_att in sense_attention]), file=outfile)
                print("", file=outfile)
        outfile.close()


class NSEEntailmentModel(EntailmentModel):
    def __init__(self, shared_memory=True, **kwargs):
        super(NSEEntailmentModel, self).__init__(**kwargs)
        self.shared_memory = shared_memory
        self.model_name_prefix = "nse_ent_shared_mem=%s" % self.shared_memory
        self.custom_objects = {"NSE": NSE}
        if self.shared_memory:
            self.custom_objects["MultipleMemoryAccessNSE"] = MultipleMemoryAccessNSE

    @overrides
    def _get_encoded_sentence_variables(self, sent1_input_layer, sent2_input_layer, dropout,
                                        embedding_file, tune_embedding):
        if embedding_file is None:
            if not tune_embedding:
                print("Pretrained embedding is not given. Setting tune_embedding to True.", file=sys.stderr)
                tune_embedding = True
            embedding = None
        else:
            # Put the embedding in a list for Keras to treat it as initiali weights of the embeddign layer.
            embedding = [self.data_processor.get_embedding_matrix(embedding_file, onto_aware=False)]
        vocab_size = self.data_processor.get_vocab_size(onto_aware=False)
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, weights=embedding,
                                    trainable=tune_embedding, mask_zero=True, name="embedding")
        embedded_sent1 = embedding_layer(sent1_input_layer)
        embedded_sent2 = embedding_layer(sent2_input_layer)
        if "embedding" in dropout:
            embedded_sent1 = Dropout(dropout["embedding"])(embedded_sent1)
            embedded_sent2 = Dropout(dropout["embedding"])(embedded_sent2)
        if self.shared_memory:
            encoder = MultipleMemoryAccessNSE(output_dim=self.embed_dim, return_mode="output_and_memory",
                                              name="encoder")
            mmanse_sent1_input = InputMemoryMerger(name="merge_sent1_input")([embedded_sent1, embedded_sent2])
            encoded_sent1_and_memory = encoder(mmanse_sent1_input)
            encoded_sent1 = OutputSplitter("output", name="get_sent1_output")(encoded_sent1_and_memory)
            shared_memory = OutputSplitter("memory", name="get_shared_memory")(encoded_sent1_and_memory)
            mmanse_sent2_input = InputMemoryMerger(name="merge_sent2_input")([embedded_sent2, shared_memory])
            encoded_sent2_and_memory = encoder(mmanse_sent2_input)
            encoded_sent2 = OutputSplitter("output", name="get_sent2_output")(encoded_sent2_and_memory)
        else:
            encoder = NSE(output_dim=self.embed_dim, name="encoder")
            encoded_sent1 = encoder(embedded_sent1)
            encoded_sent2 = encoder(embedded_sent2)
        if "encoder" in dropout:
            encoded_sent1 = Dropout(dropout["encoder"])(encoded_sent1)
            encoded_sent2 = Dropout(dropout["encoder"])(encoded_sent2)
        return encoded_sent1, encoded_sent2


class OntoNSEEntailmentModel(OntoLSTMEntailmentModel):
    def __init__(self, **kwargs):
        super(OntoNSEEntailmentModel, self).__init__(**kwargs)
        tune_embedding = kwargs["tune_embedding"]
        self.custom_objects = {"OntoAttentionNSE": OntoAttentionNSE, "OntoAwareEmbedding": OntoAwareEmbedding}
        self.model_name_prefix = "ontonse_ent_att=%s_senses=%d_hyps=%d_sense-priors=%s_tune-embedding=%s" % (
            str(self.use_attention), self.num_senses, self.num_hyps, str(self.set_sense_priors),
            str(tune_embedding))

    def get_encoder(self, bidirectional=False):
        if bidirectional:
            raise NotImplementedError, "Bidirectional NSE not implemented yet. But do you really want it?"
        encoder = OntoAttentionNSE(self.num_senses, self.num_hyps, use_attention=self.use_attention,
                                   return_attention=False, output_dim=self.embed_dim, name="onto_nse")
        return encoder


def main():
    argparser = argparse.ArgumentParser(description="Train entailment model")
    argparser.add_argument('--train_file', type=str, help="TSV file with label, premise, hypothesis in three columns")
    argparser.add_argument('--embedding_file', type=str, help="Gzipped embedding file")
    argparser.add_argument('--embed_dim', type=int, help="Word/Synset vector size", default=50)
    argparser.add_argument('--encoder', type=str, help="LSTM or NSE?", default="lstm")
    argparser.add_argument('--bidirectional', help="Encode bidirectionally followed by pooling", action='store_true')
    argparser.add_argument('--intra_attention', help="Use intra-attention to pool the output of bi-RNN",
                           action='store_true')
    argparser.add_argument('--nse_shared_memory', help="Use shared memory NSE id encoder is NSE", action='store_true')
    argparser.add_argument('--onto_aware', help="Use ontology aware encoder. If this flag is not set, will use traditional encoder", action='store_true')
    argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
    argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
    argparser.add_argument('--set_sense_priors', help="Set an exponential prior on sense probabilities", action='store_true')
    argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. If this flag is not set, will use average concept representations", action='store_true')
    argparser.add_argument('--test_file', type=str, help="Optionally provide test file for which accuracy will be computed")
    argparser.add_argument('--load_model_from_epoch', type=int, help="Load model from a specific epoch. Will load best model by default.")
    argparser.add_argument('--attention_output', type=str, help="Print attention values of the validation data in the given file")
    argparser.add_argument('--tune_embedding', help="Fine tune pretrained embedding (if provided)", action='store_true')
    argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
    argparser.add_argument('--mlp_size', type=int, help="Size of each layer in MLP (default 1024)", default=1024)
    argparser.add_argument('--num_mlp_layers', type=int, help="Number of mlp layers (default 2)", default=2)
    argparser.add_argument('--mlp_activation', type=str, help="MLP activation (default relu)", default='relu')
    argparser.add_argument('--embedding_dropout', type=float, help="Dropout after embedding", default=0.0)
    argparser.add_argument('--encoder_dropout', type=float, help="Dropout after encoder", default=0.0)
    argparser.add_argument('--output_dropout', type=float, help="Dropout after encoder", default=0.0)
    args = argparser.parse_args()
    if args.encoder == "lstm":
        if args.onto_aware:
            entailment_model = OntoLSTMEntailmentModel(num_senses=args.num_senses, num_hyps=args.num_hyps,
                                                       use_attention=args.use_attention,
                                                       set_sense_priors=args.set_sense_priors,
                                                       embed_dim=args.embed_dim,
                                                       bidirectional=args.bidirectional,
                                                       intra_attention=args.intra_attention,
                                                       tune_embedding=args.tune_embedding)
        else:
            entailment_model = LSTMEntailmentModel(embed_dim=args.embed_dim,
                                                   bidirectional=args.bidirectional,
                                                   tune_embedding=args.tune_embedding,
                                                   intra_attention=args.intra_attention)
    else:
        if args.onto_aware:
            if args.nse_shared_memory:
                raise NotImplementedError, "OntoMMANSEEntailmentModel coming soon"
            entailment_model = OntoNSEEntailmentModel(num_senses=args.num_senses, num_hyps=args.num_hyps,
                                                      use_attention=args.use_attention,
                                                      set_sense_priors=args.set_sense_priors,
                                                      embed_dim=args.embed_dim, tune_embedding=args.tune_embedding)
        else: 
            entailment_model = NSEEntailmentModel(embed_dim=args.embed_dim,
                                                  shared_memory=args.nse_shared_memory,
                                                  tune_embedding=args.tune_embedding)

    ## Train model or load trained model
    if args.train_file is None:
        entailment_model.load_model(args.load_model_from_epoch)
    else:
        train_inputs, train_labels = entailment_model.process_train_data(args.train_file, onto_aware=args.onto_aware)
        dropout = {"embedding": args.embedding_dropout,
                   "encoder": args.encoder_dropout,
                   "output": args.output_dropout}
        if args.onto_aware:
            entailment_model.train(train_inputs, train_labels, num_epochs=args.num_epochs,
                                   mlp_size=args.mlp_size, mlp_activation=args.mlp_activation,
                                   dropout=dropout, num_mlp_layers=args.num_mlp_layers,
                                   embedding_file=args.embedding_file, tune_embedding=args.tune_embedding)
        else:
            # Same as above, except no attention.
            entailment_model.train(train_inputs, train_labels, num_epochs=args.num_epochs,
                                   mlp_size=args.mlp_size, mlp_activation=args.mlp_activation,
                                   dropout=dropout, num_mlp_layers=args.num_mlp_layers,
                                   embedding_file=args.embedding_file, tune_embedding=args.tune_embedding)
    
    ## Test model
    if args.test_file is not None:
        test_inputs, test_labels = entailment_model.process_test_data(args.test_file, onto_aware=args.onto_aware)
        test_predictions = entailment_model.test(test_inputs, test_labels)
        if args.attention_output is not None:
            entailment_model.print_attention_values(args.test_file, test_inputs, args.attention_output)

if __name__ == "__main__":
    main()
