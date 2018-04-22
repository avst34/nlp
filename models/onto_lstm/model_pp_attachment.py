'''
This module defines various PP Attachment models and comes with a CLI to train and test them.
'''

import sys
import codecs
import argparse
import random
import json
import numpy
from overrides import overrides

from keras.layers import Input, Bidirectional
from keras.models import Model

from encoders import LSTMEncoder, OntoLSTMEncoder
from onto_attention import OntoAttentionLSTM
from index_data import DataProcessor
from preposition_model import PrepositionModel
from preposition_predictors import AttachmentPredictor


class PPAttachmentModel(PrepositionModel):
    '''
    Base class for PP Attachment models. Encoder for input phrases is not defined here. Subclasses
    need to do that.
    '''
    def __init__(self, tune_embedding, bidirectional, **kwargs):
        super(PPAttachmentModel, self).__init__(**kwargs)
        self.tune_embedding = tune_embedding
        self.bidirectional = bidirectional
        self.validation_split = 0.05
        self.model_name = "PP Attachment"
        self.custom_objects = {"AttachmentPredictor": AttachmentPredictor}

    def _get_input_layers(self, train_inputs):
        phrase_input_layer = Input(name="phrase", shape=train_inputs.shape[1:], dtype='int32')
        return phrase_input_layer

    def _get_output_layers(self, inputs, dropout, embedding_file, num_mlp_layers):
        encoded_phrase = self.encoder.get_encoded_phrase(inputs, dropout, embedding_file)
        predictor = AttachmentPredictor(name='attachment_predictor', proj_dim=20, composition_type='HPCT',
                                        num_hidden_layers=num_mlp_layers)
        outputs = predictor(encoded_phrase)
        return outputs

    @overrides
    def process_data(self, input_file, onto_aware, for_test=False):
        '''
        Reads an input file and makes input for training or testing.
        '''
        dataset_type = "test" if for_test else "training"
        print("Reading %s data" % dataset_type, file=sys.stderr)
        label_ind = []
        tagged_sentences = []
        max_sentence_length = 0
        all_sentence_lengths = []
        for line in open(input_file):
            lnstrp = line.strip()
            label, tagged_sentence = lnstrp.split("\t")
            sentence_length = len(tagged_sentence.split())
            all_sentence_lengths.append(sentence_length)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
            label_ind.append(int(label))
            tagged_sentences.append(tagged_sentence)
        if for_test:
            if not self.model:
                raise RuntimeError("Model not trained yet!")
            input_shape = self.model.get_input_shape_at(0)  # (num_sentences, num_words, ...)
            sentlenlimit = input_shape[1]
        else:
            sentlenlimit = max_sentence_length
        # We need to readjust the labels because padding would affect the sentence indices.
        for i in range(len(label_ind)):
            length = all_sentence_lengths[i]
            label_ind[i] += sentlenlimit - length
        if not for_test:
            # Shuffling so that when Keras does validation split, it is not always at the end.
            sentences_and_labels = list(zip(tagged_sentences, label_ind))
            random.shuffle(sentences_and_labels)
            tagged_sentences, label_ind = zip(*sentences_and_labels)
        print("Indexing %s data" % dataset_type, file=sys.stderr)
        inputs = self.data_processor.prepare_input(tagged_sentences, onto_aware=onto_aware,
                                                   sentlenlimit=sentlenlimit, for_test=for_test,
                                                   remove_singletons=False)
        labels = self.data_processor.make_one_hot(label_ind)
        return inputs, labels

    @overrides
    def write_predictions(self, inputs):
        '''
        Outputs predictions in a file named <model_name_prefix>.predictions.
        '''
        predictions = numpy.argmax(self.model.predict(inputs), axis=1)
        test_output_file = open("%s.predictions" % self.model_name_prefix, "w")
        for input_indices, prediction in zip(inputs, predictions):
            # The predictions are indices of words in padded sentences. We need to readjust them.
            padding_length = 0
            for index in input_indices:
                if numpy.all(index == 0):
                    padding_length += 1
                else:
                    break
            prediction = prediction - padding_length + 1  # +1 because the indices start at 1.
            print(prediction, file=test_output_file)


class LSTMAttachmentModel(PPAttachmentModel):
    '''
    A PP Attachment prediction model that uses an LSTM as the encoder.
    '''
    def __init__(self, **kwargs):
        super(LSTMAttachmentModel, self).__init__(**kwargs)
        self.model_name_prefix = "lstm_models/lstm_ppa_tune-embedding=%s_bi=%s" % (self.tune_embedding,
                                                                                   self.bidirectional)
        self.encoder = LSTMEncoder(self.data_processor, self.embed_dim, self.bidirectional, self.tune_embedding)
        self.custom_objects.update(self.encoder.get_custom_objects())


class OntoLSTMAttachmentModel(PPAttachmentModel):
    '''
    A PP Attachment prediction model that uses an OnotoLSTM as the encoder.
    '''
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, prep_senses_dir, **kwargs):
        super(OntoLSTMAttachmentModel, self).__init__(**kwargs)
        # Set self.data_processor again, now with the right arguments.
        process_preps = False if prep_senses_dir is None else True
        self.data_processor = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps,
                                            process_preps=process_preps, prep_senses_dir=prep_senses_dir)
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.attention_model = None  # Keras model with just embedding and encoder to output attention.
        self.set_sense_priors = set_sense_priors
        self.use_attention = use_attention
        use_prep_senses = False if prep_senses_dir is None else True
        self.encoder = OntoLSTMEncoder(self.num_senses, self.num_hyps, self.use_attention, self.set_sense_priors,
                                       data_processor=self.data_processor, embed_dim=self.embed_dim,
                                       bidirectional=self.bidirectional, tune_embedding=self.tune_embedding)
        self.model_name_prefix = ("ontolstm_models/ontolstm_ppa_att=%s_senses=%d_hyps=%d"
                                  "_sense-priors=%s_prep-senses=%s_tune-embedding=%s_bi=%s") % (
                                      str(self.use_attention), self.num_senses, self.num_hyps,
                                      str(set_sense_priors), str(use_prep_senses), str(self.tune_embedding),
                                      str(self.bidirectional))
        self.custom_objects.update(self.encoder.get_custom_objects())

    def get_attention(self, inputs):
        '''
        Takes inputs and returns pairs of synsets and corresponding attention values.
        '''
        if not self.attention_model:
            self.define_attention_model()
        attention_outputs = self.attention_model.predict(inputs)
        sent_attention_values = []
        for sentence_input, sentence_attention in zip(inputs, attention_outputs):
            word_attention_values = []
            for word_input, word_attention in zip(sentence_input, sentence_attention):
                # Size of word input is (senses, hyps+1)
                # Ignoring the last hyp index because that is just the word index pt there by
                # OntoAwareEmbedding for sense priors.
                if word_input.sum() == 0:
                    # This is just padding
                    continue
                word_input = word_input[:, :-1]  # removing last hyp index.
                sense_hyp_prod = self.num_senses * self.num_hyps
                assert len(word_attention) == sense_hyp_prod or len(word_attention) == 2 * sense_hyp_prod
                attention_per_sense = []
                if len(word_attention) == 2 * sense_hyp_prod:
                    # The encoder is Bidirectional. We have attentions from both directions.
                    forward_sense_attention = word_attention[:len(word_attention) // 2]
                    backward_sense_attention = word_attention[len(word_attention) // 2:]
                    processed_attention = zip(forward_sense_attention, backward_sense_attention)
                else:
                    # Encoder is not bidirectional
                    processed_attention = word_attention
                hyp_ind = 0
                while hyp_ind < len(processed_attention):
                    attention_per_sense.append(processed_attention[hyp_ind:hyp_ind+self.num_hyps])
                    hyp_ind += self.num_hyps

                sense_attention_values = []
                for sense_input, attention_per_hyp in zip(word_input, attention_per_sense):
                    hyp_attention_values = []
                    for hyp_input, hyp_attention in zip(sense_input, attention_per_hyp):
                        if hyp_input == 0:
                            continue
                        hyp_attention_values.append((self.data_processor.get_token_from_index(hyp_input,
                                                                                              onto_aware=True),
                                                     hyp_attention))
                    sense_attention_values.append(hyp_attention_values)
                word_attention_values.append(sense_attention_values)
            sent_attention_values.append(word_attention_values)
        return sent_attention_values

    def define_attention_model(self):
        '''
        Take necessary parts out of the model to get OntoLSTM attention.
        '''
        if not self.model:
            raise RuntimeError("Model not trained yet!")
        input_shape = self.model.get_input_shape_at(0)
        input_layer = Input(input_shape[1:], dtype='int32')  # removing batch size
        embedding_layer = None
        encoder_layer = None
        for layer in self.model.layers:
            if layer.name == "embedding":
                embedding_layer = layer
            elif layer.name == "onto_lstm":
                # We need to redefine the OntoLSTM layer with the learned weights and set return attention to True.
                # Assuming we'll want attention values for all words (return_sequences = True)
                if isinstance(layer, Bidirectional):
                    onto_lstm = OntoAttentionLSTM(input_dim=self.embed_dim, output_dim=self.embed_dim,
                                                  num_senses=self.num_senses, num_hyps=self.num_hyps,
                                                  use_attention=True, return_attention=True, return_sequences=True,
                                                  consume_less='gpu')
                    encoder_layer = Bidirectional(onto_lstm, weights=layer.get_weights())
                else:
                    encoder_layer = OntoAttentionLSTM(input_dim=self.embed_dim,
                                                      output_dim=self.embed_dim, num_senses=self.num_senses,
                                                      num_hyps=self.num_hyps, use_attention=True,
                                                      return_attention=True, return_sequences=True,
                                                      consume_less='gpu', weights=layer.get_weights())
                break
        if not embedding_layer or not encoder_layer:
            raise RuntimeError("Required layers not found!")
        attention_output = encoder_layer(embedding_layer(input_layer))
        self.attention_model = Model(inputs=input_layer, outputs=attention_output)
        print("Attention model summary:", file=sys.stderr)
        self.attention_model.summary()
        self.attention_model.compile(loss="mse", optimizer="sgd")  # Loss and optimizer do not matter!

    def print_attention_values(self, input_file, test_inputs, output_file):
        sent_attention_outputs = self.get_attention(test_inputs)
        tagged_sentences = [x.strip().split("\t")[1] for x in codecs.open(input_file).readlines()]
        outfile = codecs.open(output_file, "w", "utf-8")
        full_json_struct = []
        for sent_attention, tagged_sentence in zip(sent_attention_outputs, tagged_sentences):
            sent_json = {}
            sent_json["input"] = tagged_sentence
            sent_json["tokens"] = []
            tagged_words = tagged_sentence.split()
            for tagged_word, word_attention in zip(tagged_words, sent_attention):
                token_json = {}
                token_json["surface_form"] = tagged_word
                token_json["senses"] = []
                for sense_num, sense_attention in enumerate(word_attention):
                    if len(sense_attention) == 0:
                        continue
                    sense_json = {}
                    sense_json["id"] = sense_num
                    sense_json["hypernyms"] = []
                    for hyp_name, hyp_att in sense_attention:
                        if isinstance(hyp_att, tuple):
                            # Averaging forward and backward attention
                            sense_json["hypernyms"].append({hyp_name: {"forward": float(hyp_att[0]),
                                                                       "backward": float(hyp_att[1])}})
                        else:
                            sense_json["hypernyms"].append({hyp_name: float(hyp_att)})
                    token_json["senses"].append(sense_json)
                sent_json["tokens"].append(token_json)
            full_json_struct.append(sent_json)
        print(json.dumps(full_json_struct, indent=2), file=outfile)
        outfile.close()


def main():
    argparser = argparse.ArgumentParser(description="Train preposition phrase attachment model")
    argparser.add_argument('--train_file', type=str, help="TSV file with label and pos tagged phrase")
    argparser.add_argument('--embedding_file', type=str, help="Gzipped embedding file")
    argparser.add_argument('--embed_dim', type=int, help="Word/Synset vector size", default=50)
    argparser.add_argument('--bidirectional', help="Encode bidirectionally followed by pooling",
                           action='store_true')
    argparser.add_argument('--onto_aware', help="Use ontology aware encoder. "
                           "If this flag is not set, will use traditional encoder", action='store_true')
    argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default "
                           "2)", default=2)
    argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default "
                           "5)", default=5)
    argparser.add_argument('--prep_senses_dir', type=str, help="Directory containing preposition senses "
                           "(from Semeval07 Task 6)")
    argparser.add_argument('--set_sense_priors', help="Set an exponential prior on sense probabilities",
                           action='store_true')
    argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. "
                           "If this is not set, will use average concept representations",
                           action='store_true')
    argparser.add_argument('--test_file', type=str, help="Optionally provide test file for which accuracy will be computed")
    argparser.add_argument('--load_model_from_epoch', type=int, help="Load model from a specific epoch. Will load best model by default.")
    argparser.add_argument('--attention_output', type=str, help="Print attention values of the validation data in the given file")
    argparser.add_argument('--tune_embedding', help="Fine tune pretrained embedding (if provided)", action='store_true')
    argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
    argparser.add_argument('--num_mlp_layers', type=int, help="Number of mlp layers (default 0)", default=0)
    argparser.add_argument('--embedding_dropout', type=float, help="Dropout after embedding", default=0.0)
    argparser.add_argument('--encoder_dropout', type=float, help="Dropout after encoder", default=0.0)
    args = argparser.parse_args()
    if args.onto_aware:
        attachment_model = OntoLSTMAttachmentModel(num_senses=args.num_senses, num_hyps=args.num_hyps,
                                                   use_attention=args.use_attention,
                                                   set_sense_priors=args.set_sense_priors,
                                                   prep_senses_dir=args.prep_senses_dir,
                                                   embed_dim=args.embed_dim,
                                                   bidirectional=args.bidirectional,
                                                   tune_embedding=args.tune_embedding)
    else:
        attachment_model = LSTMAttachmentModel(embed_dim=args.embed_dim, bidirectional=args.bidirectional,
                                               tune_embedding=args.tune_embedding)

    ## Train model or load trained model
    if args.train_file is None:
        attachment_model.load_model(args.load_model_from_epoch)
    else:
        train_inputs, train_labels = attachment_model.process_data(args.train_file, onto_aware=args.onto_aware,
                                                                   for_test=False)
        dropout = {"embedding": args.embedding_dropout,
                   "encoder": args.encoder_dropout}
        attachment_model.train(train_inputs, train_labels, num_epochs=args.num_epochs,
                               dropout=dropout, num_mlp_layers=args.num_mlp_layers,
                               embedding_file=args.embedding_file)

    ## Test model
    if args.test_file is not None:
        test_inputs, test_labels = attachment_model.process_data(args.test_file, onto_aware=args.onto_aware,
                                                                 for_test=True)
        #attachment_model.test(test_inputs, test_labels)
        if args.attention_output is not None:
            attachment_model.print_attention_values(args.test_file, test_inputs, args.attention_output)

if __name__ == "__main__":
    main()
