import argparse

from models.supersenses.lstm_mlp_supersenses_model import LstmMlpSupersensesModel
from models.supersenses.preprocessing import preprocess_sentence
from models.supersenses.preprocessing.corenlp import CoreNLPServer


def main(in_file, out_file, model_path):
    corenlp = CoreNLPServer()
    corenlp.start()
    try:
        model = LstmMlpSupersensesModel.load(model_path)
        for sent in in_file:
            sample = preprocess_sentence(sent)
            pred = model.predict(sample.xs, [x.identified_for_pss  for x in sample.xs])
            out_file.write(' '.join([x.token + (':%s~%s' % (y.supersense_role, y.supersense_func) if y.supersense_role else ':_~_' if x.mwe_start_ind is not None and x.mwe_start_ind != x.ind and pred[x.mwe_start_ind].supersense_role else "") for x, y in zip(sample.xs, pred)]) + '\n')
    finally:
        corenlp.stop()


argparser = argparse.ArgumentParser(description="Preposition Supersenses Predictor")
argparser.add_argument('--file', type=str, help="Input sentences, one sentence per line", required=True)
argparser.add_argument('--model_path', type=str, help="Path to the pretrained model", required=True)
argparser.add_argument('--out_file', type=str, help="Output file with tagged sentences", required=True)

args = argparser.parse_args()


if __name__ == '__main__':
    main(open(args.file), open(args.out_file, 'w'), args.model_path)