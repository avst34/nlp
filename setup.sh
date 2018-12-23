#wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
#mkdir -p corenlp
#unzip stanford-corenlp-full-2017-06-09.zip -d corenlp

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
python embeddings/create_embeddings_txt.py wiki.en.vec models/supersenses/embeddings/wiki.en.chunked1

