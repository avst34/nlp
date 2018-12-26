wget -nc http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
mkdir -p corenlp
unzip  -n stanford-corenlp-full-2017-06-09.zip -d corenlp

wget -nc https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
python embeddings/create_embeddings_txt.py wiki.en.vec models/supersenses/embeddings/wiki.en.chunked

pip install requests
pip install h5py
