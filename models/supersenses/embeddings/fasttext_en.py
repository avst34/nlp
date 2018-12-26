import os

from embeddings.embeddings_txt import EmbeddingsTxtReader

# FASTTEXT_EN = EmbeddingsTxtReader('/cs/usr/aviramstern/lab/muse/MUSE/data/wiki.en.chunked')
FASTTEXT_EN = EmbeddingsTxtReader(os.path.dirname(__file__) + '/wiki.en.chunked')

if __name__ == '__main__':
    print(FASTTEXT_EN.dim())
    print(FASTTEXT_EN.get('hello'))
    print(FASTTEXT_EN.get('my'))
    print(FASTTEXT_EN.get('name'))
    print(FASTTEXT_EN.get('is'))
    print(FASTTEXT_EN.get('Aviram'))