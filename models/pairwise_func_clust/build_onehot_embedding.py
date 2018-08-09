def build_onehot_embedding(vocab):
    n_words = vocab.size()
    embeddings = {}
    for word in vocab.all_words():
        word_ind = vocab.get_index(word)
        vec = [0] * n_words
        vec[word_ind] = 1
        embeddings[word] = vec
    return embeddings
