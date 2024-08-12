from collections import Counter
from operator import itemgetter

import numpy as np
from numpy.linalg import norm


def ngrams(string, n):
    return (string[i:i+n] for i in range(len(string)-n+1))


def count_ngrams(strings, n):
    ngram_freq = Counter()
    for text in strings:
        ngram_freq.update(ngrams(text, n))

    ngram_ids = {
        ngr : i \
        for i, (ngr, freq) in enumerate(sorted(
            ngram_freq.items(), key=itemgetter(1), reverse=True)) }
    return ngram_ids


def vectorize(strings, n=2, dim=200, ngram_ids=None,
              normalize=True, weighting='plain'):
    # TODO make the progress printer optional

    if ngram_ids is None:
        ngram_ids = count_ngrams(strings, n)

    m = np.zeros((len(strings), dim), dtype=np.float32)
    lengths = np.zeros(len(strings), dtype=m.dtype).reshape((len(strings), 1))
    for i, string in enumerate(strings):
        string_ngr_ids = [ngram_ids[ngr] for ngr in ngrams(string, n) \
                          if ngr in ngram_ids]
        m[i,] = np.bincount([x for x in string_ngr_ids if x < dim], minlength=dim)
        if normalize:
            # y -- a vector of non-zero bigram counts (no matter which bigrams)
            y = np.bincount(np.unique(string_ngr_ids, return_inverse=True)[1])
            lengths[i] = norm(y)
    if weighting == 'sqrt':
        m = np.sqrt(m)
    elif weighting == 'binary':
        m = np.asarray(m > 0, dtype=np.float32)
    if normalize:
        m = np.divide(m, lengths, where=lengths > 0)
    return m

