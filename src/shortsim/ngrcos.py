from collections import Counter
from operator import itemgetter

import numpy as np
from numpy.linalg import norm


def ngrams(string, n):
    return (string[i:i+n] for i in range(len(string)-n+1))


def determine_top_ngrams(strings, n, dim):
    ngram_freq = Counter()
    for text in strings:
        ngram_freq.update(ngrams(text, n))

    ngram_ids = {
        ngr : i \
        for i, (ngr, freq) in enumerate(sorted(
            ngram_freq.items(), key=itemgetter(1), reverse=True)[:dim]) }
    return ngram_ids


def vectorize(strings, n=2, dim=200, ngram_ids=None,
              normalize=True, weighting='plain'):
    # TODO make the progress printer optional

    if ngram_ids is None:
        ngram_ids = determine_top_ngrams(strings, n, dim)

    m = np.zeros((len(strings), dim), dtype=np.float32)
    for i, string in enumerate(strings):
        string_ngr_ids = [ngram_ids[ngr] for ngr in ngrams(string, n) \
                          if ngr in ngram_ids]
        m[i,] = np.bincount(string_ngr_ids, minlength=dim)
    if weighting == 'sqrt':
        m = np.sqrt(m)
    elif weighting == 'binary':
        m = np.asarray(m > 0, dtype=np.float32)
    if normalize:
        lengths = norm(m, axis=1).reshape((m.shape[0], 1))
        m = np.divide(m, lengths, where=lengths > 0)
    return m

