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


# FIXME
# - change the arguments and return values to a more sensible format
def vectorize(verses, n=2, dim=200, min_ngrams=10, ngram_ids=None,
              normalize=True, weighting='plain'):
    # FIXME memory is being wasted here by storing v_ids and verses again
    # TODO make the progress printer optional

    if ngram_ids is None:
        ngram_ids = determine_top_ngrams(map(itemgetter(1), verses), n, dim)

    v_ids, v_texts, rows = [], [], []
    for (v_id, text) in verses:
        v_ngr_ids = [ngram_ids[ngr] for ngr in ngrams(text, n) \
                     if ngr in ngram_ids]
        if len(v_ngr_ids) >= min_ngrams:
            row = np.zeros(dim, dtype=np.float32)
            for ngr_id in v_ngr_ids:
                row[ngr_id] += 1
            rows.append(row)
            v_ids.append(v_id)
            v_texts.append(text)
    m = np.vstack(rows)
    if weighting == 'sqrt':
        m = np.sqrt(m)
    elif weighting == 'binary':
        m = np.asarray(m > 0, dtype=np.float32)
    if normalize:
        m = np.divide(m, norm(m, axis=1).reshape((m.shape[0], 1)))
    return v_ids, v_texts, ngram_ids, m