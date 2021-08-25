import argparse
from collections import defaultdict
from operator import itemgetter
import sys
import tqdm
import warnings

import faiss
import numpy as np
from numpy.linalg import norm


# TODO refactor some of the functionality into a library, so that it can
# also be called from Python code.


def ngrams(string, n):
    return (string[i:i+n] for i in range(len(string)-n+1))


def determine_top_ngrams(verses, n, dim):
    ngram_freq = defaultdict(lambda: 0)
    for text in map(itemgetter(1), verses):
        for ngr in ngrams(text, n):
            ngram_freq[ngr] += 1

    ngram_ids = {
        ngr : i \
        for i, (ngr, freq) in enumerate(sorted(
            ngram_freq.items(), key=itemgetter(1), reverse=True)[:dim]) }
    return ngram_ids


def vectorize(verses, ngram_ids, n=2, dim=200, min_ngrams=10):
    # FIXME memory is being wasted here by storing v_ids and verses again
    # TODO make the progress printer optional
    v_ids, v_texts, rows = [], [], []
    for (v_id, text) in tqdm.tqdm(verses):
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
    m = np.divide(m, norm(m, axis=1).reshape((m.shape[0], 1)))
    return v_ids, v_texts, m


def find_similarities(index, m, k, threshold, query_size, print_progress):
    if print_progress:
        progressbar = tqdm.tqdm(total=m.shape[0])
    for i in range(0, m.shape[0], query_size):
        query = range(i, min(m.shape[0], i+query_size))
        D, I = index.search(m[query,], k)
        for i, q in enumerate(query):
            for j in range(k):
                if q != I[i,j] and D[i,j] >= threshold:
                    yield (q, I[i,j], D[i,j])
        if print_progress:
            progressbar.update(D.shape[0])


def read_verses(fp):
    result = []
    for line in fp:
        spl = line.rstrip().split('\t')
        if len(spl) < 2: continue
        v_id, text = spl[0], spl[1]
        result.append((v_id, text))
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compute the n-gram similarities on short strings.')
    parser.add_argument(
        '-d', '--dim', type=int, default=200,
        help='The number of dimensions of n-gram vectors')
    parser.add_argument('-g', '--use-gpu', action='store_true')
    parser.add_argument(
        '-k', type=int, default=10,
        help='The number of nearest neighbors to find for each verse.')
    parser.add_argument(
        '-i', '--index-file', metavar='FILE',
        help='Read the verses to index from a separate file.')
    parser.add_argument(
        '-m', '--min-ngrams', type=int, default=10,
        help='Minimum number of known n-grams to consider a verse.')
    parser.add_argument(
        '-n', type=int, default=2,
        help='The size (`n`) of the n-grams (default: 2, i.e. ngrams).')
    parser.add_argument(
        '-q', '--query-size', type=int, default=100,
        help='The number of verses to pass in a single query '
             '(doesn\'t affect the results, only performance)')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.7,
        help='Minimum similarity to output.')
    parser.add_argument(
        '-T', '--text', action='store_true',
        help='Print the strings additionally to IDs.')
    parser.add_argument(
        '-p', '--print-progress', action='store_true',
        help='Print a progress bar.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    res = None
    if args.use_gpu:
        try:
            res = faiss.StandardGpuResources()
        except Exception:
            warnings.warn('GPU not available!')

    query_verses = read_verses(sys.stdin)
    index_verses = []
    if args.index_file is not None:
        with open(args.index_file) as fp:
            index_verses = read_verses(fp)

    sys.stderr.write('Counting n-gram frequencies\n')
    ngram_ids = determine_top_ngrams(index_verses+query_verses, args.n, args.dim)
    sys.stderr.write(' '.join(ngram_ids.keys()) + '\n')

    sys.stderr.write('Creating a dense matrix\n')
    query_v_ids, query_v_texts, query_m = \
        vectorize(query_verses, ngram_ids,
                  n=args.n, dim=args.dim, min_ngrams=args.min_ngrams)
    index_v_ids, index_v_texts, index_m = query_v_ids, query_v_texts, query_m
    if index_verses:
        index_v_ids, index_v_texts, index_m = \
            vectorize(index_verses, ngram_ids,
                      n=args.n, dim=args.dim, min_ngrams=args.min_ngrams)

    sys.stderr.write('Creating a FAISS index\n')
    index = faiss.IndexFlatIP(args.dim)
    if res is not None:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(index_m)

    sys.stderr.write('Searching for nearest neighbors\n')
    progressbar = None
    sims = find_similarities(index, query_m, args.k, args.threshold,
                             args.query_size, args.print_progress)
    for i, j, sim in sims:
        v1_id = query_v_ids[i]
        v2_id = index_v_ids[j]
        if args.text:
            v1_text = query_v_texts[i]
            v2_text = index_v_texts[j]
            print(v1_id, v1_text, v2_id, v2_text, sim, sep='\t')
        else:
            print(v1_id, v2_id, sim, sep='\t')

