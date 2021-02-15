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
        '-qs', '--query-start', type=int, default=0,
        help='The string ID from whith to start the query.')
    parser.add_argument(
        '-qe', '--query-end', type=int, default=None,
        help='The string ID at which to end the query.')
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

    verses = []
    for line in sys.stdin:
        spl = line.rstrip().split('\t')
        if len(spl) < 2: continue
        v_id, text = spl[0], spl[1]
        verses.append((v_id, text))

    sys.stderr.write('Counting n-gram frequencies\n')
    ngram_freq = defaultdict(lambda: 0)
    for text in map(itemgetter(1), verses):
        for ngr in ngrams(text, args.n):
            ngram_freq[ngr] += 1

    ngram_ids = {
        ngr : i \
        for i, (ngr, freq) in enumerate(sorted(
            ngram_freq.items(), key=itemgetter(1), reverse=True)[:args.dim]) }
    sys.stderr.write(' '.join(ngram_ids.keys()) + '\n')

    # FIXME memory is being wasted here by storing v_ids and verses again
    sys.stderr.write('Creating a dense matrix\n')
    v_ids, v_texts, rows = [], [], []
    for (v_id, text) in tqdm.tqdm(verses):
        v_ngr_ids = [ngram_ids[ngr] for ngr in ngrams(text, args.n) \
                     if ngr in ngram_ids]
        if len(v_ngr_ids) >= args.min_ngrams:
            row = np.zeros(args.dim, dtype=np.float32)
            for ngr_id in v_ngr_ids:
                row[ngr_id] += 1
            rows.append(row)
            v_ids.append(v_id)
            v_texts.append(text)
    m = np.vstack(rows)

    sys.stderr.write('Normalizing\n')
    m = np.divide(m, norm(m, axis=1).reshape((m.shape[0], 1)))

    sys.stderr.write('Creating a FAISS index\n')
    index = faiss.IndexFlatIP(args.dim)
    if res is not None:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(m)

    qs = 0
    if args.query_start > 0:
        while v_ids[qs] < args.query_start:
            qs += 1
    qe = m.shape[0]
    if args.query_end is not None:
        while v_ids[qe-1] > args.query_end:
            qe -= 1

    sys.stderr.write('Searching for nearest neighbors\n')
    progressbar = None
    if args.print_progress:
        progressbar = tqdm.tqdm(total=qe-qs)
    for i in range(qs, qe, args.query_size):
        query = range(i, min(qe, i+args.query_size))
        D, I = index.search(m[query,], args.k)
        for i, q in enumerate(query):
            for j in range(args.k):
                if D[i,j] >= args.threshold:
                    v1_id = v_ids[q]
                    v2_id = v_ids[I[i,j]]
                    if v1_id != v2_id:
                        if args.text:
                            v1_text = v_texts[q]
                            v2_text = v_texts[I[i,j]]
                            print(v1_id, v1_text, v2_id, v2_text, D[i,j], sep='\t')
                        else:
                            print(v1_id, v2_id, D[i,j], sep='\t')
        if args.print_progress:
            progressbar.update(D.shape[0])

