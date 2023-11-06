import argparse
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import sys
import tqdm

from shortsim.clustering import chinese_whispers

def load_sims(fp, max_v_id = None, min_sim = 0, loop_weight = 0):
    n = 1
    indptr = np.zeros(1024, dtype=np.uint32)
    indices = np.zeros(1024, dtype=np.uint32)
    data = np.zeros(1024)
    i, j, offset = 0, 0, 0
    print('Reading the input', file=sys.stderr)
    for line in fp:
        row = line.rstrip().split('\t')
        v1_id, v2_id, sim = int(row[0]), int(row[1]), float(row[2])
        if (max_v_id is not None and (v1_id >= max_v_id or v2_id >= max_v_id)) \
                or sim < min_sim:
            continue
        if i > v1_id:
            raise RuntimeError(
                'The input data must be sorted by the first column '
                'in ascending order!')
        while i < v1_id-1:
            if loop_weight > 0:
                indices[j] = i
                data[j] = loop_weight
                j += 1
                if j >= indices.shape[0]:
                    indices.resize(indices.shape[0] * 2, refcheck=False)
                    data.resize(data.shape[0] * 2, refcheck=False)
            indptr[i+1] = j
            # sort the newly added row by indices
            if indptr[i+1] > indptr[i]:
                k = np.argsort(indices[indptr[i]:indptr[i+1]])
                indices[indptr[i]:indptr[i+1]] = indices[indptr[i]+k]
                data[indptr[i]:indptr[i+1]] = data[indptr[i]+k]
            i += 1
            if i+1 >= indptr.shape[0]:
                indptr.resize(indptr.shape[0] * 2, refcheck=False)
        indices[j] = v2_id-1
        data[j] = sim
        j += 1
        if j >= indices.shape[0]:
            indices.resize(indices.shape[0] * 2, refcheck=False)
            data.resize(data.shape[0] * 2, refcheck=False)
    indptr.resize(i+2, refcheck=False)
    indptr[-1] = j
    indices.resize(j, refcheck=False)
    data.resize(j, refcheck=False)
    return csr_matrix((data, indices, indptr))

def load_sims_from_file(filename, **kwargs):
    with open(filename) as fp:
        return load_sims(fp, **kwargs)
 

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Cluster the verses according to similarity using'
                    ' the Chinese Whispers algorithm.')
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-l', '--loop-weight', type=float, default=0,
                        help='Add loop edges with the specified weight.')
    parser.add_argument('-n', '--max_v_id', type=int)
    parser.add_argument('-s', '--min_sim', type=float, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sims = load_sims_from_file(args.input_file, max_v_id=args.max_v_id,
                               min_sim=args.min_sim, loop_weight=args.loop_weight) \
           if args.input_file is not None \
           else load_sims(sys.stdin, max_v_id=args.max_v_id, min_sim=args.min_sim)
    c = chinese_whispers(sims)
    if args.verbose:
        s = np.array(sims.sum(axis=1)).flatten()
        scores = np.zeros(sims.shape[0])
        for i in range(sims.shape[0]):
            j = np.arange(sims.indptr[i], sims.indptr[i+1])
            j = j[c[sims.indices[j]] == c[i]]
            scores[i] = sims.data[j].sum()
        for i in range(c.shape[0]):
            print(i+1, c[i]+1, scores[i], scores[i] / s[i] if s[i] > 0 else 0, sep='\t')
    else:
        for i in range(c.shape[0]):
            print(i+1, c[i]+1, sep='\t')

