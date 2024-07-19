import argparse
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import sys
import tqdm

from shortsim.clustering import chinese_whispers

def load_sims(fp, min_sim = 0, loop_weight = 0):
    n = 1
    indptr = np.zeros(1024, dtype=np.uint32)
    indices = np.zeros(1024, dtype=np.uint32)
    data = np.zeros(1024)
    nodes, node_ids = [], {}
    i, j, offset = 0, 0, 0
    print('Reading the nodes', file=sys.stderr)
    while (line := fp.readline().rstrip()):
        node_ids[line] = len(nodes)
        nodes.append(line)
    print('Reading the edges', file=sys.stderr)
    for line_no, line in enumerate(fp, 1):
        row = line.rstrip().split('\t')
        x, y, sim = row[0], row[1], float(row[2])
        if sim < min_sim:
            continue
        # if any of the nodes is not known yet -- generate an ID for it
        if x not in node_ids or y not in node_ids:
            print('Line {}: ignoring {}: not found in the nodes list'\
                  .format(line_no, x if x not in node_ids else y),
                  file=sys.stderr)
            continue
        if i > node_ids[x]:
            raise RuntimeError(
                'Line {}: The input data must be sorted by the first column '
                'in ascending order!'.format(line_no))
        while i < node_ids[x]:
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
        indices[j] = node_ids[y]
        data[j] = sim
        j += 1
        if j >= indices.shape[0]:
            indices.resize(indices.shape[0] * 2, refcheck=False)
            data.resize(data.shape[0] * 2, refcheck=False)
    indptr.resize(i+2, refcheck=False)
    indptr[-1] = j
    indices.resize(j, refcheck=False)
    data.resize(j, refcheck=False)
    return csr_matrix((data, indices, indptr)), nodes

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
    parser.add_argument('-s', '--min_sim', type=float, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sims, nodes = \
        load_sims_from_file(args.input_file, min_sim=args.min_sim,
                            loop_weight=args.loop_weight) \
        if args.input_file is not None \
        else load_sims(sys.stdin, min_sim=args.min_sim)
    c = chinese_whispers(sims)
    if args.verbose:
        s = np.array(sims.sum(axis=1)).flatten()
        scores = np.zeros(sims.shape[0])
        for i in range(sims.shape[0]):
            j = np.arange(sims.indptr[i], sims.indptr[i+1])
            j = j[c[sims.indices[j]] == c[i]]
            scores[i] = sims.data[j].sum()
        for i in range(c.shape[0]):
            print(nodes[i], c[i]+1, scores[i], scores[i] / s[i] if s[i] > 0 else 0, sep='\t')
    else:
        for i in range(c.shape[0]):
            print(nodes[i], c[i]+1, sep='\t')

