import argparse
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import sys
import tqdm

from shortsim.clustering import chinese_whispers

def load_sims(fp, max_v_id = None, min_sim = 0, loop_weight = 0):
    n = 1
    data = []
    print('Reading the input', file=sys.stderr)
    for line in fp:
        row = line.rstrip().split('\t')
        v1_id, v2_id, sim = int(row[0]), int(row[1]), float(row[2])
        if (max_v_id is None or (v1_id < max_v_id and v2_id < max_v_id)) \
                and sim > min_sim:
            data.append((v1_id, v2_id, sim))
        if v1_id > n:
            n = v1_id
        if v2_id > n:
            n = v2_id
    print('Buiding the matrix...', file=sys.stderr)
    sims = lil_matrix((n, n), dtype=np.float32)
    if loop_weight > 0:
        for i in range(sims.shape[0]):
            sims[i,i] = loop_weight
    for v1_id, v2_id, sim in tqdm.tqdm(data):
        sims[v1_id-1, v2_id-1] = (sim-min_sim)/(1-min_sim)
    return csr_matrix(sims)

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
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sims = load_sims_from_file(args.input_file, max_v_id=args.max_v_id,
                               min_sim=args.min_sim, loop_weight=args.loop_weight) \
           if args.input_file is not None \
           else load_sims(sys.stdin, max_v_id=args.max_v_id, min_sim=args.min_sim)
    c = chinese_whispers(sims)
    for i in range(c.shape[0]):
        print(i+1, c[i]+1, sep='\t')

