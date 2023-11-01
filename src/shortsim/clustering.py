import numpy as np
from operator import itemgetter
import random
import sys
import tqdm


def chinese_whispers(sims):

    def _remap_clusters(c):
        '''
        Remap cluster IDs so that they are numbered consecutively and
        according to decreasing frequency.
        '''
        clust_freq = np.zeros(c.shape, dtype=np.int32)
        for i in range(c.shape[0]):
            clust_freq[c[i]] -= 1       # negative numbers for decreasing sort
        order = np.argsort(clust_freq)
        idmap = np.zeros(c.shape, dtype=np.int32)
        for i in range(order.shape[0]):
            idmap[order[i]] = i
        result = np.array([idmap[c[i]] for i in range(c.shape[0])])
        return result


    n = sims.shape[0]
    c = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        c[i] = i
    changed = 1
    idx = list(range(n))
    while changed > 0:
        changed = 0
        random.shuffle(idx)
        for i in tqdm.tqdm(idx):
            j, k = sims.indptr[i], sims.indptr[i+1]
            if j == k:
                continue
            cl_ids, cl_inv = np.unique(c[sims.indices[j:k]], return_inverse=True)
            scores = np.bincount(cl_inv, weights=sims.data[j:k])
            cl_max = cl_ids[np.argmax(scores)]
            if c[i] != cl_max:
                changed += 1
                c[i] = cl_max
        print('changed =', changed, file=sys.stderr)
    return _remap_clusters(c)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Cluster the verses according to similarity using'
                    ' the Chinese Whispers algorithm.')
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-n', '--max_v_id', type=int)
    parser.add_argument('-s', '--min_sim', type=float, default=0.7)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    sims = load_sims_from_file(args.input_file, args.max_v_id, args.min_sim) \
           if args.input_file is not None \
           else get_sims_from_db(args.max_v_id, args.min_sim)
    c = chinese_whispers(sims)
    for i in range(c.shape[0]):
        print(i+1, c[i]+1, sep='\t')

