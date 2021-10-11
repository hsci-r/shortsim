import argparse
from operator import itemgetter
import sys


def delenv(string, maxdels):
    result = set()
    queue = [(string, 0)]
    while queue:
        substr, ndel = queue.pop()
        result.add(substr)
        for i in range(len(substr)):
            substr2 = substr[:i]+substr[i+1:]
            if ndel < maxdels:
                queue.append((substr2, ndel+1))
    return result


def group(lst, key):
    cur_key, cur_grp = None, []
    for item in lst:
        if key(item) == cur_key:
            cur_grp.append(item)
        else:
            if cur_key is not None:
                yield cur_key, cur_grp
            cur_key, cur_grp = key(item), [item]
    if cur_key is not None:
        yield cur_key, cur_grp


def similarity(strings, maxdels, **kwargs):
    s_ss = []
    result = set()
    for (i, s) in strings:
        for ss in delenv(s, maxdels, **kwargs):
            s_ss.append((hash(ss), i))
    s_ss.sort(key=itemgetter(0))
    for ss, grp in group(s_ss, itemgetter(0)):
        for ss2, i in grp:
            for ss3, j in grp:
                if i < j:
                    result.add((i, j))
    return result


def load_words(filename):
    result = None
    with open(filename) as fp:
        result = [tuple(line.rstrip().split('\t')) for line in fp]
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compute similar strings with the FastSS algorithm.')
    parser.add_argument(
        '-k', type=int, default=1,
        help='The maximum number of deletions.')
    parser.add_argument(
        '-T', '--text', action='store_true',
        help='Print the strings additionally to IDs.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    words = [tuple(line.rstrip().split('\t')) for line in sys.stdin]
    word_idx = { w_id: w for (w_id, w) in words }
    sims = similarity(words, args.k)
    for (i, j) in sims:
        if args.text:
            print(i, word_idx[i], j, word_idx[j], sep='\t')
        else:
            print(i, j, sep='\t')

