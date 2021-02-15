import numpy as np


def align(s1, s2, empty='', insdel_cost=1, dist_fun=None, opt_fun=min):
    'Implements the Wagner-Fischer algorithm.'

    if dist_fun is None:
        dist_fun = lambda i, j: int(s1[i] != s2[j])

    # build the distance matrix
    d = np.zeros(shape=(len(s2)+1, len(s1)+1))
    for j, y in enumerate(s1, 1):
        d[0,j] = j*insdel_cost
    for i, x in enumerate(s2, 1):
        d[i,0] = i*insdel_cost
        for j, y in enumerate(s1, 1):
            d[i,j] = opt_fun(
                d[i-1, j]+insdel_cost,
                d[i,j-1]+insdel_cost,
                d[i-1, j-1] + dist_fun(j-1, i-1))

    # extract the alignment
    alignment = []
    i, j = len(s2), len(s1)
    while i > 0 and j > 0:
        if d[i, j] == d[i-1, j-1] + dist_fun(j-1, i-1):
            alignment.append((s1[j-1], s2[i-1], dist_fun(j-1, i-1)))
            i -= 1
            j -= 1
        elif d[i, j] == d[i-1, j] + insdel_cost:
            alignment.append((empty, s2[i-1], insdel_cost))
            i -= 1
        else:
            alignment.append((s1[j-1], empty, insdel_cost))
            j -= 1
    while i > 0:
        alignment.append((empty, s2[i-1], insdel_cost))
        i -= 1
    while j > 0:
        alignment.append((s1[j-1], empty, insdel_cost))
        j -= 1
    alignment.reverse()
    return alignment


