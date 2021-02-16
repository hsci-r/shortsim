# Short string similarity toolkit

## Installation

```
python3 setup.py install
```

This will install the package and all dependencies. If you want to run
`shortsim-ngrcos` on the GPU, you need to install the CUDA library and FAISS
with GPU support. See
(the instructions here)[https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#install-via-conda]
on how to do it.

## Commands

The package makes the following commands available:

### `shortsim-ngrcos`

This computes the cosine similarity of character n-grams for a given list
of short strings. It uses the
[FAISS library](https://github.com/facebookresearch/faiss/) for computing
nearest neighbors wrt. cosine similarity, so that it works efficiently even
for millions of strings (especially on a GPU).

The script takes (on stdin) the input data in following format
(`id <TAB> string`):

```
1	This is an example.
2	This is a second example.
3	This would be a third example.
```

And returns the three-column list of: `id_1 <TAB> id_2 <TAB> similarity`:

```
1       2       0.8058231
1       3       0.5938157
2       1       0.8058231
2       3       0.6250541
3       2       0.6250541
3       1       0.5938157
```

The results are limited to `k` nearest neighbors for every target string,
i.e. for a single value in the left column there might be up to `k` rows. The
script takes the following CLI parameters:

* `-d`, `--dim`: the dimensionality of the n-gram vectors: `-d` most frequent
n-grams are used (default: 200). Large values may cause high RAM usage.
* `-g`, `--use-gpu`: use the GPU if possible. If not, it will automatically
fall back to CPU.
* `-k`: the number of nearest neighbors to compute for every string. (default:
10, in practice higher values are useful)
* `-m`, `--min-ngrams`: The minimum number of used n-grams (from our vocabulary
of `d` most frequent) to consider a string. Strings containing less than `m`
n-grams are discarded, so that we avoid vectors with almost only zeros being
all similar to each other. (default: 10)
* `-n`: the "n" in "n-gram" (default: 2, i.e. bigrams)
* `-q`, `--query-size`: how much points to pass to FAISS in one query. This
doesn't affect the results, only performance, and it's safe to leave the
default value in place. (default: 100)
* `-qs`, `--query-start`: only compute nearest neighbors of strings with ID
`>= qs` (default: 0).
* `-qe`, `--query-end`: only compute nearest neighbors of strings with ID
`<= qe` (default: `None`, i.e. not applied).
* `-t`, `--threshold`: minimum similarity to output a pair. (default: 0.7)
* `-T`, `--text`: additionaly to IDs, print also the strings on the output. The
output is then a five-column list instead of three.
* `-p`, `--print-progress`: print a progress bar while searching for
similarities.

### `shortsim-cluster`

This implements the Chinese Whispers clustering algorithm. It takes as input
a three-column list of similarities ( `id_1 <TAB> id_2 <TAB> similarity`):

```
1       2       0.8058231
2       1       0.8058231
```

And prints out a list of `id <TAB> cluster_id`. The cluster IDs are ordered
by size, so that 1 is the largest cluster:

```
1       125
2       91
3       485
```

The Chinese Whispers algorithms repeatedly iterates over all points and
changes their cluster assignment. A progress bar and the number of changed
nodes are printed for every iteration (on stderr). The algorithm stops if no
nodes are changed. Thus, the number of iterations is not known in advance,
but the number of changed nodes provides some information about the remaining
runtime.

Unlike in other scripts, the input must be read from a file, rather than
stdin. (This will be fixed.)

CLI parameters:

* `-i`, `--input-file`: the input file,
* `-n`, `--max_v_id`: take only IDs `< n` (by default not applied),
* `-s`, `--min_sim`: take only similarities larger than this threshold
(default: 0.7).

### `shortsim-align`

This script aligns string pairs using the Wagner-Fisher algorithm (i.e. the
standard edit distance algorithm). It takes the input in format
`id_1 <TAB> string_1 <TAB> id_2 <TAB> string_2` on stdin:

```
1       This is the first example.      2       This is the second example.
```

And returns the original columns plus `alignment`,
`edit_distance` and `edit_similarity`, with edit similarity being:
`(alignment_length-edit_distance)/alignment_length`.

```
1       This is the first example.      2       This is the second example.     T:T h:h i:i s:s  :  i:i s:s  :  t:t h:h e:e  :  -:s f:e i:c r:o s:n t:d  :  e:e x:x a:a m:m p:p l:l e:e .:.     6       0.7777777777777778
```

Currently it takes no parameters.
