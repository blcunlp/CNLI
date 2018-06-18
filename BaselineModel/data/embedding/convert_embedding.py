#!/usr/bin/env python

# read a vocab and precompute a .npy embedding matrix.
# if a vocab entry is in the provided glove embeddings then use the glove data. 
# if it's not, generate a random vector but scale it to the median length of the glove embeddings.
# reserve row 0 in the matrix for the PAD embedding (always set to {0}) 
# reserve row 1 in the matrix for the UNK embedding (given a random value)
import argparse
import numpy as np
import sys
from sklearn import random_projection

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, help="reference vocab of non glove data; token \t idx")
parser.add_argument("--glove-data", required=True, help="glove data. ssv, token, e_d1, e_d2, ...")
parser.add_argument("--npy", required=True, help="npy output")
parser.add_argument("--random-projection-dimensionality", default=None, type=float, 
                    help="if set we randomly project the glove data to a smaller dimensionality")
opts = parser.parse_args()

# slurp vocab entries. assume idxs are valid, ie 1 < i < |v|, no dups, no gaps, etc
# (recall reserving 0 for UNK)
# TODO: use vocab.py
vocab = {}  # token => idx
for line in open(opts.vocab, "r"):
    token, idx = line.strip().split("\t")
    if idx == 0:
        assert token == '_PAD', "expecting to reserve 0 for _PAD"
    elif idx == 1:
        assert token == '_UNK', "expecting to reserve 1 for _UNK"
    elif idx ==2:
        assert token == '_GO',  "expecting to reverse 2 for _GO"
    elif idx ==3:
        assert token == '_EOS',  "expecting to reverse 3 for _EOS"
    else:
        vocab[token] = int(idx)
print "vocab has", len(vocab), "entries (not _PAD or _UNK or _GO or _EOS)"

# alloc output after we see first glove embedding (so we know it's dimensionality)
embeddings = None
glove_dimensionality = None

# pass over glove data copying data into embedddings array
# for the cases where the token is in the reference vocab.
tokens_requiring_random = set(vocab.keys())
glove_embedding_norms = []
for line in open(opts.glove_data, "r"):
    cols = line.strip().split(" ")
    token = cols[0]
    if token in vocab:
        glove_embedding = np.array(cols[1:], dtype=np.float32)
        if embeddings is None:
            glove_dimensionality = len(glove_embedding)
            embeddings = np.empty((len(vocab), glove_dimensionality), dtype=np.float32)  # +1 for pad & unk
        assert len(glove_embedding) == glove_dimensionality, "differing dimensionality in glove data?"
        embeddings[vocab[token]] = glove_embedding
        tokens_requiring_random.remove(token)
        glove_embedding_norms.append(np.linalg.norm(glove_embedding))

# given these embeddings we can calculate the median norm of the glove data
median_glove_embedding_norm = np.median(glove_embedding_norms)

print >>sys.stderr, "build .npy file" 
print >>sys.stderr, "after passing over glove there are", len(tokens_requiring_random), \
    "tokens requiring a random alloc"

# return a random embedding with the same norm as the glove data median norm
def random_embedding():
    random_embedding = np.random.randn(1, glove_dimensionality)
    random_embedding /= np.linalg.norm(random_embedding)
    random_embedding *= median_glove_embedding_norm
    return random_embedding

# assign PAD and UNK random embeddings (pre projection)
embeddings[0] = random_embedding()  # PAD
embeddings[1] = random_embedding()  # UNK

# assign random projections for every other fields requiring it
for token in tokens_requiring_random:
    embeddings[vocab[token]] = random_embedding()

# randomly project (if configured to do so)
if opts.random_projection_dimensionality is not None:
    # assign a temp random embedding for PAD before projection (and zero it after)
    p = random_projection.GaussianRandomProjection(n_components=opts.random_projection_dimensionality)
    embeddings = p.fit_transform(embeddings)

# zero out PAD embedding
embeddings[0] = [0] * embeddings.shape[1]

# write embeddings npy to disk
np.save(opts.npy, embeddings)



