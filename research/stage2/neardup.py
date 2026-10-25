"""Near-duplicate detection with word 5-gram shingles and Jaccard similarity.

Used by audit.py (to report overlap) and train.py (to drop test dialogues
that are near-copies of training dialogues, and to keep near-copies in the
same cross-validation fold).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from common import full_text, normalize


def shingle_matrix(dialogues, n=5):
    """Binary docs x shingles matrix (hashed word n-grams of both speakers)."""
    texts = [normalize(full_text(d)) for d in dialogues]
    hv = HashingVectorizer(analyzer="word", ngram_range=(n, n), n_features=2 ** 24,
                           alternate_sign=False, norm=None, binary=True,
                           token_pattern=r"\S+", lowercase=False)
    X = hv.transform(texts).tocsr()
    X.data[:] = 1.0
    return X


def near_pairs(X, threshold=0.3, block=400):
    """All pairs (i, j, jaccard) with i < j and jaccard >= threshold."""
    X = X.astype(np.float32).tocsr()
    sizes = np.asarray(X.sum(axis=1)).ravel()
    XT = X.T.tocsc()
    out_i, out_j, out_s = [], [], []
    for start in range(0, X.shape[0], block):
        stop = min(start + block, X.shape[0])
        inter = (X[start:stop] @ XT).tocoo()
        i = inter.row + start
        j = inter.col
        keep = i < j
        i, j, v = i[keep], j[keep], inter.data[keep]
        union = sizes[i] + sizes[j] - v
        jac = np.where(union > 0, v / np.maximum(union, 1), 0.0)
        k = jac >= threshold
        out_i.append(i[k]); out_j.append(j[k]); out_s.append(jac[k])
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)


def components(n, i, j):
    """Connected components (cluster id per doc) of the near-duplicate graph."""
    g = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = sp.csgraph.connected_components(g, directed=False)
    return labels
