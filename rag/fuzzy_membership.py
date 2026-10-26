"""Fuzzy membership functions and fuzzy formulas used by the RAG module.

Eq. (8)  mu(w)            Gaussian term membership
Eq. (10) mu_h(w)          membership used inside the fuzzy entropy
Eq. (9)  H(c_i)           fuzzy information entropy of a chunk
         H_min^fuzzy      dynamic entropy threshold
Eq. (11) sim_fuzzy        cosine similarity refined by fuzzy membership
Fig. 7(a) triangular / trapezoidal functions (for the smoothness comparison only)
"""
import math
from collections import Counter

import numpy as np

import config


# ---------------------------------------------------------------- Eq. (8)
def gaussian_membership(ratio: float, center: float = None, sigma: float = None) -> float:
    c = config.GAUSS_CENTER if center is None else center
    s = config.GAUSS_SIGMA if sigma is None else sigma
    return math.exp(-((ratio - c) ** 2) / (2 * s ** 2))


def triangular_membership(ratio, a=0.4, b=0.8, c=1.0):
    if ratio <= a or ratio >= c:
        return 0.0
    return (ratio - a) / (b - a) if ratio <= b else (c - ratio) / (c - b)


def trapezoidal_membership(ratio, a=0.4, b=0.7, c=0.9, d=1.0):
    if ratio <= a or ratio >= d:
        return 0.0
    if ratio < b:
        return (ratio - a) / (b - a)
    if ratio <= c:
        return 1.0
    return (d - ratio) / (d - c)


# --------------------------------------------------------------- Eq. (10)
def numeric_flags(tokens, window: int = None):
    """I_num(w): 1 if the token sits in a numerical context (a number within +-window tokens)."""
    from rag.term_weighting import is_number
    window = config.NUM_CONTEXT_WINDOW if window is None else window
    isnum = [is_number(t) for t in tokens]
    flags = {}
    for i, t in enumerate(tokens):
        lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
        near = any(isnum[j] for j in range(lo, hi) if j != i) or isnum[i]
        flags[t] = max(flags.get(t, 0), 1 if near else 0)
    return flags


def mu_h(w: str, p_w: float, weighter, inum: int) -> float:
    boost = 1.0 + config.NUM_BOOST * inum
    if w in weighter.terms:
        return weighter.membership(w) * boost
    return math.exp(-((p_w - config.MUH_CENTER) ** 2) / (2 * config.MUH_SIGMA ** 2)) * boost


# ---------------------------------------------------------------- Eq. (9)
def fuzzy_entropy(tokens, weighter) -> float:
    """H(c) = - sum p(w) log2 p(w) * (1 + 0.3 * (sum mu_h(w)) / |c|)"""
    n = len(tokens)
    if n == 0:
        return 0.0
    counts = Counter(tokens)
    flags = numeric_flags(tokens)
    base = 0.0
    mu_sum = 0.0
    for w, c in counts.items():
        p = c / n
        base -= p * math.log2(p)
        mu_sum += c * mu_h(w, p, weighter, flags.get(w, 0))
    return base * (1.0 + config.ENTROPY_MU_COEF * mu_sum / n)


def h_min_fuzzy(h: float, h_min: float = None) -> float:
    """H_min^fuzzy = H_min * (1.0 + 0.2 * exp(-(H(c)-H_min)^2 / (4 * 0.5^2)))"""
    h_min = config.H_MIN if h_min is None else h_min
    return h_min * (1.0 + config.HF_AMP * math.exp(-((h - h_min) ** 2) / (4 * config.HF_SIGMA ** 2)))


# --------------------------------------------------------------- Eq. (11)
def m_fuzzy(query_text: str, chunk_text: str, weighter) -> float:
    """M_fuzzy(c_i, c_j) = 1.0 + (W_term + B_context) * 2
    W_term    : average Eq. (8) fuzzy membership of shared domain terms
    B_context : |common terms| / |terms in c_i|
    c_i = query chunk, c_j = candidate chunk."""
    from rag.term_weighting import tokenize
    ti, tj = tokenize(query_text), tokenize(chunk_text)
    if not ti:
        return 1.0
    si, sj = set(ti), set(tj)
    shared = si & sj
    dom_shared = [w for w in shared if w in weighter.terms]
    if dom_shared:
        w_term = float(np.mean([weighter.membership(w) for w in dom_shared]))
    else:
        w_term = 0.0
    b_context = len(shared) / max(len(si), 1)
    return 1.0 + (w_term + b_context) * config.M_FUZZY_SCALE


def sim_fuzzy(v_i, v_j, query_text: str, chunk_text: str, weighter) -> float:
    """Eq. (11): cosine similarity multiplied directly by the fuzzy score."""
    v_i, v_j = np.asarray(v_i, dtype=float), np.asarray(v_j, dtype=float)
    cos = float(v_i @ v_j / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-12))
    return cos * m_fuzzy(query_text, chunk_text, weighter)
