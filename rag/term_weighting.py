"""Text loading and automatic term weighting -- paper Sec. III-B-2(a), Eq. (8).

Term importance = TF-IDF, normalised by the maximum score (w_score / w_max) and
mapped to a fuzzy membership value with a Gaussian function:

    mu(w) = exp( -(w_score/w_max - 0.8)^2 / (2 * 0.2^2) )
"""
import math
import re
from collections import Counter

import config
from rag.fuzzy_membership import gaussian_membership

# one token = one CJK character | an English word | a number
TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z]+(?:[-'’][A-Za-z]+)*|\d+(?:\.\d+)?")
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

STOP_EN = set("""he she him her his hers they them their we us our you your i me my a an the of in on at to for from by with without and or but if then than that this these those is are
was were be been being it its as into onto over under between among such can could may might will would should shall
do does did has have had not no nor so very also more most less least when while which who whom whose what where how
each per via within""".split())


def tokenize(text: str, drop_stop: bool = True):
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    toks = [t for t in toks if len(t) > 1 or not t.isascii() or t.isdigit()]   # drop stray letters (s, o, m ...)
    if drop_stop:
        toks = [t for t in toks if t not in STOP_EN]
    return toks


def is_number(tok: str) -> bool:
    return bool(NUM_RE.match(tok))


class TermWeighter:
    """Fits corpus statistics, then answers mu(w) and membership of the `terms` set."""

    def __init__(self, docs):
        self.docs = [d for d in docs if d and d.strip()]
        self.N = max(len(self.docs), 1)
        self.fit()

    def fit(self):
        toks_per_doc = [tokenize(d) for d in self.docs]
        df = Counter()
        for toks in toks_per_doc:
            df.update(set(toks))
        # dynamic stop words: terms present in too many documents carry no domain signal
        self.dynamic_stop = {w for w, c in df.items()
                             if c / self.N >= config.DYN_STOP_DF_RATIO and self.N >= 10 and not is_number(w)}
        score = Counter()
        for toks in toks_per_doc:
            if not toks:
                continue
            tf = Counter(toks)
            for w, c in tf.items():
                if w in self.dynamic_stop or is_number(w):
                    continue
                idf = math.log((1 + self.N) / (1 + df[w])) + 1.0
                score[w] += (c / len(toks)) * idf          # TF-IDF accumulated over documents
        self.score = dict(score)
        self.w_max = max(self.score.values()) if self.score else 1.0
        self.mu = {w: gaussian_membership(s / self.w_max) for w, s in self.score.items()}   # Eq. (8)
        self.terms = {w for w, m in self.mu.items() if m >= config.TERM_MU_THRESHOLD}        # high-TF-IDF domain terms

    def membership(self, w: str) -> float:
        return self.mu.get(w, 0.0)

    def top_terms(self, k=15):
        return sorted(self.mu.items(), key=lambda kv: -kv[1])[:k]
