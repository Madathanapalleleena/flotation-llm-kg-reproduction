"""Fuzzy semantic chunking with entropy optimisation -- paper Sec. III-B-2(b).

Sentence boundaries are the initial separators.  Segments are merged
recursively using the dynamic fuzzy-entropy threshold:

    c_i* = c_i'                          if H(c_i') >= H_min^fuzzy  and |c_i'| <= 50
         = c_i' + sum_j c'_{i+j}         if H(c_i') <  H_min^fuzzy  and H(merged) >= H_min^fuzzy and |merged| <= 100
"""
import re

import config
from rag.fuzzy_membership import fuzzy_entropy, h_min_fuzzy
from rag.term_weighting import tokenize

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+(?=[A-Z(\[])|\n+")
_CLAUSE_SPLIT = re.compile(r"(?<=[，,、:：])\s*")


def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    out = []
    for p in parts:                       # over-long sentences: fall back to clause separators
        if len(p) <= config.HARD_MAX_LEN:
            out.append(p)
            continue
        buf = ""
        for cl in _CLAUSE_SPLIT.split(p):
            if buf and len(buf) + len(cl) + 1 > config.HARD_MAX_LEN:
                out.append(buf.strip())
                buf = cl
            else:
                buf = (buf + " " + cl).strip() if buf else cl
        if buf:
            out.append(buf.strip())
    return out


class EntropyChunker:
    def __init__(self, weighter, h_min: float = None):
        self.w = weighter
        self.h_min = config.H_MIN if h_min is None else h_min

    def H(self, text):
        return fuzzy_entropy(tokenize(text, drop_stop=False), self.w)

    @staticmethod
    def _join(a, b):
        return a + b if re.search(r"[一-鿿]$", a) else a + " " + b

    def chunk(self, text: str):
        segs = split_sentences(text)
        out, i = [], 0
        while i < len(segs):
            cur = segs[i]
            h = self.H(cur)
            hf = h_min_fuzzy(h, self.h_min)
            if h >= hf and len(cur) <= config.SOFT_MAX_LEN:      # case 1: keep
                out.append(cur)
                i += 1
                continue
            if h < hf:                                           # case 2: merge forward
                merged, j, done = cur, i + 1, False
                while j < len(segs) and len(self._join(merged, segs[j])) <= config.HARD_MAX_LEN:
                    merged = self._join(merged, segs[j])
                    j += 1
                    if self.H(merged) >= h_min_fuzzy(self.H(merged), self.h_min):
                        done = True
                        break
                if done or j > i + 1:
                    out.append(merged)
                    i = j
                    continue
            out.append(cur)                                      # neither rule applies: keep as is
            i += 1
        return out


class FixedChunker:
    """Baseline for 'normal RAG' (Step 5): fixed-size, sentence-unaware windows."""

    def __init__(self, size=100, overlap=20):
        self.size, self.overlap = size, overlap

    def chunk(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        step = max(self.size - self.overlap, 1)
        return [text[i:i + self.size] for i in range(0, len(text), step) if text[i:i + self.size].strip()]
