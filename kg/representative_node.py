"""Representative node selection -- paper Eq. (18) and the image link of Eq. (21).

    v* = argmax_{v in S} deg(v)        S = entity nodes of the subgraph formed by T_m
    E_img = {(v*, HAS_IMG, u) | v* in V_caption}   u = image URL
"""
from collections import Counter


def representative_node(triples):
    """Return (name, type) of the highest-degree entity in the subgraph formed by the triples, or None."""
    deg, first, types = Counter(), {}, {}
    for t in triples:
        for role in ("subject", "object"):
            key = (t[role].lower(), t[f"{role}_type"])
            deg[key] += 1
            first.setdefault(key, len(first))
            types[key] = t[role]
    if not deg:
        return None
    best = max(deg, key=lambda k: (deg[k], -first[k]))      # ties -> earliest mentioned
    return dict(key=best, name=types[best], type=best[1], degree=deg[best])


def img_edges(rep, image_url):
    """Eq. (21): link the representative node of a caption subgraph to its image u."""
    return [(rep, "HAS_IMG", image_url)] if rep and image_url else []
