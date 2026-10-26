"""Triple-extraction metrics -- paper Eq. (7).

    P_re  = N_valid / N_pred          R_ec = N_valid / N_exp
    F1    = 2 P R / (P + R)           C_sis = 1 - N_conflict / N_total
    S_triplet(t1,t2) = 0.3 S_subject + 0.3 S_object + 0.25 S_relation + 0.15 S_label
A predicted triple is a valid match when S_triplet > 0.85.
"""
import re
from difflib import SequenceMatcher

import config
from kg.triple_validator import count_conflicts


def _n(s):
    s = re.sub(r"\b(the|a|an)\b", " ", str(s).lower())
    return re.sub(r"[^a-z0-9一-鿿]+", " ", s).strip()


def str_sim(a, b):
    a, b = _n(a), _n(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def triplet_similarity(t1: dict, t2: dict) -> float:
    s_sub = str_sim(t1["subject"], t2["subject"])
    s_obj = str_sim(t1["object"], t2["object"])
    s_rel = 1.0 if _n(t1["relation"]) == _n(t2["relation"]) else str_sim(t1["relation"], t2["relation"])
    s_lab = 0.5 * (t1.get("subject_type", "").lower() == t2.get("subject_type", "").lower()) + \
        0.5 * (t1.get("object_type", "").lower() == t2.get("object_type", "").lower())
    return (config.W_SUBJECT * s_sub + config.W_OBJECT * s_obj + config.W_RELATION * s_rel + config.W_LABEL * s_lab)


def match_triples(pred, gold, thr: float = None):
    """One-to-one greedy matching on S_triplet > thr.  Returns N_valid."""
    thr = config.MATCH_THRESHOLD if thr is None else thr
    pairs = sorted(((triplet_similarity(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, valid = set(), set(), []
    for s, i, j in pairs:
        if s <= thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        valid.append((i, j, s))
    return valid


def prf(n_valid, n_pred, n_exp):
    p = n_valid / n_pred if n_pred else (1.0 if n_exp == 0 else 0.0)
    r = n_valid / n_exp if n_exp else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(preds: dict, golds: dict):
    """preds/golds: {sample_id: [triple,...]}.  Metrics averaged over samples (as in Table III) + micro totals."""
    ps, rs, fs = [], [], []
    tv = tp = te = 0
    all_pred = []
    per_sample = {}
    missed, spurious = {}, {}
    for sid, gold in golds.items():
        pred = preds.get(sid, [])
        valid = match_triples(pred, gold)
        p, r, f = prf(len(valid), len(pred), len(gold))
        ps.append(p); rs.append(r); fs.append(f)
        tv += len(valid); tp += len(pred); te += len(gold)
        all_pred += pred
        per_sample[sid] = dict(P=p, R=r, F1=f, n_pred=len(pred), n_exp=len(gold), n_valid=len(valid))
        mi = {j for _, j, _ in valid}
        pi = {i for i, _, _ in valid}
        missed[sid] = [g for j, g in enumerate(gold) if j not in mi]
        spurious[sid] = [t for i, t in enumerate(pred) if i not in pi]
    n = max(len(golds), 1)
    n_conf = sum(count_conflicts(preds.get(sid, [])) for sid in golds)
    c_sis = 1.0 - n_conf / max(len(all_pred), 1)
    mp, mr, mf = prf(tv, tp, te)
    return dict(precision=sum(ps) / n, recall=sum(rs) / n, f1=sum(fs) / n, c_sis=c_sis,
                micro_precision=mp, micro_recall=mr, micro_f1=mf,
                n_valid=tv, n_pred=tp, n_exp=te, n_conflict=n_conf, n_samples=len(golds),
                per_sample=per_sample, missed=missed, spurious=spurious)


if __name__ == "__main__":
    a = dict(subject="pulp concentration", subject_type="Parameter", relation="increases", object="recovery rate", object_type="Product")
    print(triplet_similarity(a, dict(a, subject="the pulp concentration")))
