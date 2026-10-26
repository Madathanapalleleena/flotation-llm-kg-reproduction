"""Structural assessment of the KG -- paper Sec. IV-B-1, Eq. (22)-(28).

I_s  isolated node ratio              = N_isolated / N_total                                   (22)
C_s  process coverage rate            = N_covered / N_core                                     (23)
B_s  bidirectional relation balance   = 1 - | (N_down - N_up) / (N_down + N_up + 1) |          (24)
Z_s  process node knowledge coverage  = sigmoid( (mean_stage_degree - mu) / delta )            (25)
P_s  longest process path length      = min(1, L_max / L_optimal)                              (26)
R_s  process loop rationality         = sigmoid( T_mid / (T_min + |D - T_target|) ) * exp(-max(0, |D - T_max|))   (27)
     D = N_process / N_edges ,  T_min = 20/N_edges ,  T_max = 60/N_edges
S_1 = .25(1-I_s) + .25 C_s + .2 B_s + .15 Z_s + .1 P_s + .05 R_s                               (28)

[assumed]  The paper does not define T_mid / T_target; they are set to the middle of [T_min, T_max].
           N_process = number of directed simple cycles (up to 100 000) in the ontology process subgraph.
"""
import itertools
import math

import networkx as nx

import config
from ontology.knowledge_tree import KnowledgeTree


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def build_nx(graph):
    G = nx.MultiDiGraph()
    for n in graph["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph["edges"]:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], type=e["type"], kind=e.get("kind"))
    return G


def structural_assessment(graph, kt: KnowledgeTree = None):
    kt = kt or KnowledgeTree()
    G = build_nx(graph)
    n_total = G.number_of_nodes()
    n_edges = G.number_of_edges()
    deg = dict(G.degree())

    # (22) isolated node ratio
    n_iso = sum(1 for d in deg.values() if d == 0)
    I_s = n_iso / max(n_total, 1)

    # (23) process coverage: a core stage is covered if extracted knowledge is attached to it
    covered = 0
    for sid in kt.core_stages:
        node = f"onto:{sid}"
        aliases = [a.lower() for a in kt.nodes[sid].aliases + [kt.nodes[sid].label]]
        linked = any(d.get("type") == "HAS_ENTITIES" for _, _, d in G.out_edges(node, data=True)) if node in G else False
        mentioned = any(d.get("type") == "ProcessStage" and d.get("name", "").lower() in aliases
                        for _, d in G.nodes(data=True))
        covered += 1 if (linked or mentioned) else 0
    C_s = covered / max(len(kt.core_stages), 1)

    # (24) bidirectional balance over the *_DOWN / *_UP relations
    n_down = sum(1 for *_, d in G.edges(data=True) if str(d["type"]).endswith("_DOWN"))
    n_up = sum(1 for *_, d in G.edges(data=True) if str(d["type"]).endswith("_UP"))
    B_s = 1.0 - abs((n_down - n_up) / (n_down + n_up + 1))

    # (25) knowledge richness of process nodes
    vals = list(deg.values()) or [0]
    mu = sum(vals) / len(vals)
    delta = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)) or 1.0
    stage_nodes = [n for n, d in G.nodes(data=True)
                   if (d.get("kind") == "Ontology" and d.get("level") == "stage") or d.get("type") == "ProcessStage"]
    mean_stage = sum(deg[n] for n in stage_nodes) / max(len(stage_nodes), 1)
    Z_s = _sigmoid((mean_stage - mu) / delta)

    # (26) longest process path on the forward flow edges (NEXT / OUTPUT, downstream direction)
    flow = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d["type"] in ("HAS_NEXT_DOWN", "HAS_OUTPUT_DOWN"):
            flow.add_edge(u, v)
    L_max = nx.dag_longest_path_length(flow) if flow.number_of_edges() and nx.is_directed_acyclic_graph(flow) else 0
    P_s = min(1.0, L_max / max(kt.L_optimal, 1))

    # (27) process loop rationality
    proc = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("kind") == "ontology":
            proc.add_edge(u, v)
    n_process = sum(1 for _ in itertools.islice(nx.simple_cycles(proc), 100000))
    D = n_process / max(n_edges, 1)
    t_min, t_max = config.R_TMIN_NUM / max(n_edges, 1), config.R_TMAX_NUM / max(n_edges, 1)
    t_mid = t_target = (t_min + t_max) / 2.0
    R_s = _sigmoid(t_mid / (t_min + abs(D - t_target))) * math.exp(-max(0.0, abs(D - t_max)))

    w = config.S1_WEIGHTS
    S1 = (w["I"] * (1 - I_s) + w["C"] * C_s + w["B"] * B_s + w["Z"] * Z_s + w["P"] * P_s + w["R"] * R_s)
    return dict(n_nodes=n_total, n_edges=n_edges, I_s=I_s, C_s=C_s, B_s=B_s, Z_s=Z_s, P_s=P_s, R_s=R_s, S1=S1,
                detail=dict(n_isolated=n_iso, covered_stages=covered, n_down=n_down, n_up=n_up, mean_stage_degree=mean_stage,
                            mu=mu, delta=delta, L_max=L_max, L_optimal=kt.L_optimal, n_process_loops=n_process, D=D))
