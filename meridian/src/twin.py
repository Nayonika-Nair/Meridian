"""Vendor Digital Twin (NetworkX).

Meridian data -> Vendor -> Subprocessor -> AI / LLM service

Edge probability  = noisy-OR over the evidence rows on that edge
Row probability   = BASE_PROB[tag] * exp(-age_days / tau)          (time decay)
P_ai              = Monte Carlo probability that SOURCE can reach an AI node

Evidence honesty: an edge whose evidence only shows *AI use* (claim_type =
ai_use_only) can be discounted with `flow_discount` (<1) because AI use is not
proof that Meridian data reaches the AI. Rows marked `data_flow_proven` are not
discounted. P_ai stays a model-based estimate.
"""
import math, random
import networkx as nx

SOURCE = 'Meridian data'
FLOW_PROVEN = 'data_flow_proven'      # evidence shows Meridian data reaches the target
AI_USE_ONLY = 'ai_use_only'           # evidence shows AI use, not the data flow (default)

BASE_PROB = {
    'Marketing claim only': 0.10,
    'Corroborated (2+ independent sources)': 0.40,
    'Technical proof (docs/SDK/job posts)': 0.70,
    'Confirmed': 0.95,
}
_AI_NAMES = {'openai', 'anthropic', 'google', 'llm provider'}


def decayed_probability(tag, age_days, tau=365.0):
    return BASE_PROB.get(tag, 0.10) * math.exp(-age_days / tau)


def noisy_or(ps):
    x = 1.0
    for p in ps:
        x *= 1 - p
    return 1 - x


def target_kind(row):
    """service | ai | unknown, plus whether an AI node is an external provider."""
    k = row.get('target_kind')
    if k == 'ai_provider':
        return 'ai', True
    if k in ('ai', 'service', 'unknown'):
        return k, False
    t = row['target'].lower()                       # legacy rows without target_kind
    return ('ai' if ('ai' in t or t in _AI_NAMES) else 'service'), False


def build_vendor_graph(vendor, evidence_rows, flow_discount=1.0, tau=365.0):
    G = nx.DiGraph()
    vn = vendor['vendor_name']
    G.add_node(SOURCE, kind='source')
    G.add_node(vn, kind='vendor')
    G.add_edge(SOURCE, vn, p=1.0, evidence=['Contractual vendor relationship'])
    grouped = {}
    for e in evidence_rows:
        grouped.setdefault((e.get('parent', vn), e['target']), []).append(e)
    for (parent, target), signals in grouped.items():
        kind, external = target_kind(signals[0])
        ps = []
        for e in signals:
            p = decayed_probability(e['tag'], e.get('age_days', 0), tau)
            if kind == 'ai' and e.get('claim_type', AI_USE_ONLY) != FLOW_PROVEN:
                p *= flow_discount
            ps.append(p)
        if parent not in G:
            G.add_node(parent, kind='service')
        if target not in G:
            G.add_node(target, kind=kind, external=external)
        G.add_edge(parent, target, p=noisy_or(ps), evidence=signals)
    return G


def core_graph(G):
    """Graph used for P_ai: unresolved ('unknown') destinations are excluded."""
    return G.subgraph([n for n, d in G.nodes(data=True) if d.get('kind') != 'unknown']).copy()


def ai_nodes(G):
    return [n for n, d in G.nodes(data=True) if d.get('kind') == 'ai']


def simulate(G, source, ai_nodes, n=10000, seed=42):
    """Monte Carlo P_ai: sample each edge as open with probability p, test reachability."""
    rng = random.Random(seed)
    edges = [(u, v, d['p']) for u, v, d in G.edges(data=True)]
    targets = {t for t in ai_nodes if t in G}
    if source not in G or not targets:
        return 0.0
    hits = 0
    for _ in range(n):
        adj = {}
        for u, v, p in edges:
            if rng.random() < p:
                adj.setdefault(u, []).append(v)
        seen, stack, found = {source}, [source], False
        while stack and not found:
            for w in adj.get(stack.pop(), ()):
                if w in targets:
                    found = True
                    break
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        hits += found
    return hits / n


def most_likely_path(G, source, ai_nodes):
    best = (0.0, [])
    for t in ai_nodes:
        try:
            path = nx.shortest_path(G, source, t)
            prob = 1.0
            for u, v in zip(path, path[1:]):
                prob *= G[u][v]['p']
            if prob > best[0]:
                best = (prob, path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return best


def top_evidence(G, path, limit=5):
    rows = []
    for u, v in zip(path, path[1:]):
        for e in G[u][v].get('evidence', []):
            rows.append((G[u][v]['p'], e))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in rows[:limit]]


def blast_radius(G, source=SOURCE):
    """How many organisations could potentially receive Meridian data?"""
    if source not in G:
        return _empty_blast()
    depth = nx.single_source_shortest_path_length(G, source)
    reach = {source: 1.0}                                   # best-path probability of reaching each node
    for u in nx.topological_sort(G) if nx.is_directed_acyclic_graph(G) else depth:
        if u not in reach:
            continue
        for v in G.successors(u):
            reach[v] = max(reach.get(v, 0.0), reach[u] * G[u][v]['p'])
    kinds = {n: G.nodes[n].get('kind') for n in depth if n != source}
    direct = [n for n in G.successors(source) if kinds.get(n) == 'vendor']
    sub = [n for n, k in kinds.items() if k == 'service']
    ai = [n for n, k in kinds.items() if k == 'ai']
    ext = [n for n in ai if G.nodes[n].get('external')]
    unk = [n for n, k in kinds.items() if k == 'unknown']
    nodes = [{'name': n, 'kind': k, 'depth': depth[n], 'best_path_p': reach.get(n, 0.0),
              'external': bool(G.nodes[n].get('external'))} for n, k in kinds.items()]
    return {'direct_vendors': len(direct), 'subprocessors': len(sub), 'ai_systems': len(ai),
            'external_ai_providers': len(ext), 'unknown_destinations': len(unk),
            'organizations': len(direct) + len(sub) + len(ext) + len(unk),
            'max_depth': max([depth[n] for n in kinds] or [0]), 'nodes': nodes}


def _empty_blast():
    return {'direct_vendors': 0, 'subprocessors': 0, 'ai_systems': 0, 'external_ai_providers': 0,
            'unknown_destinations': 0, 'organizations': 0, 'max_depth': 0, 'nodes': []}
