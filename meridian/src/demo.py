import json
from .config import DATA, BASE, AI_USE_FLOW_DISCOUNT
from .twin import (build_vendor_graph, core_graph, ai_nodes, simulate, most_likely_path, top_evidence, blast_radius, SOURCE)
from .budget import allowance_breakdown


def load(include_overlay=False, overlay_path=None):
    """Vendors (from the workbook) + evidence (stored separately so analysts can edit it)."""
    vendors = json.loads((DATA / 'vendors.json').read_text())
    evidence = json.loads((DATA / 'evidence.json').read_text())
    if include_overlay:
        from .store import load_overlay
        for vid, rows in load_overlay(overlay_path).items():
            evidence.setdefault(vid, []).extend(rows)
    return vendors, evidence


def vendor_assessment(vendor, evidence_rows, base=BASE, trials=10000, flow_discount=AI_USE_FLOW_DISCOUNT,
                      debt=0.0, p_ai_override=None, seed=42):
    """Full twin assessment for one vendor. Pure function: never touches stored state.
    `p_ai_override` is for what-if simulation only."""
    G = build_vendor_graph(vendor, evidence_rows, flow_discount)
    Gc = core_graph(G)
    ai = ai_nodes(Gc)
    p = simulate(Gc, SOURCE, ai, trials, seed) if p_ai_override is None else float(p_ai_override)
    pp, path = most_likely_path(Gc, SOURCE, ai)
    bd = allowance_breakdown(base, p, vendor['criticality'], vendor['criticality_factor'], debt)
    return {'graph': G, 'core_graph': Gc, 'p_ai': p, 'path': path, 'path_prob': pp,
            'evidence': top_evidence(Gc, path), 'allowance': bd['allowance'], 'breakdown': bd,
            'blast': blast_radius(G), 'vendor': vendor}
