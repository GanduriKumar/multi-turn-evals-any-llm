from __future__ import annotations
import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set

from .commerce_taxonomy import load_commerce_config


@dataclass
class Scenario:
    id: str
    domain: str
    behavior: str
    axes: Dict[str, str]
    risk_tier: str


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def enumerate_all(tax_cfg: Dict[str, Any]) -> List[Scenario]:
    tax = tax_cfg["taxonomy"]
    risk = tax_cfg["risk_tiers"]["risk"]
    axes = tax["axes"]
    axis_names = list(axes.keys())
    scenarios: List[Scenario] = []
    for d in tax["domains"]:
        for b in tax["behaviors"]:
            # Exclude out_of_policy from this Allowed suite
            bins_lists = [
                [bin for bin in axes[a] if not (a == "policy_boundary" and bin == "out_of_policy")] for a in axis_names
            ]
            for combo in itertools.product(*bins_lists):
                axis_vals = {a: v for a, v in zip(axis_names, combo)}
                risk_labels = []
                # domain/behavior tier
                risk_labels.append((risk.get("domains", {}).get(d, "medium")))
                risk_labels.append((risk.get("behaviors", {}).get(b, "medium")))
                # axis tiers
                for a, v in axis_vals.items():
                    label = risk.get("axes", {}).get(a, {}).get(v, "medium")
                    if label != "excluded":
                        risk_labels.append(label)
                # Overall risk = max severity among labels (high > medium > low)
                tier = "low"
                if "high" in risk_labels:
                    tier = "high"
                elif "medium" in risk_labels:
                    tier = "medium"
                slug = f"{d}|{b}|" + "|".join(f"{k}:{v}" for k, v in axis_vals.items())
                sid = f"{d}.{b}." + ",".join(f"{k}={v}" for k, v in axis_vals.items()) + f".{_stable_hash(slug)}"
                scenarios.append(Scenario(id=sid, domain=d, behavior=b, axes=axis_vals, risk_tier=tier))
    return scenarios


def compute_risk_tier(tax_cfg: Dict[str, Any], domain: str, behavior: str, axes: Dict[str, str]) -> str:
    """Compute overall risk tier for a specific domain/behavior/axes combo
    using the same aggregation rule as enumerate_all.

    - Collect labels from domain, behavior, and each axis bin
    - Ignore bins labeled 'excluded'
    - Overall tier = highest severity among labels: high > medium > low
    """
    tax = tax_cfg["taxonomy"]
    risk = tax_cfg["risk_tiers"]["risk"]
    labels: List[str] = []
    # domain/behavior tiers
    labels.append(risk.get("domains", {}).get(domain, "medium"))
    labels.append(risk.get("behaviors", {}).get(behavior, "medium"))
    # axis tiers
    for a, v in (axes or {}).items():
        lab = risk.get("axes", {}).get(a, {}).get(v, "medium")
        if lab != "excluded":
            labels.append(lab)
    if "high" in labels:
        return "high"
    if "medium" in labels:
        return "medium"
    return "low"


def _pair_coverage(selected: List[Scenario], axis_names: List[str]) -> float:
    """Estimate pair coverage across all axis pairs and their bin pairs."""
    # build universe of pairs
    from collections import defaultdict
    pairs_total = 0
    covered = set()
    # universe from taxonomy bins implicit in selected
    bins_by_axis: Dict[str, set] = {a: set() for a in axis_names}
    for s in selected:
        for a in axis_names:
            bins_by_axis[a].add(s.axes[a])
    universe: set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            for bi in bins_by_axis[ai]:
                for bj in bins_by_axis[aj]:
                    universe.add(((ai, bi), (aj, bj)))
    # cover pairs present in selection
    for s in selected:
        for i in range(len(axis_names)):
            for j in range(i + 1, len(axis_names)):
                ai, aj = axis_names[i], axis_names[j]
                pair = ((ai, s.axes[ai]), (aj, s.axes[aj]))
                covered.add(pair)
    pairs_total = len(universe)
    return (len(covered) / pairs_total) if pairs_total else 1.0


def _covered_pairs(selected: List[Scenario], axis_names: List[str]) -> Set[Tuple[Tuple[str, str], Tuple[str, str]]]:
    covered: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    for s in selected:
        for i in range(len(axis_names)):
            for j in range(i + 1, len(axis_names)):
                ai, aj = axis_names[i], axis_names[j]
                covered.add(((ai, s.axes[ai]), (aj, s.axes[aj])))
    return covered


def _candidate_pairs(s: Scenario, axis_names: List[str]) -> Set[Tuple[Tuple[str, str], Tuple[str, str]]]:
    pairs: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            pairs.add(((ai, s.axes[ai]), (aj, s.axes[aj])))
    return pairs


def sample_for_behavior(
    tax_cfg: Dict[str, Any],
    behavior: str,
) -> Dict[str, Any]:
    tax = tax_cfg["taxonomy"]
    strat = tax_cfg["risk_tiers"]["strategy"]
    alloc = strat["allocation"]
    rng = random.Random(strat.get("rng_seed", 42))

    all_scenarios = [s for s in enumerate_all(tax_cfg) if s.behavior == behavior]

    # Partition by risk tier
    by_tier: Dict[str, List[Scenario]] = {"high": [], "medium": [], "low": []}
    for s in all_scenarios:
        if s.risk_tier in by_tier:
            by_tier[s.risk_tier].append(s)

    # Domain floors
    min_per_domain = int(alloc.get("min_per_domain", 3))
    selected: List[Scenario] = []
    by_domain: Dict[str, List[Scenario]] = {d: [] for d in tax["domains"]}

    # Greedy round-robin to satisfy floors using high→medium→low
    for d in tax["domains"]:
        need = min_per_domain
        for tier in ("high", "medium", "low"):
            if need <= 0:
                break
            pool = [s for s in by_tier[tier] if s.domain == d]
            rng.shuffle(pool)
            take = min(need, len(pool))
            selected.extend(pool[:take])
            by_domain[d].extend(pool[:take])
            # remove taken from tier pools
            picked_ids = {p.id for p in pool[:take]}
            by_tier[tier] = [s for s in by_tier[tier] if s.id not in picked_ids]
            need -= take

    # Quotas by tier
    goal_high, goal_med, goal_low = int(alloc["high"]), int(alloc["medium"]), int(alloc["low"])
    have_high = sum(1 for s in selected if s.risk_tier == "high")
    have_med = sum(1 for s in selected if s.risk_tier == "medium")
    have_low = sum(1 for s in selected if s.risk_tier == "low")

    def fill_quota(tier: str, goal: int):
        nonlocal selected
        pool = by_tier[tier]
        rng.shuffle(pool)
        needed = max(0, goal - sum(1 for s in selected if s.risk_tier == tier))
        take = min(needed, len(pool))
        selected.extend(pool[:take])
        picked = {p.id for p in pool[:take]}
        by_tier[tier] = [s for s in pool if s.id not in picked]

    fill_quota("high", goal_high)
    fill_quota("medium", goal_med)
    fill_quota("low", goal_low)

    # If still under per_behavior_total, fill from remaining pools high→medium→low maximizing pair coverage
    per_behavior_total = int(alloc["per_behavior_total"])
    axis_names = list(tax["axes"].keys())
    # incremental coverage tracking for speed
    covered_pairs = _covered_pairs(selected, axis_names)
    while len(selected) < per_behavior_total and any(by_tier.values()):
        # choose candidate from a limited sample that maximizes new pair coverage
        union_pool = [s for tier_list in by_tier.values() for s in tier_list]
        if not union_pool:
            break
        sample_size = min(100, len(union_pool))
        rng.shuffle(union_pool)
        pool_sample = union_pool[:sample_size]
        best = None
        best_gain = -1
        for s in pool_sample:
            cand_pairs = _candidate_pairs(s, axis_names)
            gain = len(cand_pairs - covered_pairs)
            if gain > best_gain:
                best_gain = gain
                best = s
        if best is None:
            break
        selected.append(best)
        # update covered pairs
        covered_pairs |= _candidate_pairs(best, axis_names)
        # remove best from pools
        for t in ("high", "medium", "low"):
            by_tier[t] = [s for s in by_tier[t] if s.id != best.id]

    # Final coverage check
    pair_cov = _pair_coverage(selected, axis_names)

    manifest = {
        "behavior": behavior,
        "per_behavior_total": per_behavior_total,
        "selected_count": len(selected),
        "pair_coverage": pair_cov,
        "scenarios": [
            {
                "id": s.id,
                "domain": s.domain,
                "axes": s.axes,
                "risk_tier": s.risk_tier,
            }
            for s in selected
        ],
    }
    return manifest


def sample_all_behaviors() -> Dict[str, Any]:
    cfg = load_commerce_config()
    tax = cfg["taxonomy"]
    return {b: sample_for_behavior(cfg, b) for b in tax["behaviors"]}
