from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, List, Tuple, Any, Optional

from .coverage_config import CoverageConfig


AXES_ORDER = [
    "price_sensitivity",
    "brand_bias",
    "availability",
    "policy_boundary",
]


@dataclass(frozen=True)
class Scenario:
    domain: str
    behavior: str
    axes: Tuple[Tuple[str, str], ...]  # ((axis_name, bin), ... in AXES_ORDER)

    @property
    def id(self) -> str:
        # Stable ID composed from axis values in fixed order
        parts = [f"{k}={v}" for k, v in self.axes]
        return f"{self.domain}|{self.behavior}|" + "|".join(parts)


def _axis_index_map(taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    idx: Dict[str, Dict[str, int]] = {}
    axes = taxonomy.get("axes", {})
    for a in AXES_ORDER:
        vals = axes.get(a, [])
        idx[a] = {v: i for i, v in enumerate(vals)}
    return idx


def enumerate_scenarios(taxonomy: Dict[str, Any], domain: str, behavior: str) -> List[Scenario]:
    axes = taxonomy["axes"]
    items: List[Scenario] = []
    for ps in axes["price_sensitivity"]:
        for bb in axes["brand_bias"]:
            for av in axes["availability"]:
                for pb in axes["policy_boundary"]:
                    items.append(
                        Scenario(
                            domain=domain,
                            behavior=behavior,
                            axes=tuple(
                                (k, v)
                                for k, v in zip(
                                    AXES_ORDER, [ps, bb, av, pb]
                                )
                            ),
                        )
                    )
    return items


def _matches_filter(sc: Scenario, cond: Dict[str, List[str]]) -> bool:
    # Returns True when sc satisfies all keys present in cond
    for k, allowed in cond.items():
        if not allowed:
            continue
        val = dict(sc.axes).get(k)
        if val not in set(allowed):
            return False
    return True


def _exclude_scenarios(scenarios: List[Scenario], rule: Dict[str, Any]) -> List[Scenario]:
    when = rule.get("when", {}) or {}
    exclude_axes = ((rule.get("exclude") or {}).get("axes")) or {}
    if not exclude_axes:
        return scenarios
    kept: List[Scenario] = []
    for sc in scenarios:
        if when and not _matches_filter(sc, when):
            kept.append(sc)
            continue
        # if any axis in exclude matches, drop
        sc_axes = dict(sc.axes)
        drop = False
        for ax, values in exclude_axes.items():
            if values and sc_axes.get(ax) in set(values):
                drop = True
                break
        if not drop:
            kept.append(sc)
    return kept


def _cap_scenarios(
    scenarios: List[Scenario], rule: Dict[str, Any], taxonomy: Dict[str, Any], seed: int
) -> List[Scenario]:
    cap = rule.get("cap")
    if not isinstance(cap, int) or cap < 1:
        return scenarios
    when = rule.get("when", {}) or {}
    # Partition into matching and non-matching
    matching: List[Scenario] = []
    non_matching: List[Scenario] = []
    for sc in scenarios:
        (matching if _matches_filter(sc, when) else non_matching).append(sc)

    if len(matching) <= cap:
        return scenarios  # nothing to do

    # Deterministic selection of cap scenarios from matching
    idx_map = _axis_index_map(taxonomy)

    def stable_key(sc: Scenario) -> Tuple:
        # 1) axis index order; 2) hash with seed for tie-break
        sc_axes = dict(sc.axes)
        order_tuple = tuple(idx_map[a][sc_axes[a]] for a in AXES_ORDER)
        h = blake2b(
            f"{seed}|{sc.id}".encode("utf-8"), digest_size=8
        ).hexdigest()
        # convert to int for ordering
        return order_tuple + (int(h, 16),)

    selected = sorted(matching, key=stable_key)[:cap]
    # Preserve overall order deterministically by re-sorting all by stable_key, but keeping only selected for matching subset
    selected_ids = {sc.id for sc in selected}
    result: List[Scenario] = []
    # Keep selected from matching (sorted) + all non-matching
    result.extend(sorted(selected, key=lambda sc: sc.id))
    result.extend(non_matching)
    return result


def _applies_to(domain: str, behavior: str, applies: Optional[Dict[str, Any]]) -> bool:
    if not applies:
        return True
    doms = applies.get("domains")
    behs = applies.get("behaviors")
    if doms and domain not in doms:
        return False
    if behs and behavior not in behs:
        return False
    return True


def apply_exclusions(
    taxonomy: Dict[str, Any],
    exclusions: Dict[str, Any],
    domain: str,
    behavior: str,
    seed: int = 42,
) -> List[Scenario]:
    scenarios = enumerate_scenarios(taxonomy, domain, behavior)
    rules: List[Dict[str, Any]] = exclusions.get("rules", [])
    # Apply rules in given order. Process excludes first (rules with 'exclude'), then caps for the same rule if present.
    for rule in rules:
        if not _applies_to(domain, behavior, rule.get("applies")):
            continue
        if rule.get("exclude"):
            scenarios = _exclude_scenarios(scenarios, rule)
        if rule.get("cap"):
            scenarios = _cap_scenarios(scenarios, rule, taxonomy, seed)
    return scenarios


class CoverageEngine:
    def __init__(self, config: Optional[CoverageConfig] = None) -> None:
        self.config = config or CoverageConfig()
        self.taxonomy = self.config.load_taxonomy()
        self.exclusions = self.config.load_exclusions()
        self.coverage = self.config.load_coverage()

    def scenarios_for(self, domain: str, behavior: str, seed: int = 42) -> List[Scenario]:
        # Generate base scenarios then optimize according to coverage settings
        mode = (self.coverage or {}).get("mode", "exhaustive")
        per_behavior_budget = (self.coverage or {}).get("per_behavior_budget")
        anchors = (self.coverage or {}).get("anchors") or []

        scenarios = apply_exclusions(self.taxonomy, self.exclusions, domain, behavior, seed)

        # Short-circuit for exhaustive
        if mode == "exhaustive":
            return scenarios

        # Pairwise selection (t=2). We implement a simple greedy covering array builder.
        # Keep anchors first (if they match this domain/behavior), then fill with pairwise until budget.

        # Build value domains per axis
        axes_vals: Dict[str, List[str]] = {a: list(self.taxonomy.get("axes", {}).get(a, [])) for a in AXES_ORDER}
        # Compute all required pairs we must cover: (axisA, valA, axisB, valB) for A<B order
        required_pairs: set[Tuple[str, str, str, str]] = set()
        for i in range(len(AXES_ORDER)):
            for j in range(i + 1, len(AXES_ORDER)):
                a, b = AXES_ORDER[i], AXES_ORDER[j]
                for va in axes_vals.get(a, []):
                    for vb in axes_vals.get(b, []):
                        required_pairs.add((a, va, b, vb))

        # Helper to compute which pairs a scenario covers
        def scenario_pairs(sc: Scenario) -> List[Tuple[str, str, str, str]]:
            pairs: List[Tuple[str, str, str, str]] = []
            axes_dict = dict(sc.axes)
            for i in range(len(AXES_ORDER)):
                for j in range(i + 1, len(AXES_ORDER)):
                    a, b = AXES_ORDER[i], AXES_ORDER[j]
                    pairs.append((a, axes_dict[a], b, axes_dict[b]))
            return pairs

        # Seed with anchors
        selected: List[Scenario] = []
        def applies_to(sc: Scenario, rule: Dict[str, Any]) -> bool:
            applies = rule.get("applies") or {}
            # domain/behavior filtering
            doms = applies.get("domains")
            behs = applies.get("behaviors")
            if doms and sc.domain not in doms:
                return False
            if behs and sc.behavior not in behs:
                return False
            when = rule.get("when") or {}
            return _matches_filter(sc, when)

        if anchors:
            for sc in scenarios:
                if any(applies_to(sc, a) for a in anchors):
                    selected.append(sc)

        # Track which pairs are already covered
        covered: set[Tuple[str, str, str, str]] = set()
        for sc in selected:
            covered.update(scenario_pairs(sc))

        # Greedy selection: repeatedly add scenario that covers the most uncovered pairs
        remaining = [sc for sc in scenarios if sc not in selected]

        def score(sc: Scenario) -> int:
            c = 0
            for p in scenario_pairs(sc):
                if p not in covered:
                    c += 1
            return c

        # Budget handling
        max_count = per_behavior_budget if isinstance(per_behavior_budget, int) and per_behavior_budget > 0 else None

        # If no budget, set a loose upper bound to avoid runaway (cap at len(scenarios))
        upper_bound = max_count if max_count is not None else len(scenarios)

        while remaining and len(selected) < upper_bound and len(covered) < len(required_pairs):
            best = max(remaining, key=score, default=None)
            if not best or score(best) == 0:
                break
            selected.append(best)
            covered.update(scenario_pairs(best))
            remaining.remove(best)

        # If we still have budget and uncovered pairs (rare), fill arbitrarily while respecting budget
        if max_count is not None and len(selected) < max_count and remaining:
            fill = max_count - len(selected)
            selected.extend(remaining[:fill])

        # Always return in a stable order by scenario id
        return sorted(selected or scenarios, key=lambda s: s.id)
