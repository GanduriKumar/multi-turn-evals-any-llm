from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Iterable
from copy import deepcopy

from .risk_sampler import sample_for_behavior
from .convgen_v2 import build_records
from .commerce_taxonomy import load_commerce_config
from .coverage_config import CoverageConfig


def _apply_sampler_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply runtime overrides from configs/coverage.json to the risk strategy in cfg.
    Supports: per_behavior_total (M), rng_seed, min_per_domain. Recomputes high/med/low to match M.
    """
    try:
        cov = CoverageConfig().load_coverage()
    except Exception:
        cov = {"mode": "exhaustive"}
    sampler = (cov or {}).get("sampler") or {}
    if not sampler:
        return cfg
    cfg2 = deepcopy(cfg)
    strat = cfg2.get("risk_tiers", {}).get("strategy", {})
    alloc = strat.get("allocation", {})
    # overrides
    rng_seed = sampler.get("rng_seed")
    per_total = sampler.get("per_behavior_total")
    min_per_domain = sampler.get("min_per_domain")
    if isinstance(rng_seed, (int, float)):
        strat["rng_seed"] = int(rng_seed)
    if isinstance(min_per_domain, (int, float)):
        alloc["min_per_domain"] = int(min_per_domain)
    if isinstance(per_total, (int, float)) and per_total > 0:
        # scale high/medium/low to new total using original ratios
        h0 = float(alloc.get("high", 0) or 0)
        m0 = float(alloc.get("medium", 0) or 0)
        l0 = float(alloc.get("low", 0) or 0)
        s0 = max(1.0, h0 + m0 + l0)
        M = int(per_total)
        # initial rounded
        h = int(round(M * (h0 / s0)))
        m = int(round(M * (m0 / s0)))
        l = max(0, M - h - m)
        # adjust to ensure sum == M
        delta = (h + m + l) - M
        if delta != 0:
            # adjust largest bucket by subtracting delta
            buckets = [(h, 'high'), (m, 'medium'), (l, 'low')]
            buckets.sort(reverse=True)
            if buckets[0][1] == 'high':
                h -= delta
            elif buckets[0][1] == 'medium':
                m -= delta
            else:
                l -= delta
        alloc["per_behavior_total"] = M
        alloc["high"], alloc["medium"], alloc["low"] = int(h), int(m), int(l)
    # write back
    cfg2["risk_tiers"]["strategy"] = strat
    cfg2["risk_tiers"]["strategy"]["allocation"] = alloc
    return cfg2


def build_per_behavior_datasets_v2(
    *,
    domains: Optional[Iterable[str]] = None,
    behaviors: Optional[Iterable[str]] = None,
    version: str = "1.0.0",
    seed: int = 42,
    user_turns: int = 2,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cfg = load_commerce_config()
    cfg = _apply_sampler_overrides(cfg)
    tax = cfg["taxonomy"]
    all_domains = list(domains) if domains is not None else list(tax.get("domains", []))
    all_behaviors = list(behaviors) if behaviors is not None else list(tax.get("behaviors", []))

    outputs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for b in all_behaviors:
        manifest = sample_for_behavior(cfg, b)
        # Filter by domain if provided
        for sc in manifest.get("scenarios", []):
            d = sc.get("domain")
            if domains is not None and d not in all_domains:
                continue
            axes = sc.get("axes", {})
            ds, gd = build_records(domain=d, behavior=b, axes=axes, version=version, seed=seed, user_turns=user_turns)
            outputs.append((ds, gd))
    return outputs


def build_domain_combined_datasets_v2(
    *,
    domains: Optional[Iterable[str]] = None,
    behaviors: Optional[Iterable[str]] = None,
    version: str = "1.0.0",
    seed: int = 42,
    user_turns: int = 2,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Group per domain: aggregate conversations and goldens
    per = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=version, seed=seed, user_turns=user_turns)
    by_domain: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    import re
    def _slug(s: str) -> str:
        s2 = re.sub(r"[^A-Za-z0-9._-]+", "-", (s or "").strip().lower())
        s2 = re.sub(r"-+", "-", s2).strip('-')
        return s2
    for ds, gd in per:
        # each ds currently has one conversation; append into domain bucket
        meta = ds.get("conversations", [{}])[0].get("metadata", {})
        domain_label = meta.get("domain_label") or (ds.get("dataset_id", "").split("-")[0])
        key = _slug(domain_label)
        if key not in by_domain:
            by_domain[key] = ({
                "dataset_id": f"coverage-{key}-combined-{version}",
                "version": version,
                "metadata": {"domain": "commerce", "difficulty": "mixed", "tags": ["combined", domain_label]},
                "conversations": [],
            }, {"dataset_id": f"coverage-{key}-combined-{version}", "version": version, "entries": []})
        # Merge conversations and golden entries
        by_domain[key][0]["conversations"].extend(ds.get("conversations", []))
        by_domain[key][1]["entries"].extend(gd.get("entries", []))
    return list(by_domain.values())


def build_global_combined_dataset_v2(
    *,
    domains: Optional[Iterable[str]] = None,
    behaviors: Optional[Iterable[str]] = None,
    version: str = "1.0.0",
    seed: int = 42,
    user_turns: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=version, seed=seed, user_turns=user_turns)
    ds = {
        "dataset_id": f"coverage-global-combined-{version}",
        "version": version,
        "metadata": {"domain": "commerce", "difficulty": "mixed", "tags": ["combined", "global"]},
        "conversations": [],
    }
    gd = {"dataset_id": ds["dataset_id"], "version": version, "entries": []}
    for d, g in per:
        ds["conversations"].extend(d.get("conversations", []))
        gd["entries"].extend(g.get("entries", []))
    return ds, gd
