from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _key_for_conv(c: Dict[str, Any]) -> str:
    # Prefer stable human keys, fallback to id
    for k in ("conversation_slug", "conversation_title", "conversation_id"):
        v = c.get(k)
        if isinstance(v, str) and v:
            return v
    return str(c.get("conversation_id") or "")


def _aggregate(results: Dict[str, Any]) -> Dict[str, Any]:
    convs = list(results.get("conversations", []) or [])
    conv_total = len(convs)
    conv_pass = 0
    turn_total = 0
    turn_pass = 0
    metrics_names = ["exact", "semantic", "consistency", "adherence", "hallucination"]
    metric_counts = {m: {"total": 0, "pass": 0} for m in metrics_names}
    high_sev = 0
    domains: Dict[Tuple[str, str], Dict[str, int]] = {}
    for c in convs:
        summ = c.get("summary") or {}
        if bool(summ.get("conversation_pass", True)):
            conv_pass += 1
        if summ.get("high_severity_violation"):
            high_sev += 1
        dom = c.get("domain") or ""
        beh = c.get("behavior") or ""
        key = (dom, beh)
        if key not in domains:
            domains[key] = {"conv_total": 0, "conv_pass": 0}
        domains[key]["conv_total"] += 1
        if bool(summ.get("conversation_pass", True)):
            domains[key]["conv_pass"] += 1
        for t in c.get("turns", []) or []:
            turn_total += 1
            if bool(t.get("turn_pass", True)):
                turn_pass += 1
            mets = t.get("metrics") or {}
            for m in metrics_names:
                if m in mets and isinstance(mets[m], dict) and ("pass" in mets[m]):
                    metric_counts[m]["total"] += 1
                    if bool(mets[m].get("pass")):
                        metric_counts[m]["pass"] += 1
    def rate(p: int, tot: int) -> float:
        return (100.0 * p / tot) if tot > 0 else 0.0
    agg = {
        "conv": {"total": conv_total, "pass": conv_pass, "pass_rate": rate(conv_pass, conv_total)},
        "turn": {"total": turn_total, "pass": turn_pass, "pass_rate": rate(turn_pass, turn_total)},
        "metrics": {m: {"total": metric_counts[m]["total"], "pass": metric_counts[m]["pass"], "pass_rate": rate(metric_counts[m]["pass"], metric_counts[m]["total"]) } for m in metrics_names},
        "high_severity_count": high_sev,
        "domains": [
            {"domain": k[0], "behavior": k[1], "conv_total": v["conv_total"], "conv_pass": v["conv_pass"], "pass_rate": rate(v["conv_pass"], v["conv_total"]) }
            for k, v in domains.items()
        ],
        "input_tokens_total": results.get("input_tokens_total"),
        "output_tokens_total": results.get("output_tokens_total"),
    }
    return agg


def _align(results_a: Dict[str, Any], results_b: Dict[str, Any]) -> Dict[str, Any]:
    a_convs = list(results_a.get("conversations", []) or [])
    b_convs = list(results_b.get("conversations", []) or [])
    a_map = {_key_for_conv(c): c for c in a_convs}
    b_map = {_key_for_conv(c): c for c in b_convs}
    keys_a = set(a_map.keys())
    keys_b = set(b_map.keys())
    matched_keys = sorted(list(keys_a & keys_b))
    unmatched_a = sorted(list(keys_a - keys_b))
    unmatched_b = sorted(list(keys_b - keys_a))
    return {
        "matched_keys": matched_keys,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "a_map": a_map,
        "b_map": b_map,
    }


def _per_conversation_deltas(a_map: Dict[str, Any], b_map: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k in keys:
        ca = a_map.get(k) or {}
        cb = b_map.get(k) or {}
        sa = (ca.get("summary") or {})
        sb = (cb.get("summary") or {})
        a_pass = bool(sa.get("conversation_pass", True))
        b_pass = bool(sb.get("conversation_pass", True))
        a_failed_turns = int(sa.get("failed_turns_count") or 0)
        b_failed_turns = int(sb.get("failed_turns_count") or 0)
        out.append({
            "key": k,
            "a": {"conversation_pass": a_pass, "failed_turns_count": a_failed_turns},
            "b": {"conversation_pass": b_pass, "failed_turns_count": b_failed_turns},
            "delta": {"conversation_pass_changed": (a_pass != b_pass), "failed_turns_delta": (b_failed_turns - a_failed_turns)},
        })
    return out


def _per_turn_deltas(a_map: Dict[str, Any], b_map: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    metrics_names = ["exact", "semantic", "consistency", "adherence", "hallucination"]
    for k in keys:
        ca = a_map.get(k) or {}
        cb = b_map.get(k) or {}
        ta = list(ca.get("turns", []) or [])
        tb = list(cb.get("turns", []) or [])
        # index by turn_index
        ia = {int(t.get("turn_index", 0)): t for t in ta}
        ib = {int(t.get("turn_index", 0)): t for t in tb}
        turn_keys = sorted(set(ia.keys()) | set(ib.keys()))
        for tid in turn_keys:
            va = ia.get(tid) or {}
            vb = ib.get(tid) or {}
            row: Dict[str, Any] = {"key": k, "turn_index": tid}
            a_pass = bool(va.get("turn_pass", True))
            b_pass = bool(vb.get("turn_pass", True))
            row["turn_pass"] = {"a": a_pass, "b": b_pass, "changed": (a_pass != b_pass)}
            mets_a = va.get("metrics") or {}
            mets_b = vb.get("metrics") or {}
            md: Dict[str, Any] = {}
            for m in metrics_names:
                ma = mets_a.get(m) or {}
                mb = mets_b.get(m) or {}
                pa = ma.get("pass")
                pb = mb.get("pass")
                if pa is None and pb is None:
                    continue
                md[m] = {"a": bool(pa) if pa is not None else None, "b": bool(pb) if pb is not None else None, "changed": (pa is not None and pb is not None and bool(pa) != bool(pb))}
            row["metrics"] = md
            out.append(row)
    return out


def diff_results(results_a: Dict[str, Any], results_b: Dict[str, Any]) -> Dict[str, Any]:
    agg_a = _aggregate(results_a)
    agg_b = _aggregate(results_b)
    align = _align(results_a, results_b)
    matched = align["matched_keys"]
    per_conv = _per_conversation_deltas(align["a_map"], align["b_map"], matched)
    per_turn = _per_turn_deltas(align["a_map"], align["b_map"], matched)
    def delta_rate(b: float, a: float) -> float:
        return float(b) - float(a)
    # metric deltas
    md = {}
    for m, va in (agg_a.get("metrics") or {}).items():
        vb = (agg_b.get("metrics") or {}).get(m) or {"pass_rate": 0}
        md[m] = {
            "a_pass_rate": va.get("pass_rate", 0.0),
            "b_pass_rate": vb.get("pass_rate", 0.0),
            "delta": delta_rate(vb.get("pass_rate", 0.0), va.get("pass_rate", 0.0)),
        }
    # domain/behavior deltas
    def _db_map(agg: Dict[str, Any]):
        out = {}
        for r in agg.get("domains") or []:
            out[(r.get("domain") or "", r.get("behavior") or "")] = r
        return out
    db_a = _db_map(agg_a)
    db_b = _db_map(agg_b)
    db_keys = sorted(set(db_a.keys()) | set(db_b.keys()))
    db_rows = []
    for k in db_keys:
        ra = db_a.get(k) or {"pass_rate": 0.0, "conv_total": 0, "conv_pass": 0}
        rb = db_b.get(k) or {"pass_rate": 0.0, "conv_total": 0, "conv_pass": 0}
        db_rows.append({
            "domain": k[0],
            "behavior": k[1],
            "a": {"pass_rate": ra.get("pass_rate", 0.0), "conv_total": ra.get("conv_total", 0), "conv_pass": ra.get("conv_pass", 0)},
            "b": {"pass_rate": rb.get("pass_rate", 0.0), "conv_total": rb.get("conv_total", 0), "conv_pass": rb.get("conv_pass", 0)},
            "delta": {"pass_rate": delta_rate(rb.get("pass_rate", 0.0), ra.get("pass_rate", 0.0))},
        })
    return {
        "runA": {"run_id": results_a.get("run_id"), "dataset_id": results_a.get("dataset_id"), "model_spec": results_a.get("model_spec"), "summary": agg_a},
        "runB": {"run_id": results_b.get("run_id"), "dataset_id": results_b.get("dataset_id"), "model_spec": results_b.get("model_spec"), "summary": agg_b},
        "alignment": {
            "matched": matched,
            "unmatched_a": align["unmatched_a"],
            "unmatched_b": align["unmatched_b"],
            "note": "This analysis compares any two runs. If datasets differ, treat results as directional — not an apples-to-apples comparison.",
        },
        "metrics_delta": md,
        "domain_behavior_delta": db_rows,
        "per_conversation": per_conv,
        "per_turn": per_turn,
    }
