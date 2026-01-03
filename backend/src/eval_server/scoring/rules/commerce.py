from __future__ import annotations

from typing import Any, Mapping

from ..constraints import register_rule


def _refund_after_ship(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    shipped = bool(context.get("shipped", False))
    requested_refund = bool(context.get("requested_refund", False))
    return not (shipped and requested_refund)


def _max_discount_percent(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    try:
        max_val = float(params.get("max"))
        disc = float(context.get("discount_percent"))
    except Exception:
        return False
    return disc <= max_val


def _require_reason_code_in(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    allowed = params.get("allowed", []) or []
    rc = context.get("reason_code")
    return isinstance(rc, str) and rc in allowed


def _order_total_min(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    try:
        min_val = float(params.get("min"))
        total = float(context.get("total"))
    except Exception:
        return False
    return total >= min_val


def _allowed_countries(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    allowed = [str(x).strip().casefold() for x in (params.get("allowed", []) or [])]
    c = context.get("country")
    return isinstance(c, str) and c.strip().casefold() in allowed


# Register commerce rules under a namespace
register_rule("commerce.refund_after_ship", _refund_after_ship)
register_rule("commerce.max_discount_percent", _max_discount_percent)
register_rule("commerce.require_reason_code_in", _require_reason_code_in)
register_rule("commerce.order_total_min", _order_total_min)
register_rule("commerce.allowed_countries", _allowed_countries)
