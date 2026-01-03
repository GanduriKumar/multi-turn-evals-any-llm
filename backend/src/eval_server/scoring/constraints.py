from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple


class ConstraintError(Exception):
    pass


@dataclass
class ConstraintResult:
    passed: bool
    message: str | None = None


def _safe_eval_expr(expr: str, context: Mapping[str, Any]) -> bool:
    """Evaluate a boolean expression safely.

    Supported: literals, boolean ops, comparisons, names from context, attribute/item access.
    Disallowed: function calls, comprehensions, lambdas, etc.
    """
    node = ast.parse(expr, mode="eval")

    def eval_node(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in context:
                raise ConstraintError(f"Unknown name: {n.id}")
            return context[n.id]
        if isinstance(n, ast.BoolOp):
            vals = [eval_node(v) for v in n.values]
            if isinstance(n.op, ast.And):
                return all(vals)
            if isinstance(n.op, ast.Or):
                return any(vals)
            raise ConstraintError("Unsupported boolean operator")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not bool(eval_node(n.operand))
        if isinstance(n, ast.Compare):
            left = eval_node(n.left)
            result = True
            for op, comparator in zip(n.ops, n.comparators):
                right = eval_node(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.In):
                    ok = left in right
                elif isinstance(op, ast.NotIn):
                    ok = left not in right
                else:
                    raise ConstraintError("Unsupported comparison operator")
                if not ok:
                    result = False
                    break
                left = right
            return result
        if isinstance(n, ast.Subscript):
            val = eval_node(n.value)
            sl = eval_node(n.slice)
            return val[sl]
        if isinstance(n, ast.Attribute):
            val = eval_node(n.value)
            return getattr(val, n.attr)
        if isinstance(n, ast.Dict):
            return {eval_node(k): eval_node(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.List):
            return [eval_node(e) for e in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(eval_node(e) for e in n.elts)
        raise ConstraintError(f"Unsupported expression element: {type(n).__name__}")

    val = eval_node(node)
    if not isinstance(val, bool):
        raise ConstraintError("Constraint expression did not evaluate to a boolean")
    return val


def _apply_builtin_rule(rule: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    # Dispatch to registry
    fn = _RULES.get(rule)
    if fn is None:
        raise ConstraintError(f"Unknown rule: {rule}")
    return fn(params, context)


def evaluate_constraints(expected_constraints: list[Mapping[str, Any]] | None, context: Mapping[str, Any]) -> Tuple[bool, list[ConstraintResult]]:
    """Evaluate a list of constraints against a context.

    - expected_constraints: list of objects with either {rule, params?} or {expr}.
    - context: arbitrary mapping with fields referenced by expressions or rules.
    Returns (all_passed, detailed_results).
    """
    if not expected_constraints:
        return True, []

    results: list[ConstraintResult] = []
    all_ok = True
    for c in expected_constraints:
        try:
            if "expr" in c:
                ok = _safe_eval_expr(str(c["expr"]), context)
                msg = c.get("message") if not ok else None
                results.append(ConstraintResult(ok, msg))
                all_ok = all_ok and ok
            elif "rule" in c:
                ok = _apply_builtin_rule(str(c["rule"]), c.get("params", {}), context)
                msg = c.get("message") if not ok else None
                results.append(ConstraintResult(ok, msg))
                all_ok = all_ok and ok
            else:
                raise ConstraintError("Constraint must define either 'expr' or 'rule'")
        except ConstraintError as e:
            results.append(ConstraintResult(False, str(e)))
            all_ok = False
    return all_ok, results


# ----- Rule Registry -----
RuleFunc = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
_RULES: Dict[str, RuleFunc] = {}


def register_rule(name: str, fn: RuleFunc) -> None:
    key = name.strip()
    if not key:
        raise ValueError("Rule name must be non-empty")
    if key in _RULES:
        raise ValueError(f"Rule already registered: {name}")
    _RULES[key] = fn


def get_rule(name: str) -> RuleFunc:
    try:
        return _RULES[name]
    except KeyError as e:
        raise ConstraintError(f"Unknown rule: {name}") from e


# Register generic, domain-agnostic rules here
def _rule_text_must_include(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    text = str(context.get("text", ""))
    terms = [str(x) for x in (params.get("terms", []) or [])]
    t = text.lower()
    return all(term.lower() in t for term in terms)


def _rule_text_must_not_include(params: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    text = str(context.get("text", ""))
    terms = [str(x) for x in (params.get("terms", []) or [])]
    t = text.lower()
    return all(term.lower() not in t for term in terms)


register_rule("text_must_include", _rule_text_must_include)
register_rule("text_must_not_include", _rule_text_must_not_include)

# Attempt to auto-load domain-specific rule packs
try:  # pragma: no cover - best-effort import
    from .rules import commerce as _commerce_rules  # noqa: F401
except Exception:
    pass


__all__ = [
    "ConstraintError",
    "ConstraintResult",
    "evaluate_constraints",
    "register_rule",
    "get_rule",
]
