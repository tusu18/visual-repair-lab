"""Semantic output comparison: rounding is exclusively a display operation."""
import math

ABS_TOL = 1e-6
REL_TOL = 1e-9


def equivalent(a, b):
    if isinstance(b, bool) or b is None or isinstance(b, str):
        return type(a) is type(b) and a == b
    if type(b) in (int, float):
        return type(a) in (int, float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if isinstance(b, dict):
        return isinstance(a, dict) and a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return False
