"""Execute a small, resource-bounded subset of Python for visual repair tasks.

Candidates receive only JSON state in a fresh interpreter with a temporary working
directory, empty environment and a restricted builtin/method vocabulary. Source,
AST, input, output, arithmetic/container growth and wall time are capped; OS
CPU/memory limits are also attempted where supported. This is not a secure OS
sandbox for hostile, adversarial Python.
In particular, some platforms cannot enforce the optional process memory limit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from typing import Any

MAX_SOURCE_BYTES = 16_000
MAX_AST_NODES = 4_000
MAX_JSON_BYTES = 128 * 1024
MAX_TIMEOUT_SECONDS = 10.0
MAX_SEQUENCE_ITEMS = 16_384
MAX_INTEGER_BITS = 4_096

_BUILTINS = (
    "min",
    "max",
    "abs",
    "round",
    "len",
    "range",
    "enumerate",
    "sorted",
    "any",
    "all",
    "zip",
    "int",
    "float",
    "bool",
    "sum",
    "list",
    "dict",
    "tuple",
    "str",
    "reversed",
)
_METHODS = {
    "get",
    "items",
    "keys",
    "values",
    "copy",
    "append",
    "extend",
    "pop",
    "remove",
    "count",
    "index",
    "lower",
    "upper",
    "split",
    "strip",
    "replace",
    "startswith",
    "endswith",
    "join",
}
_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Name,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.ListComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Load,
    ast.Store,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}

# The runner is standalone: it never imports the repository or exposes its own
# imports/globals to the candidate. It returns only fixed, public error messages.
_RUNNER = r"""
import ast
import builtins
import json
import math
import operator
import sys

try:
    import resource
    for name, limit in (("RLIMIT_CPU", 8), ("RLIMIT_AS", 256 * 1024 * 1024),
                        ("RLIMIT_FSIZE", 0)):
        try:
            resource.setrlimit(getattr(resource, name), (limit, limit))
        except (AttributeError, OSError, ValueError):
            pass
except ImportError:
    pass

LIMIT = 128 * 1024
ITEM_LIMIT = 16384
BIT_LIMIT = 4096
allocation_budget = [262144, 1024 * 1024]

class LimitError(Exception):
    pass

class FuelError(Exception):
    pass

fuel = [250000]
def trace_fuel(frame, event, arg):
    if event in ("line", "call"):
        fuel[0] -= 1
        if fuel[0] < 0:
            raise FuelError()
    return trace_fuel

def charge(items=0, characters=0):
    allocation_budget[0] -= items
    allocation_budget[1] -= characters
    if min(allocation_budget) < 0:
        raise LimitError()

def bounded(value):
    kind = type(value)
    if kind is int and value.bit_length() > BIT_LIMIT:
        raise LimitError()
    if kind is str:
        if len(value) > LIMIT:
            raise LimitError()
        charge(characters=len(value))
    elif kind in (list, tuple, dict):
        if len(value) > ITEM_LIMIT:
            raise LimitError()
        charge(items=len(value))
    return value

def bounded_list(values=()):
    result = []
    for value in values:
        if len(result) >= ITEM_LIMIT:
            raise LimitError()
        charge(items=1)
        result.append(bounded(value))
    return result

def bounded_dict(values=(), **kwargs):
    result = {}
    values = values.items() if type(values) is dict else values
    for key, value in values:
        if key not in result:
            if len(result) >= ITEM_LIMIT:
                raise LimitError()
            charge(items=1)
        result[key] = value
    for key, value in kwargs.items():
        bounded_set(result, key, value)
    return result

def bounded_range(*args):
    result = range(*args)
    if len(result) > ITEM_LIMIT:
        raise LimitError()
    return result

def bounded_str(value=''):
    if type(value) not in (str, int, float, bool, type(None)):
        raise LimitError()
    return bounded(str(value))

def bounded_sum(values, start=0):
    values = bounded_list(values)
    if type(start) not in (int, float, bool) or any(type(value) not in (int, float, bool) for value in values):
        raise LimitError()
    return bounded(sum(values, start))

def bounded_round(value, ndigits=None):
    if ndigits is not None and abs(ndigits) > 1000:
        raise LimitError()
    return bounded(round(value) if ndigits is None else round(value, ndigits))

def bounded_set(container, key, value):
    if type(container) is dict and key not in container:
        if len(container) >= ITEM_LIMIT:
            raise LimitError()
        charge(items=1)
    container[key] = value

def bounded_assign(value, container, key):
    bounded_set(container, key, value)

OPERATORS = {name: getattr(operator, method) for name, method in (
    ('Add', 'add'), ('Sub', 'sub'), ('Mult', 'mul'), ('Div', 'truediv'),
    ('FloorDiv', 'floordiv'), ('Mod', 'mod'), ('Pow', 'pow'),
    ('BitAnd', 'and_'), ('BitOr', 'or_'), ('BitXor', 'xor'),
    ('LShift', 'lshift'), ('RShift', 'rshift'))}

def bounded_op(name, left, right):
    if name == 'Mult':
        sequence, count = (left, right) if type(left) in (str, list, tuple) else (right, left)
        if type(sequence) in (str, list, tuple) and type(count) in (int, bool):
            maximum = LIMIT if type(sequence) is str else ITEM_LIMIT
            if len(sequence) * max(0, count) > maximum:
                raise LimitError()
        if type(left) is int and type(right) is int:
            if left and right and left.bit_length() + right.bit_length() > BIT_LIMIT + 1:
                raise LimitError()
    elif name == 'Add' and type(left) in (str, list, tuple) and type(right) is type(left):
        if len(left) + len(right) > (LIMIT if type(left) is str else ITEM_LIMIT):
            raise LimitError()
    elif name == 'Pow' and type(right) in (int, float):
        if abs(right) > 1000:
            raise LimitError()
        if type(left) is int and type(right) is int and right > 0:
            if left.bit_length() * right > BIT_LIMIT:
                raise LimitError()
    elif name == 'LShift' and type(right) is int:
        if right > BIT_LIMIT or (type(left) is int and left.bit_length() + right > BIT_LIMIT):
            raise LimitError()
    elif name == 'Mod' and type(left) is str:
        raise LimitError()  # printf-style field widths can allocate before a post-check.
    return bounded(OPERATORS[name](left, right))

def bounded_augop(name, left, right):
    # Python's list += and *= retain aliases. Keep that behavior without
    # allowing either operation to grow beyond the existing container budget.
    if type(left) is list and name == 'Add':
        bounded_method(left, 'extend', right)
        return left
    result = bounded_op(name, left, right)
    if type(left) is list and name == 'Mult':
        left[:] = result
        return left
    return result

def bounded_method(receiver, name, *args, **kwargs):
    if name == 'append' and type(receiver) is list:
        if len(receiver) >= ITEM_LIMIT:
            raise LimitError()
        charge(items=1)
    elif name == 'extend' and type(receiver) is list:
        values = bounded_list(args[0])
        if len(receiver) + len(values) > ITEM_LIMIT:
            raise LimitError()
        charge(items=len(values))
        return receiver.extend(values)
    elif name == 'join' and type(receiver) is str:
        values = bounded_list(args[0])
        length = sum(len(value) for value in values) + len(receiver) * max(0, len(values) - 1)
        if length > LIMIT:
            raise LimitError()
        return bounded(receiver.join(values))
    elif name == 'replace' and type(receiver) is str:
        old, new = args[:2]
        count = args[2] if len(args) > 2 else kwargs.get('count', -1)
        occurrences = receiver.count(old)
        occurrences = occurrences if count < 0 else min(occurrences, count)
        if len(receiver) + occurrences * (len(new) - len(old)) > LIMIT:
            raise LimitError()
    return bounded(getattr(receiver, name)(*args, **kwargs))

def call_helper(name, *args):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])

class Guard(ast.NodeTransformer):
    def __init__(self):
        self.temporary_count = 0

    def temporary(self, context=ast.Load):
        self.temporary_count += 1
        return ast.Name(id='__rt_tmp_' + str(self.temporary_count), ctx=context())

    def assign_target(self, target, value):
        if isinstance(target, ast.Subscript):
            return [ast.Expr(value=call_helper('__rt_assign', value,
                                               target.value, target.slice))]
        if isinstance(target, (ast.Tuple, ast.List)):
            # Unpack this level before visiting its children, exactly as normal
            # assignment does. Deferring nested unpacking preserves left-to-right
            # stores, index expressions and alias-dependent targets.
            temporaries = [self.temporary(ast.Store) for _ in target.elts]
            statements = [ast.Assign(targets=[ast.Tuple(elts=temporaries, ctx=ast.Store())],
                                     value=value)]
            for child, temporary in zip(target.elts, temporaries):
                statements.extend(self.assign_target(child, ast.Name(id=temporary.id, ctx=ast.Load())))
            return statements
        return [ast.Assign(targets=[target], value=value)]

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return ast.copy_location(call_helper('__rt_op', ast.Constant(type(node.op).__name__),
                                             node.left, node.right), node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            node.args = [node.func.value, ast.Constant(node.func.attr)] + node.args
            node.func = ast.Name(id='__rt_method', ctx=ast.Load())
        return node

    def visit_List(self, node):
        self.generic_visit(node)
        return ast.copy_location(call_helper('__rt_value', node), node) if isinstance(node.ctx, ast.Load) else node

    visit_Tuple = visit_List

    def visit_Dict(self, node):
        self.generic_visit(node)
        return ast.copy_location(call_helper('__rt_value', node), node)

    def visit_ListComp(self, node):
        self.generic_visit(node)
        generator = ast.GeneratorExp(elt=node.elt, generators=node.generators)
        return ast.copy_location(call_helper('__rt_list', generator), node)

    def visit_DictComp(self, node):
        self.generic_visit(node)
        pair = ast.Tuple(elts=[node.key, node.value], ctx=ast.Load())
        generator = ast.GeneratorExp(elt=pair, generators=node.generators)
        return ast.copy_location(call_helper('__rt_dict', generator), node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store)
               for target in node.targets for child in ast.walk(target)):
            temporary = self.temporary(ast.Store)
            statements = [ast.Assign(targets=[temporary], value=node.value)]
            for target in node.targets:
                statements.extend(self.assign_target(target, ast.Name(id=temporary.id, ctx=ast.Load())))
            return [ast.copy_location(statement, node) for statement in statements]
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        name = ast.Constant(type(node.op).__name__)
        if isinstance(node.target, ast.Subscript):
            target = node.target
            container = self.temporary(ast.Store)
            key = self.temporary(ast.Store)
            previous = self.temporary(ast.Store)
            container_value = ast.Name(id=container.id, ctx=ast.Load())
            key_value = ast.Name(id=key.id, ctx=ast.Load())
            previous_value = ast.Name(id=previous.id, ctx=ast.Load())
            # Unlike ordinary assignment, augmented assignment evaluates and
            # reads its target before evaluating the RHS, exactly once each.
            statements = [
                ast.Assign(targets=[container], value=target.value),
                ast.Assign(targets=[key], value=target.slice),
                ast.Assign(targets=[previous], value=ast.Subscript(
                    value=container_value, slice=key_value, ctx=ast.Load())),
                ast.Expr(value=call_helper('__rt_assign',
                    call_helper('__rt_augop', name, previous_value, node.value),
                    container_value, key_value)),
            ]
            return [ast.copy_location(statement, node) for statement in statements]
        value = call_helper('__rt_augop', name, ast.Name(id=node.target.id, ctx=ast.Load()), node.value)
        return ast.copy_location(ast.Assign(targets=[node.target], value=value), node)

def valid_json(value, depth=0, active=None, budget=None):
    if active is None:
        active = set()
        budget = [16384]
    budget[0] -= 1
    if budget[0] < 0 or depth > 40:
        return False
    kind = type(value)
    if kind is float:
        return math.isfinite(value)
    if kind is int:
        return value.bit_length() <= BIT_LIMIT
    if kind in (str, bool, type(None)):
        return True
    if kind not in (dict, list) or id(value) in active:
        return False
    active.add(id(value))
    if kind is dict:
        okay = all(type(key) is str and valid_json(item, depth + 1, active, budget)
                   for key, item in value.items())
    else:
        okay = all(valid_json(item, depth + 1, active, budget) for item in value)
    active.remove(id(value))
    return okay

def encode_limited(value):
    chunks = []
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=True).iterencode(value):
        size += len(chunk)
        if size > LIMIT:
            return None
        chunks.append(chunk)
    return ''.join(chunks)

response = {"ok": False, "result": None, "error": "Program failed."}
try:
    payload = json.loads(sys.stdin.buffer.read(LIMIT + 1))
    allowed = {name: getattr(builtins, name) for name in BUILTIN_NAMES}
    allowed.update(range=bounded_range, list=bounded_list, dict=bounded_dict, str=bounded_str,
                   sum=bounded_sum, round=bounded_round,
                   tuple=lambda values=(): bounded(tuple(bounded_list(values))),
                   sorted=lambda values, **kwargs: bounded(sorted(bounded_list(values), **kwargs)),
                   reversed=lambda values: reversed(bounded(values)),
                   int=lambda *args, **kwargs: bounded(int(*args, **kwargs)))
    scope = {"__builtins__": allowed, "__rt_op": bounded_op, "__rt_method": bounded_method,
             "__rt_value": bounded, "__rt_list": bounded_list, "__rt_dict": bounded_dict,
             "__rt_assign": bounded_assign, "__rt_augop": bounded_augop}
    tree = ast.fix_missing_locations(Guard().visit(ast.parse(payload["source"])))
    code = compile(tree, "<candidate>", "exec")
    sys.settrace(trace_fuel)
    exec(code, scope, scope)
    result = scope["solve"](payload["state"])
    sys.settrace(None)
    if not valid_json(result):
        response["error"] = "Program returned non-JSON or non-finite data."
    else:
        encoded_result = encode_limited(result)
        if encoded_result is None or len(encoded_result) > 8192:
            response["error"] = "Program result is too large."
        else:
            response = {"ok": True, "result": result, "error": ""}
except FuelError:
    sys.settrace(None)
    response["error"] = "Program exceeded the deterministic instruction budget."
except (MemoryError, LimitError, OverflowError):
    sys.settrace(None)
    response["error"] = "Program exceeded a resource limit."
except BaseException:
    sys.settrace(None)
    response["error"] = "Candidate raised an exception."
try:
    encoded = encode_limited(response)
    if encoded is None:
        encoded = '{"ok":false,"result":null,"error":"Program result is too large."}'
    sys.stdout.write(encoded)
except BaseException:
    sys.stdout.write('{"ok":false,"result":null,"error":"Program failed."}')
""".replace("BUILTIN_NAMES", repr(_BUILTINS))


def _validate_source(source: str) -> str | None:
    try:
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
    except (SyntaxError, ValueError, RecursionError, UnicodeError):
        return "Program is not valid Python."
    if len(nodes) > MAX_AST_NODES:
        return "Program is too complex."
    functions = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    solve = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"), None
    )
    if solve is None:
        return "Program must define solve(state)."
    args = solve.args
    positional = args.posonlyargs + args.args
    if (
        len(positional) != 1
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
    ):
        return "Program must define solve(state)."
    call_targets = set(_BUILTINS) | functions
    parents = {id(child): node for node in nodes for child in ast.iter_child_nodes(node)}
    attribute_calls = {
        id(node.func) for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for node in nodes:
        if type(node) not in _NODES:
            return "Unsupported Python construct: " + type(node).__name__ + "."
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value.bit_length() > MAX_INTEGER_BITS
        ):
            return "Program contains an oversized integer."
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            parent = parents.get(id(node))
            while isinstance(parent, (ast.Tuple, ast.List)) and isinstance(parent.ctx, ast.Store):
                parent = parents.get(id(parent))
            if isinstance(node.slice, ast.Slice) or not isinstance(parent, (ast.AugAssign, ast.Assign)):
                return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else (node.arg if isinstance(node, ast.arg) else node.name)
            )
            if "__" in name:
                return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, ast.FunctionDef) and (
            node.decorator_list
            or node.returns
            or any(
                argument.annotation
                for argument in (
                    node.args.posonlyargs
                    + node.args.args
                    + node.args.kwonlyargs
                    + ([node.args.vararg] if node.args.vararg else [])
                    + ([node.args.kwarg] if node.args.kwarg else [])
                )
            )
        ):
            return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, ast.Attribute) and (
            node.attr not in _METHODS or id(node) not in attribute_calls
        ):
            return "Unsupported method or attribute: " + node.attr + "."
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in call_targets:
                    return "Unsupported function: " + node.func.id + "."
            elif not isinstance(node.func, ast.Attribute):
                return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, ast.keyword) and node.arg is None:
            return "Unsupported Python construct: " + type(node).__name__ + "."
        if isinstance(node, ast.comprehension) and node.is_async:
            return "Unsupported Python construct: " + type(node).__name__ + "."
    return None


def _json_state(
    value: Any, depth: int = 0, active: set[int] | None = None, budget: list[int] | None = None
) -> bool:
    budget = [MAX_JSON_BYTES] if budget is None else budget
    budget[0] -= 1
    if depth > 40 or budget[0] < 0:
        return False
    kind = type(value)
    if kind is float:
        return math.isfinite(value)
    if kind is int:
        return value.bit_length() <= MAX_INTEGER_BITS
    if kind is str:
        return len(value) <= MAX_JSON_BYTES
    if kind in (bool, type(None)):
        return True
    if kind not in (dict, list):
        return False
    if len(value) > MAX_SEQUENCE_ITEMS:
        return False
    active = set() if active is None else active
    if id(value) in active:
        return False
    active.add(id(value))
    if kind is dict:
        okay = all(
            type(key) is str and len(key) <= MAX_JSON_BYTES and _json_state(item, depth + 1, active, budget)
            for key, item in value.items()
        )
    else:
        okay = all(_json_state(item, depth + 1, active, budget) for item in value)
    active.remove(id(value))
    return okay


def run_program(source: str, state: dict, *, timeout_seconds: float = MAX_TIMEOUT_SECONDS) -> dict:
    """Run ``solve(state)`` and return a public JSON-compatible execution report.

    ``timeout_seconds`` can shorten, but cannot increase, the ten-second wall-time
    fallback. Deterministic instruction fuel is the candidate limit. Elapsed time includes validation, interpreter startup and execution.
    Candidate exceptions, paths and input values are never included in errors.
    """
    started = time.perf_counter()
    source_hash = hashlib.sha256()
    if isinstance(source, str):
        for offset in range(0, len(source), 8192):
            source_hash.update(source[offset : offset + 8192].encode("utf-8", errors="replace"))
    digest = source_hash.hexdigest()

    def report(error: str = "", result: Any = None) -> dict:
        return {
            "ok": not bool(error),
            "error_code": error_code(error),
            "infrastructure_error": error_code(error).startswith("infrastructure_"),
            "result": result,
            "error": error,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "sha256": digest,
        }

    if not isinstance(source, str):
        return report("Program must be text.")
    if len(source) > MAX_SOURCE_BYTES or len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return report("Program is too large.")
    error = _validate_source(source)
    if error:
        return report(error)
    if type(state) is not dict or not _json_state(state):
        return report("State must be a finite JSON object.")
    try:
        timeout = min(float(timeout_seconds), MAX_TIMEOUT_SECONDS)
        chunks, size = [], 0
        for chunk in json.JSONEncoder(allow_nan=False).iterencode({"source": source, "state": state}):
            size += len(chunk)
            if size > MAX_JSON_BYTES:
                return report("Input is too large.")
            chunks.append(chunk)
        payload = "".join(chunks).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return report("Input is not valid JSON or the execution limit is invalid.")
    if not math.isfinite(timeout) or timeout <= 0:
        return report("Execution limit is invalid.")
    if len(payload) > MAX_JSON_BYTES:
        return report("Input is too large.")
    try:
        with tempfile.TemporaryDirectory(prefix="visual-repair-") as directory:
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=payload,
                capture_output=True,
                cwd=directory,
                env={},
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return report("Evaluator wall-clock deadline exceeded.")
    except (OSError, ValueError):
        return report("Program runner failed.")
    if process.returncode != 0 or len(process.stdout) > MAX_JSON_BYTES:
        return report("Evaluator worker exited without a valid result.")
    try:
        response = json.loads(process.stdout)
        if (
            type(response) is not dict
            or type(response.get("ok")) is not bool
            or type(response.get("error")) is not str
        ):
            return report("Program runner failed.")
        return report(response["error"], response.get("result"))
    except (ValueError, UnicodeError):
        return report("Program runner failed.")


def error_code(error):
    if not error:
        return "ok"
    if error == "Evaluator wall-clock deadline exceeded.":
        return "infrastructure_deadline"
    if error in ("Program runner failed.", "Evaluator worker exited without a valid result."):
        return "infrastructure_worker"
    if "deterministic instruction" in error:
        return "candidate_instruction_limit"
    if "resource limit" in error:
        return "candidate_allocation_limit"
    if error == "Candidate raised an exception.":
        return "candidate_exception"
    if "returned" in error or "result is too large" in error:
        return "candidate_output"
    return "candidate_validation"


class EvaluationInfrastructureError(RuntimeError):
    """No reward/grade may be emitted for this transition. Abort and quarantine."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def checked_run(source, state):
    result = run_program(source, state)
    if result['infrastructure_error']:
        raise EvaluationInfrastructureError(result['error_code'])
    return result


def language_contract():
    return {
        "entry_point": "def solve(state): return a finite JSON object",
        "builtins": list(_BUILTINS), "methods": sorted(_METHODS),
        "ast_nodes": sorted(n.__name__ for n in _NODES),
        "source_bytes": MAX_SOURCE_BYTES, "ast_nodes_limit": MAX_AST_NODES,
        "instruction_events": 250000, "container_items": MAX_SEQUENCE_ITEMS,
        "json_bytes": MAX_JSON_BYTES, "result_bytes": 8192,
        "unsupported": ["imports", "set() and set literals", "classes", "lambda", "try/except", "file/network access", "dunder names", "slice assignment", "decorators", "annotations"],
        "hint": "Use lists with membership checks instead of sets; list.remove is supported.",
        "infrastructure": "Wall-clock/worker failure aborts the episode without a score; it is not a candidate failure.",
    }


def checked_batch(source, states):
    """One isolated worker, fresh candidate globals and deterministic fuel per case.

    Batching avoids interpreter-startup time dominating protected evaluation.
    A worker/deadline failure invalidates the whole grading operation, never a
    single randomly selected case. No retry or partial score is synthesized.
    """
    if not states:
        return []
    if len(states) > 128:
        raise EvaluationInfrastructureError('infrastructure_batch_size')
    validation = _validate_source(source) if isinstance(source,str) and len(source.encode()) <= MAX_SOURCE_BYTES else 'Program is too large.'
    if validation:
        return [{'ok':False,'result':None,'error':validation,'error_code':'candidate_validation',
                 'infrastructure_error':False} for _ in states]
    if any(type(s) is not dict or not _json_state(s) for s in states):
        raise EvaluationInfrastructureError('infrastructure_invalid_state')
    marker = 'response = {"ok": False, "result": None, "error": "Program failed."}'
    prefix, body = _RUNNER.split(marker,1)
    body = marker + body.split('try:\n    encoded = encode_limited(response)',1)[0]
    body = body.replace('    payload = json.loads(sys.stdin.buffer.read(LIMIT + 1))\n','')
    function = 'def evaluate(payload):\n    global allocation_budget\n    allocation_budget = [262144, 1024*1024]\n    fuel[0] = 250000\n' + '\n'.join('    '+line for line in body.splitlines()) + '\n    return response\n'
    runner = prefix + function + '''
payload = json.loads(sys.stdin.buffer.read(2097153))
results = [evaluate({'source':payload['source'],'state':state}) for state in payload['states']]
sys.stdout.write(json.dumps(results,allow_nan=False,separators=(',',':')))
'''
    payload = json.dumps({'source':source,'states':states},allow_nan=False).encode()
    if len(payload)>2097152:
        raise EvaluationInfrastructureError('infrastructure_batch_size')
    try:
        with tempfile.TemporaryDirectory(prefix='visual-repair-batch-') as directory:
            process = subprocess.run([sys.executable,'-I','-S','-c',runner],input=payload,
                capture_output=True,cwd=directory,env={},timeout=10,check=False)
    except subprocess.TimeoutExpired:
        raise EvaluationInfrastructureError('infrastructure_deadline') from None
    except OSError:
        raise EvaluationInfrastructureError('infrastructure_worker') from None
    if process.returncode or len(process.stdout)>2097152:
        raise EvaluationInfrastructureError('infrastructure_worker')
    try:
        results = json.loads(process.stdout)
        if type(results) is not list or len(results)!=len(states):
            raise ValueError
        for result in results:
            if type(result) is not dict or type(result.get('ok')) is not bool or type(result.get('error')) is not str:
                raise ValueError
            result['error_code'] = error_code(result['error'])
            result['infrastructure_error'] = False
        return results
    except (ValueError,KeyError,TypeError):
        raise EvaluationInfrastructureError('infrastructure_worker') from None
