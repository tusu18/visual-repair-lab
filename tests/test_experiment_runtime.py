"""Ordinary assignment syntax stays compatible inside the bounded runtime."""

import textwrap

import pytest

from vision_sort.visual_discovery.runtime import run_program


def execute(source, expected):
    report = run_program(textwrap.dedent(source), {})
    assert report["ok"], report["error"]
    assert report["result"] == expected


@pytest.mark.parametrize("parameter", ["state", "s", "scene", "visual_state"])
def test_single_positional_parameter_can_use_any_ordinary_name(parameter):
    execute(f"def solve({parameter}):\n return {parameter}\n", {})


def test_positional_only_parameter_is_also_supported():
    execute("def solve(scene, /):\n return scene\n", {})


@pytest.mark.parametrize("signature", ["", "s, t", "s=None", "*s", "**s", "*, s"])
def test_other_entrypoint_signatures_remain_rejected(signature):
    report = run_program(f"def solve({signature}):\n return {{}}\n", {})
    assert not report["ok"]


def test_dict_subscript_unpacking_snapshots_the_rhs_before_any_write():
    execute("""
        def solve(s):
            b = {'x': 2, 'y': 7}
            b['x'], b['y'] = (b['y'], b['x'])
            return b
    """, {"x": 7, "y": 2})


def test_nested_list_tuple_targets_evaluate_stores_left_to_right():
    execute("""
        def solve(s):
            values = [0, 0, 0]
            i = 0
            i, [values[i], (values[0], i)] = (1, [10, (20, 2)])
            return {'values': values, 'i': i}
    """, {"values": [20, 10, 0], "i": 2})


def test_unpack_rhs_precedes_target_expressions_and_is_evaluated_once():
    execute("""
        def rhs(log):
            log.append('rhs')
            return [1, 2]
        def owner(log, d):
            log.append('owner')
            return d
        def key(log, k):
            log.append(k)
            return k
        def solve(s):
            log = []
            d = {}
            owner(log, d)[key(log, 'a')], owner(log, d)[key(log, 'b')] = rhs(log)
            return {'d': d, 'log': log}
    """, {"d": {"a": 1, "b": 2}, "log": ["rhs", "owner", "a", "owner", "b"]})


def test_chained_assignment_retains_shared_values_and_target_order():
    execute("""
        def solve(s):
            d = {}
            a = d['value'] = []
            a.append(3)
            i = 0
            values = [0, 0]
            i = values[i] = 1
            return {'a': a, 'd': d, 'values': values}
    """, {"a": [3], "d": {"value": [3]}, "values": [0, 1]})


def test_chained_unpacking_consumes_the_same_iterable_in_python_order():
    execute("""
        def solve(s):
            d = {}
            a, b = d['pair'] = (10, 20)
            return {'a': a, 'b': b, 'pair': list(d['pair'])}
    """, {"a": 10, "b": 20, "pair": [10, 20]})


def test_augmented_subscript_reads_value_before_rhs_and_target_once():
    execute("""
        def owner(log, d):
            log.append('owner')
            return d
        def key(log):
            log.append('key')
            return 'x'
        def rhs(log, d):
            log.append('rhs')
            d['x'] = 100
            return 3
        def solve(s):
            log = []
            d = {'x': 10}
            owner(log, d)[key(log)] += rhs(log, d)
            return {'d': d, 'log': log}
    """, {"d": {"x": 13}, "log": ["owner", "key", "rhs"]})


def test_augmented_lists_keep_aliases_for_names_and_subscripts():
    execute("""
        def solve(s):
            original = [1]
            alias = original
            original += (2, 3)
            d = {'v': original}
            d['v'] *= 2
            return {'original': original, 'alias': alias, 'd': d}
    """, {"original": [1, 2, 3, 1, 2, 3], "alias": [1, 2, 3, 1, 2, 3],
            "d": {"v": [1, 2, 3, 1, 2, 3]}})


@pytest.mark.parametrize("source", [
    "import os\ndef solve(s):\n return {}\n",
    "def solve(s):\n return open('/etc/passwd').read()\n",
    "def solve(s):\n return s.__class__\n",
    "def solve(s):\n return __rt_tmp_1\n",
    "def solve(s):\n a = [1]\n a[:], s['x'] = ([2], 3)\n return s\n",
    "def solve(s):\n for s['x'] in [1, 2]:\n  pass\n return s\n",
])
def test_import_io_attributes_helpers_and_unhandled_stores_stay_blocked(source):
    report = run_program(source, {})
    assert not report["ok"]
    assert report["error"] == "Program uses unsupported Python features."


def test_unpacking_cannot_bypass_dictionary_growth_limit():
    report = run_program(textwrap.dedent("""
        def solve(s):
            d = {}
            for i in range(10000):
                d[str(i)], d[str(i + 10000)] = (1, 2)
            return {'size': len(d)}
    """), {})
    assert not report["ok"]
    assert "resource limit" in report["error"]


def test_augmented_lists_cannot_bypass_container_growth_limit():
    report = run_program("def solve(s):\n a = [1]\n a *= 20000\n return a\n", {})
    assert not report["ok"]
    assert "resource limit" in report["error"]


def test_infinite_loop_is_still_stopped_by_wall_clock_cap():
    report = run_program("def solve(s):\n while True:\n  pass\n", {}, timeout_seconds=0.25)
    assert not report["ok"]
    assert report["error"] == "Program exceeded the time limit."


@pytest.mark.parametrize("source", [
    "def solve(s):\n d = {}\n d['a'], d['b'] = [1]\n return d\n",
    "def solve(s):\n d = {}\n d['a'], d['b'] = [1, 2, 3]\n return d\n",
])
def test_unpack_arity_failures_remain_fixed_public_errors(source):
    report = run_program(source, {})
    assert not report["ok"]
    assert report["error"] == "Program failed."
