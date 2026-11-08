"""Autonomous visual investigations; private goals enter RGB and terminal grading only."""

from __future__ import annotations

import copy
import inspect
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .runtime import run_program

PANEL_WIDTH, HEIGHT, WIDTH = 600, 700, 1200
FAMILIES = ("canvas_layout", "chart_encoding", "platform_layers")
CARD_IDS = tuple("ABCDEFGHI")
SERIES_IDS = tuple(f"S{i}" for i in range(1, 7))
PLATFORM_IDS = tuple(f"P{i}" for i in range(6))
COLORS = {
    "blue": "#2563eb",
    "orange": "#ea850b",
    "green": "#159447",
    "red": "#d83240",
    "purple": "#8b42c4",
    "brown": "#956039",
}
MARKERS = ("circle", "square", "triangle", "diamond", "cross", "plus")
CLASSES = ("solid", "one_way", "decorative")


def interaction_contract(family: str) -> list[dict]:
    """Public ordinary-control schemas; no private goals, probes, or test sequences."""

    def number(low, high, **extra):
        return {"type": "number", "minimum": low, "maximum": high, **extra}

    def enum(values):
        return {"type": "string", "enum": list(values)}

    def control(kind, parameters=None, *, optional=(), effect="", constraints=()):
        parameters = parameters or {}
        return {
            "kind": kind,
            "event_schema": {
                "type": "object",
                "properties": {"kind": {"type": "string", "const": kind}, **parameters},
                "required": ["kind", *[key for key in parameters if key not in optional]],
                "additionalProperties": True,
            },
            "effect": effect,
            "constraints": list(constraints),
        }

    controls = [
        control(
            "reset_scene",
            effect="Restore initial public inputs; retain the hidden reference and candidate source.",
        )
    ]
    modes = enum(("replace", "add", "subtract"))
    point = ("x", "y")
    if family == "canvas_layout":
        controls += [
            control(
                "resize",
                {key: number(200, 600) for key in ("width", "height")},
                effect="Change public board dimensions; keep selection history.",
            ),
            control(
                "pan",
                {key: number(-500, 500) for key in ("dx", "dy")},
                effect="Add the offsets to public pan; keep selection history.",
            ),
            control(
                "zoom",
                {"factor": number(0.5, 2), **{key: number(-5000, 5000) for key in point}},
                effect="Scale around the screen anchor; resulting scale is clamped to [0.3,2].",
            ),
            control(
                "click",
                {**{key: number(-5000, 5000) for key in point}, "mode": modes},
                effect="Append one click gesture with its current public board and transform.",
                constraints=("Selection gesture history may contain at most 64 entries.",),
            ),
            control(
                "drag",
                {**{key: number(-5000, 5000) for key in ("x0", "y0", "x1", "y1")}, "mode": modes},
                effect="Append one drag gesture with its current public board and transform; endpoints may be reversed.",
                constraints=("Selection gesture history may contain at most 64 entries.",),
            ),
            control(
                "clear_selection",
                effect="Append one clear-selection gesture.",
                constraints=("Selection gesture history may contain at most 64 entries.",),
            ),
            control("reorder", effect="Reverse the public input ID order."),
        ]
    elif family == "chart_encoding":
        controls += [
            control(
                "set_view",
                {
                    "context": enum(("overview", "A", "B")),
                    "page": {"type": "integer", "minimum": 0, "maximum": 2},
                },
                effect="Set public context and page; preserve brush and selection history.",
            ),
            control(
                "brush",
                {**{key: number(-5000, 5000) for key in ("x0", "y0", "x1", "y1")}, "mode": modes},
                effect="Display the rectangle and append a gesture recording current public view/data; endpoints may be reversed.",
                constraints=("Selection gesture history may contain at most 64 entries.",),
            ),
            control(
                "clear_selection",
                effect="Append one clear-selection gesture.",
                constraints=("Selection gesture history may contain at most 64 entries.",),
            ),
            control("clear_brush", effect="Hide the displayed brush rectangle; keep selection history."),
            control(
                "resize",
                {key: number(120, 470) for key in ("width", "height")},
                effect="Change public plot dimensions; keep selection history.",
            ),
            control("new_data", effect="Change public points deterministically; keep selection history."),
            control("reorder", effect="Reverse the public input series order."),
        ]
    elif family == "platform_layers":
        body_id = enum(tuple(f"B{i}" for i in range(6)))
        velocity = {key: number(-4000, 4000) for key in ("vx", "vy")}
        controls += [
            control(
                "set_context",
                {"context": enum(("A", "B"))},
                effect="Set context; retain launches and duration.",
            ),
            control(
                "launch",
                {
                    "id": body_id,
                    **{key: number(0, 580) for key in point},
                    **velocity,
                    "width": number(8, 40, default=20),
                    "height": number(8, 40, default=24),
                },
                optional=("width", "height"),
                effect="Create or replace that public body launch; reset duration to zero.",
            ),
            control(
                "set_velocity", velocity, effect="Set all public launch velocities; reset duration to zero."
            ),
            control(
                "translate_bodies",
                {key: number(-500, 500) for key in ("dx", "dy")},
                effect="Add offsets to all public launches; reset duration to zero.",
            ),
            control("remove_body", {"id": body_id}, effect="Remove that public body; keep duration."),
            control(
                "physics",
                {"gravity": number(-1000, 1500), "dt": number(0.01, 0.3)},
                effect="Set public gravity and timestep; keep duration.",
            ),
            control(
                "advance",
                {"ticks": {"type": "integer", "minimum": 1, "maximum": 30}},
                effect="Add ticks to public duration and replay from public launches.",
                constraints=("Current state.ticks plus the supplied ticks must be at most 40.",),
            ),
            control("restart", effect="Reset duration to zero; keep launches and physics parameters."),
            control(
                "reset_bodies",
                effect="Restore neutral public bodies relative to current platforms; reset duration to zero.",
            ),
            control(
                "move_platform",
                {
                    "id": enum(PLATFORM_IDS),
                    **{key: number(0, 580) for key in point},
                    **{key: number(8, 180) for key in ("width", "height")},
                },
                effect="Set the named platform's public rectangle; keep launches and duration.",
            ),
            control("reorder", effect="Reverse public platform and body input order."),
        ]
    else:
        raise ValueError("Unknown family")
    return controls


def task_catalog() -> list[dict]:
    common = (
        "Repair Python solve(state) to match the running approved application on the left. "
        "The right panel runs your candidate. Initial views can hide behavior. You choose all "
        "observations and tests using ordinary controls. There are no supplied verification cases. "
        "Hidden reference configuration is available through reference pixels, never source or scene text. "
        "Public execution results only report execution and output-schema validity. Edit saves; run "
        "activates; interactions execute the active source. Submit ends the episode and grades additional "
        "states, including interaction histories. Pure Python, no imports or set builtin; lists and dictionaries work. Source is limited to16000bytes/4000ASTnodes. Return exactly the specified "
        "keys, JSON values, and finite numbers bounded by1000000. Screenshot panels each600x700; "
        "the full composite is1200x700. Observe restores full view; crop enlarges existing pixels. "
        "Every interaction event must include the selector field kind, whose value is one of the listed "
        "control names, plus that control's named parameters. Thus a control CONTROL{PARAM,...} "
        "uses the flat event {'kind':'CONTROL','PARAM':value,...}. The outer tool action is "
        "{'type':'interact','event':that_event}; the native response wrapper represents that_event "
        "as name/value entries, including {'name':'kind','value':'CONTROL'}. "
        "reset_scene uses {'kind':'reset_scene'} and resets public inputs. "
    )
    return [
        {
            "family": "canvas_layout",
            "title": "Stateful responsive selection",
            "description": "Repair a responsive design canvas whose selection drifts across gestures and viewport changes.",
            "prompt": common
            + (
                "state: idsA-I, board{width,height,gap}, transform{scale,pan_x,pan_y}, pointer{x,y}, "
                "gestures[]. Return {'positions':{id:[world_x,world_y,w,h]},'selected_ids':[sorted IDs]}. "
                "The grid is3x3, with equal gap padding around and between cells. The middle layout is "
                "alphabetical. Narrow and wide layouts use private permutations. Narrow is width<low, "
                "wide is width>=high; low is320 or360, high is480 or520. Transform maps world to screen. "
                "Selection persists through layout/transform changes. A click hits a card's half-open box; "
                "a drag selects cards whose boxes have positive-area intersection with the rectangle. "
                "A gesture applies replace/add/subtract to the existing set. Each gesture stores its public "
                "board/transform at the time it occurred; later resizing must not reinterpret prior gestures. "
                "Controls: resize{width,height}200..600; zoom{factor,x,y} factor0.5..2 (screen anchor, "
                "scale clamped0.3..2); pan{dx,dy}abs<=500; click{x,y,mode}; drag{x0,y0,x1,y1,mode} "
                "coordsabs<=5000, mode replace/add/subtract; clear_selection; reorder reverses input ids."
            ),
            "interactions": interaction_contract("canvas_layout"),
        },
        {
            "family": "chart_encoding",
            "title": "Nonlinear chart brushing",
            "description": "Repair chart projection and persistent stroke selection across views and brushes.",
            "prompt": common
            + (
                "state: context overview/A/B, page0..2, plot{left,top,width,height}, series[{id,points:[{x,y}]}], "
                "brush orNone, gestures[]. Page0 showsS1,S2, page1S3,S4, page2S5,S6. Return "
                "{'styles':{all IDs:{'color':name,'marker':name}},'projected_points':{all IDs:[[sx,sy],...]}, "
                "'selected_ids':[sorted IDs]}. In overview colors blue,orange,green,red,purple,brown and "
                "markers circle,square,triangle,diamond,cross,plus map toS1..S6. A/B have independent "
                "private permutations. Each axis maps normalized data using value**exponent then optional "
                "reversal; exponent is0.5,1 or2. Approved tick labels calibrate each axis. Overview is "
                "linear, x increases right, y increases up. Points remain in input order and connect as "
                "polylines. A closed screen-space brush selects a displayed series when its STROKE touches "
                "the rectangle, including a segment passing through with both endpoints outside. Brush "
                "replace/add/subtract applies to the persistent ID set across contexts/pages. Each gesture "
                "records the public view/data at its time, so new data does not reinterpret selection. "
                "Controls: set_view{context,page}; brush{x0,y0,x1,y1,mode} endpointsabs<=5000 and may "
                "reverse; clear_selection; clear_brush (hides rectangle only); resize{width,height}120..470; "
                "new_data changes coordinates deterministically; reorder reverses series input order."
            ),
            "interactions": interaction_contract("chart_encoding"),
        },
        {
            "family": "platform_layers",
            "title": "Continuous collision investigation",
            "description": "Repair a multi-step2D collision engine by designing your own motion experiments.",
            "prompt": common
            + (
                "state: contextA/B, platforms[{id,x,y,width,height}], bodies[{id,x,y,width,height,vx,vy}], "
                "gravity,dt,ticks. Return {'classes':{P0..P5:class},'next':{body_id:{x,y,vx,vy,grounded,"
                "platform_id}}}. Each context independently assigns exactly2solid,2one_way,2decorative "
                "platforms; all look gray. Classes must be inferred from motion. Y grows downward. "
                "Solid blocks from all sides, one_way only from above while descending, decorative never. "
                "Each tick first adds gravity*dt tovy, then moves at constant velocity fordt with continuous "
                "collision detection (including high speed). At a contact, zero the normal velocity and "
                "use the rest of the tick to slide. Multiple contacts can occur. Initial overlap is not "
                "resolved by teleporting. Ties choose lexicographically first platform ID. grounded and "
                "platform_id describe the final tick's support/contact; contact-free meansFalse/None. "
                "Controls: set_context{context}; launch{id,x,y,vx,vy,width?,height?} creates/replaces a body "
                "(idB0..B5; geometry screen bounds0..580, size8..40; velocitiesabs<=4000); "
                "remove_body{id}; set_velocity{vx,vy} changes all launch velocities and resets duration; "
                "translate_bodies{dx,dy} shifts all launches and resets duration; physics{gravity,dt} gravity-1000..1500,dt0.01..0.3; "
                "advance{ticks}1..30 adds to duration, at most40; restart resets duration to0 and keeps "
                "launches; reset_bodies restores neutral bodies; move_platform{id,x,y,width,height}; "
                "reorder reverses bodies/platforms. Advancing always replays from public launches, not "
                "private outcomes. Outlined bodies are launches, filled bodies are final outcomes."
            ),
            "interactions": interaction_contract("platform_layers"),
        },
    ]


def make_goal(family: str, goal_seed: int) -> dict:
    if family not in FAMILIES or type(goal_seed) is not int or not 0 <= goal_seed < 2**128:
        raise ValueError("Invalid private goal")
    rng = random.Random(goal_seed)
    if family == "canvas_layout":
        narrow, wide = list(CARD_IDS), list(CARD_IDS)
        rng.shuffle(narrow)
        rng.shuffle(wide)
        return {"narrow": narrow, "wide": wide, "low": rng.choice((320, 360)), "high": rng.choice((480, 520))}
    out = {}
    for context in ("A", "B"):
        if family == "chart_encoding":
            colors, markers = list(COLORS), list(MARKERS)
            rng.shuffle(colors)
            rng.shuffle(markers)
            out[context] = {
                "styles": {
                    key: {"color": colors[i], "marker": markers[i]} for i, key in enumerate(SERIES_IDS)
                },
                "x_reversed": bool(rng.randrange(2)),
                "y_reversed": bool(rng.randrange(2)),
                "x_exponent": rng.choice((0.5, 1, 2)),
                "y_exponent": rng.choice((0.5, 1, 2)),
            }
        else:
            classes = list(CLASSES) * 2
            rng.shuffle(classes)
            out[context] = {"classes": dict(zip(PLATFORM_IDS, classes))}
    return out


def _neutral_bodies(platforms):
    return [
        {"id": f"B{i}", "x": p["x"] + 18, "y": p["y"] - 90, "width": 20, "height": 24, "vx": 0, "vy": 0}
        for i, p in enumerate(sorted(platforms, key=lambda p: p["id"]))
    ]


def make_state(family: str, scene_seed: int) -> dict:
    rng = random.Random(scene_seed)
    if family == "canvas_layout":
        ids = list(CARD_IDS)
        rng.shuffle(ids)
        return {
            "ids": ids,
            "board": {"width": 420, "height": 420, "gap": 12},
            "transform": {"scale": 0.85, "pan_x": 35, "pan_y": 190},
            "pointer": {"x": 12, "y": 140},
            "gestures": [],
        }
    if family == "chart_encoding":
        ids = list(SERIES_IDS)
        rng.shuffle(ids)
        return {
            "context": "overview",
            "page": 0,
            "plot": {"left": 70, "top": 240, "width": 450, "height": 320},
            "series": [
                {
                    "id": key,
                    "points": [
                        {"x": round(rng.uniform(0.08, 0.92), 3), "y": round(rng.uniform(0.08, 0.92), 3)}
                        for _ in range(4)
                    ],
                }
                for key in ids
            ],
            "brush": None,
            "gestures": [],
        }
    if family != "platform_layers":
        raise ValueError("Unknown family")
    platforms = [
        {"id": key, "x": 35 + i * 93, "y": 375, "width": 65, "height": 14}
        for i, key in enumerate(PLATFORM_IDS)
    ]
    return {
        "context": "A",
        "platforms": platforms,
        "bodies": _neutral_bodies(platforms),
        "gravity": 0,
        "dt": 0.1,
        "ticks": 0,
    }


def _positions(state, goal):
    b = state["board"]
    width = b["width"]
    order = (
        goal["narrow"]
        if width < goal["low"]
        else (goal["wide"] if width >= goal["high"] else list("ABCDEFGHI"))
    )
    gap = b["gap"]
    w = (b["width"] - 4 * gap) / 3
    h = (b["height"] - 4 * gap) / 3
    return {key: [gap + (i % 3) * (w + gap), gap + (i // 3) * (h + gap), w, h] for i, key in enumerate(order)}


def _apply(selected, hits, mode):
    if mode == "replace":
        return sorted(dict((key, True) for key in hits))
    if mode == "add":
        return sorted(dict((key, True) for key in selected + hits))
    return [key for key in selected if key not in hits]


def _canvas_result(state, goal):
    selected = []
    for event in state["gestures"]:
        if event["kind"] == "clear_selection":
            selected = []
            continue
        positions = _positions(event, goal)
        t = event["transform"]
        if event["kind"] == "click":
            x = (event["x"] - t["pan_x"]) / t["scale"]
            y = (event["y"] - t["pan_y"]) / t["scale"]
            hits = [
                key
                for key, (left, top, w, h) in positions.items()
                if left <= x < left + w and top <= y < top + h
            ]
        else:
            x0 = (event["x0"] - t["pan_x"]) / t["scale"]
            x1 = (event["x1"] - t["pan_x"]) / t["scale"]
            y0 = (event["y0"] - t["pan_y"]) / t["scale"]
            y1 = (event["y1"] - t["pan_y"]) / t["scale"]
            hits = [
                key
                for key, (left, top, w, h) in positions.items()
                if max(min(x0, x1), left) < min(max(x0, x1), left + w)
                and max(min(y0, y1), top) < min(max(y0, y1), top + h)
            ]
        selected = _apply(selected, hits, event["mode"])
    return {"positions": _positions(state, goal), "selected_ids": sorted(selected)}


def _chart_goal(state, goal):
    if state["context"] != "overview":
        return goal[state["context"]]
    return {
        "styles": {key: {"color": list(COLORS)[i], "marker": MARKERS[i]} for i, key in enumerate(SERIES_IDS)},
        "x_reversed": False,
        "y_reversed": False,
        "x_exponent": 1,
        "y_exponent": 1,
    }


def _project(state, config):
    plot = state["plot"]
    result = {}
    for series in state["series"]:
        points = []
        for p in series["points"]:
            x = p["x"] ** config["x_exponent"]
            y = p["y"] ** config["y_exponent"]
            if config["x_reversed"]:
                x = 1 - x
            if not config["y_reversed"]:
                y = 1 - y
            points.append([plot["left"] + x * plot["width"], plot["top"] + y * plot["height"]])
        result[series["id"]] = points
    return result


def _stroke_hits(points, brush):
    left, right = sorted((brush["x0"], brush["x1"]))
    top, bottom = sorted((brush["y0"], brush["y1"]))
    if any(left <= x <= right and top <= y <= bottom for x, y in points):
        return True
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        low, high = 0.0, 1.0
        dx = x1 - x0
        dy = y1 - y0
        valid = True
        for p, q in ((-dx, x0 - left), (dx, right - x0), (-dy, y0 - top), (dy, bottom - y0)):
            if p == 0:
                if q < 0:
                    valid = False
                    break
            elif p < 0:
                low = max(low, q / p)
            else:
                high = min(high, q / p)
        if valid and low <= high:
            return True
    return False


def _chart_result(state, goal):
    selected = []
    for event in state["gestures"]:
        if event["kind"] == "clear_selection":
            selected = []
            continue
        projected = _project(event, _chart_goal(event, goal))
        visible = SERIES_IDS[event["page"] * 2 : event["page"] * 2 + 2]
        hits = [key for key in visible if _stroke_hits(projected[key], event["brush"])]
        selected = _apply(selected, hits, event["mode"])
    config = _chart_goal(state, goal)
    return {
        "styles": copy.deepcopy(config["styles"]),
        "projected_points": _project(state, config),
        "selected_ids": sorted(selected),
    }


def _sweep(body, p, dx, dy, kind):
    x, y, w, h = body["x"], body["y"], body["width"], body["height"]
    if kind == "decorative":
        return None
    if kind == "one_way":
        if dy <= 0 or y + h > p["y"] + 1e-9:
            return None
        t = (p["y"] - y - h) / dy
        if 0 <= t <= 1 and x + dx * t + w > p["x"] and x + dx * t < p["x"] + p["width"]:
            return (t, 0, -1)
        return None
    if x < p["x"] + p["width"] and x + w > p["x"] and y < p["y"] + p["height"] and y + h > p["y"]:
        return None
    entries = []
    exits = []
    for pos, size, start, span, delta in ((x, w, p["x"], p["width"], dx), (y, h, p["y"], p["height"], dy)):
        if delta == 0:
            if pos + size <= start or pos >= start + span:
                return None
            entries.append(-1e30)
            exits.append(1e30)
        elif delta > 0:
            entries.append((start - pos - size) / delta)
            exits.append((start + span - pos) / delta)
        else:
            entries.append((start + span - pos) / delta)
            exits.append((start - pos - size) / delta)
    entry = max(entries)
    exit = min(exits)
    if entry < 0 or entry > 1 or entry > exit:
        return None
    if entries[0] > entries[1]:
        return (entry, -1 if dx > 0 else 1, 0)
    return (entry, 0, -1 if dy > 0 else 1)


def _motion(state, body, classes):
    b = dict(body)
    grounded = False
    contact = None
    for _ in range(state["ticks"]):
        b["vy"] += state["gravity"] * state["dt"]
        remaining = state["dt"]
        grounded = False
        contact = None
        for _iteration in range(4):
            dx = b["vx"] * remaining
            dy = b["vy"] * remaining
            hits = []
            if dx == 0 and dy == 0:
                break
            for p in state["platforms"]:
                hit = _sweep(b, p, dx, dy, classes[p["id"]])
                if hit is not None:
                    hits.append((hit[0], p["id"], hit[1], hit[2]))
            if not hits:
                b["x"] += dx
                b["y"] += dy
                break
            t, pid, nx, ny = min(hits)
            b["x"] += dx * t
            b["y"] += dy * t
            remaining *= 1 - t
            contact = pid
            if nx:
                b["vx"] = 0
            if ny:
                b["vy"] = 0
            if ny == -1:
                grounded = True
            if remaining <= 1e-12:
                break
        supports = [
            p["id"]
            for p in state["platforms"]
            if classes[p["id"]] != "decorative"
            and abs(b["y"] + b["height"] - p["y"]) <= 1e-7
            and b["x"] + b["width"] > p["x"]
            and b["x"] < p["x"] + p["width"]
            and b["vy"] == 0
        ]
        if supports:
            grounded = True
            contact = min(supports)
    return {
        "x": b["x"],
        "y": b["y"],
        "vx": b["vx"],
        "vy": b["vy"],
        "grounded": grounded,
        "platform_id": contact,
    }


def expected_result(family, state, goal):
    if family == "canvas_layout":
        return _canvas_result(state, goal)
    if family == "chart_encoding":
        return _chart_result(state, goal)
    classes = goal[state["context"]]["classes"]
    return {
        "classes": copy.deepcopy(classes),
        "next": {body["id"]: _motion(state, body, classes) for body in state["bodies"]},
    }


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1000000


def validate_result(family, state, result):
    if not isinstance(result, dict):
        return False
    if family == "canvas_layout":
        if set(result) != {"positions", "selected_ids"} or not isinstance(result["positions"], dict):
            return False
        if set(result["positions"]) != set(state["ids"]):
            return False
        if not all(
            isinstance(v, list) and len(v) == 4 and all(_number(n) for n in v) and v[2] > 0 and v[3] > 0
            for v in result["positions"].values()
        ):
            return False
        ids = set(state["ids"])
    elif family == "chart_encoding":
        if set(result) != {"styles", "projected_points", "selected_ids"}:
            return False
        ids = {s["id"] for s in state["series"]}
        if any(
            not isinstance(result[k], dict) or set(result[k]) != ids for k in ("styles", "projected_points")
        ):
            return False
        for series in state["series"]:
            style = result["styles"][series["id"]]
            points = result["projected_points"][series["id"]]
            if (
                not isinstance(style, dict)
                or set(style) != {"color", "marker"}
                or not isinstance(style["color"], str)
                or not isinstance(style["marker"], str)
                or style["color"] not in COLORS
                or style["marker"] not in MARKERS
            ):
                return False
            if (
                not isinstance(points, list)
                or len(points) != len(series["points"])
                or not all(isinstance(p, list) and len(p) == 2 and all(_number(n) for n in p) for p in points)
            ):
                return False
    else:
        if (
            set(result) != {"classes", "next"}
            or not isinstance(result["classes"], dict)
            or not isinstance(result["next"], dict)
        ):
            return False
        ids = {p["id"] for p in state["platforms"]}
        if set(result["classes"]) != ids or not all(
            isinstance(c, str) and c in CLASSES for c in result["classes"].values()
        ):
            return False
        if set(result["next"]) != {b["id"] for b in state["bodies"]}:
            return False
        return all(
            isinstance(v, dict)
            and set(v) == {"x", "y", "vx", "vy", "grounded", "platform_id"}
            and all(_number(v[k]) for k in ("x", "y", "vx", "vy"))
            and type(v["grounded"]) is bool
            and (v["platform_id"] is None or isinstance(v["platform_id"], str) and v["platform_id"] in ids)
            for v in result["next"].values()
        )
    selected = result["selected_ids"]
    return (
        isinstance(selected, list)
        and all(isinstance(v, str) and v in ids for v in selected)
        and selected == sorted(set(selected))
    )


def _numeric_event(event, keys, lo=-5000, hi=5000):
    if not all(_number(event.get(k)) and lo <= event[k] <= hi for k in keys):
        raise ValueError("Invalid numbers")


def interact_state(family, state, event):
    if not isinstance(event, dict):
        raise ValueError("Invalid event")
    s = copy.deepcopy(state)
    kind = event.get("kind")
    if family == "canvas_layout":
        if kind == "resize":
            _numeric_event(event, ("width", "height"), 200, 600)
            s["board"].update({k: event[k] for k in ("width", "height")})
        elif kind == "pan":
            _numeric_event(event, ("dx", "dy"), -500, 500)
            s["transform"]["pan_x"] += event["dx"]
            s["transform"]["pan_y"] += event["dy"]
        elif kind == "zoom":
            _numeric_event(event, ("factor",), 0.5, 2)
            _numeric_event(event, ("x", "y"))
            t = s["transform"]
            new_scale = min(2, max(0.3, t["scale"] * event["factor"]))
            ratio = new_scale / t["scale"]
            t["pan_x"] = event["x"] - (event["x"] - t["pan_x"]) * ratio
            t["pan_y"] = event["y"] - (event["y"] - t["pan_y"]) * ratio
            t["scale"] = new_scale
        elif kind in ("click", "drag"):
            keys = ("x", "y") if kind == "click" else ("x0", "y0", "x1", "y1")
            _numeric_event(event, keys)
            if event.get("mode") not in ("replace", "add", "subtract"):
                raise ValueError("Invalid mode")
            g = {k: event[k] for k in (*keys, "mode", "kind")}
            g.update({"board": copy.deepcopy(s["board"]), "transform": copy.deepcopy(s["transform"])})
            s["gestures"].append(g)
            s["pointer"] = {"x": event[keys[-2]], "y": event[keys[-1]]}
        elif kind == "clear_selection":
            s["gestures"].append({"kind": kind})
        elif kind == "reorder":
            s["ids"].reverse()
        else:
            raise ValueError("Unknown event")
    elif family == "chart_encoding":
        if kind == "set_view":
            if (
                event.get("context") not in ("overview", "A", "B")
                or type(event.get("page")) is not int
                or not 0 <= event["page"] <= 2
            ):
                raise ValueError("Invalid view")
            s["context"], s["page"] = event["context"], event["page"]
        elif kind == "brush":
            _numeric_event(event, ("x0", "y0", "x1", "y1"))
            if event.get("mode") not in ("replace", "add", "subtract"):
                raise ValueError("Invalid mode")
            s["brush"] = {k: event[k] for k in ("x0", "y0", "x1", "y1")}
            s["gestures"].append(
                {
                    "kind": "brush",
                    "mode": event["mode"],
                    "context": s["context"],
                    "page": s["page"],
                    "plot": copy.deepcopy(s["plot"]),
                    "series": copy.deepcopy(s["series"]),
                    "brush": copy.deepcopy(s["brush"]),
                }
            )
        elif kind == "clear_selection":
            s["gestures"].append({"kind": kind})
        elif kind == "clear_brush":
            s["brush"] = None
        elif kind == "resize":
            _numeric_event(event, ("width", "height"), 120, 470)
            s["plot"].update({k: event[k] for k in ("width", "height")})
        elif kind == "new_data":
            for series in s["series"]:
                for p in series["points"]:
                    p["x"], p["y"] = round((p["x"] + 0.237) % 1, 3), round((p["y"] + 0.419) % 1, 3)
        elif kind == "reorder":
            s["series"].reverse()
        else:
            raise ValueError("Unknown event")
    else:
        if kind == "set_context":
            if event.get("context") not in ("A", "B"):
                raise ValueError("Invalid context")
            s["context"] = event["context"]
        elif kind == "launch":
            if event.get("id") not in tuple(f"B{i}" for i in range(6)):
                raise ValueError("Invalid body")
            _numeric_event(event, ("x", "y"), 0, 580)
            _numeric_event(event, ("vx", "vy"), -4000, 4000)
            body = {
                "id": event["id"],
                **{k: event[k] for k in ("x", "y", "vx", "vy")},
                "width": event.get("width", 20),
                "height": event.get("height", 24),
            }
            _numeric_event(body, ("width", "height"), 8, 40)
            s["bodies"] = [b for b in s["bodies"] if b["id"] != body["id"]] + [body]
            s["ticks"] = 0
        elif kind == "set_velocity":
            _numeric_event(event, ("vx", "vy"), -4000, 4000)
            for body in s["bodies"]:
                body["vx"], body["vy"] = event["vx"], event["vy"]
            s["ticks"] = 0
        elif kind == "translate_bodies":
            _numeric_event(event, ("dx", "dy"), -500, 500)
            for body in s["bodies"]:
                body["x"] += event["dx"]
                body["y"] += event["dy"]
            s["ticks"] = 0
        elif kind == "remove_body":
            if event.get("id") not in tuple(f"B{i}" for i in range(6)):
                raise ValueError("Invalid body")
            s["bodies"] = [b for b in s["bodies"] if b["id"] != event["id"]]
        elif kind == "physics":
            _numeric_event(event, ("gravity",), -1000, 1500)
            _numeric_event(event, ("dt",), 0.01, 0.3)
            s["gravity"], s["dt"] = event["gravity"], event["dt"]
        elif kind == "advance":
            if (
                type(event.get("ticks")) is not int
                or not 1 <= event["ticks"] <= 30
                or s["ticks"] + event["ticks"] > 40
            ):
                raise ValueError("Invalid duration")
            s["ticks"] += event["ticks"]
        elif kind == "restart":
            s["ticks"] = 0
        elif kind == "reset_bodies":
            s["bodies"] = _neutral_bodies(s["platforms"])
            s["ticks"] = 0
        elif kind == "move_platform":
            if event.get("id") not in PLATFORM_IDS:
                raise ValueError("Invalid platform")
            _numeric_event(event, ("x", "y"), 0, 580)
            _numeric_event(event, ("width", "height"), 8, 180)
            for p in s["platforms"]:
                if p["id"] == event["id"]:
                    p.update({k: event[k] for k in ("x", "y", "width", "height")})
        elif kind == "reorder":
            s["platforms"].reverse()
            s["bodies"].reverse()
        else:
            raise ValueError("Unknown event")
    if len(s.get("gestures", [])) > 64:
        raise ValueError("Gesture history full")
    return s


def evidence_view(family, state):
    if family == "canvas_layout":
        return f"width:{state['board']['width']}:gestures:{len(state['gestures'])}"
    if family == "chart_encoding":
        return f"{state['context']}:page{state['page']}:gestures:{len(state['gestures'])}"
    return f"{state['context']}:ticks{state['ticks']}:bodies{len(state['bodies'])}"


def development_actions(family):
    if family == "canvas_layout":
        events = [{"kind": "resize", "width": w, "height": 420} for w in (260, 340, 500, 580)]
    elif family == "chart_encoding":
        events = [{"kind": "set_view", "context": c, "page": p} for c in ("A", "B") for p in range(3)]
    else:
        # Ordinary user-designed simultaneous launch experiments, never policy inputs.
        events = []
        for context in ("A", "B"):
            events.append({"kind": "set_context", "context": context})
            for vy, y in ((1000, 290), (-1000, 440)):
                events.extend(
                    {"kind": "launch", "id": f"B{i}", "x": 53 + i * 93, "y": y, "vx": 0, "vy": vy}
                    for i in range(6)
                )
                events.append({"kind": "advance", "ticks": 1})
    return [{"type": "interact", "event": e} for e in events]


def _sequence(family, seed, events):
    state = make_state(family, seed)
    for event in events:
        state = interact_state(family, state, event)
    return state


def canonical_states(family, scene_seed):
    states = []
    if family == "canvas_layout":
        for index, width in enumerate(
            (260, 319, 320, 339, 340, 359, 360, 420, 479, 480, 499, 500, 519, 520, 580)
        ):
            state = make_state(family, scene_seed + index)
            state["board"].update(width=width, height=330 + index * 9, gap=6 + index % 7)
            state["transform"] = {
                "scale": 0.65 + index * 0.03,
                "pan_x": -30 + index * 5,
                "pan_y": 90 - index * 3,
            }
            for event in (
                {"kind": "drag", "x0": 390, "y0": 410, "x1": 10, "y1": 110, "mode": "replace"},
                {"kind": "resize", "width": 580 if width < 420 else 260, "height": 420},
                {"kind": "zoom", "factor": 1.3, "x": 200, "y": 250},
                {"kind": "click", "x": 150, "y": 240, "mode": "add"},
                {"kind": "drag", "x0": 60, "y0": 230, "x1": 230, "y1": 400, "mode": "subtract"},
            ):
                state = interact_state(family, state, event)
                if event["kind"] in ("drag", "click"):
                    states.append(copy.deepcopy(state))
        return states
    if family == "chart_encoding":
        for c in ("A", "B", "overview"):
            for page in range(3):
                state = make_state(family, scene_seed + page * 7919)
                state["context"], state["page"] = c, page
                # Deliberate line-crossing case: no endpoint lies in central brush.
                for series in state["series"]:
                    series["points"] = [
                        {"x": 0, "y": 0},
                        {"x": 1, "y": 1},
                        {"x": 0, "y": 1},
                        {"x": 1, "y": 0},
                    ]
                events = [
                    {"kind": "brush", "x0": 340, "y0": 440, "x1": 240, "y1": 350, "mode": "replace"},
                    {"kind": "set_view", "context": "B" if c != "B" else "A", "page": (page + 1) % 3},
                    {"kind": "resize", "width": 320, "height": 240},
                    {"kind": "brush", "x0": 70, "y0": 240, "x1": 390, "y1": 480, "mode": "add"},
                    {"kind": "new_data"},
                    {"kind": "brush", "x0": 160, "y0": 320, "x1": 300, "y1": 400, "mode": "subtract"},
                ]
                for event in events:
                    state = interact_state(family, state, event)
                    if event["kind"] == "brush":
                        states.append(copy.deepcopy(state))
        return states
    for context in ("A", "B"):
        for mode in ("down", "up", "fast_down", "side", "gravity", "slide", "overlap", "tie"):
            s = make_state(family, scene_seed)
            s["context"] = context
            s["ticks"] = 1
            for i, b in enumerate(s["bodies"]):
                p = s["platforms"][i]
                if mode == "down":
                    b.update(y=290, vy=1000)
                elif mode == "up":
                    b.update(y=440, vy=-1000)
                elif mode == "fast_down":
                    b.update(y=220, vy=4000)
                elif mode == "side":
                    b.update(x=p["x"] - 28, y=p["y"] - 3, vx=800, vy=0)
                elif mode == "gravity":
                    b.update(y=280, vy=20)
                    s.update(gravity=1000, dt=0.15, ticks=4)
                elif mode == "slide":
                    b.update(y=280, vx=120, vy=1000)
                    s.update(gravity=300, dt=0.12, ticks=3)
                elif mode == "overlap":
                    b.update(y=370, vy=500, vx=100)
                else:
                    b.update(x=110, y=160, vy=4000)
            if mode == "tie":
                for p in s["platforms"]:
                    p.update(x=100, y=300)
            s["platforms"].reverse()
            s["bodies"].reverse()
            states.append(s)
        # Two contacts in one tick: top support then side wall (roles are goal-dependent).
        s = make_state(family, scene_seed)
        s["context"] = context
        s.update(ticks=1, dt=0.3, gravity=0)
        s["bodies"] = [{"id": "B0", "x": 80, "y": 210, "width": 20, "height": 24, "vx": 900, "vy": 600}]
        for i, p in enumerate(s["platforms"]):
            p.update(x=100 + i * 45, y=300 if i % 2 == 0 else 240, width=40, height=100 if i % 2 else 14)
        states.append(s)
    return states


def _close(actual, expected):
    if type(expected) in (bool, str) or expected is None:
        return actual == expected and type(actual) is type(expected)
    if isinstance(expected, (int, float)):
        return _number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_close(a, b) for a, b in zip(actual, expected))
        )
    return (
        isinstance(actual, dict)
        and set(actual) == set(expected)
        and all(_close(actual[k], v) for k, v in expected.items())
    )


def evaluate_program(family, source, scene_seed, goal):
    spec_key = {"canvas_layout": "positions", "chart_encoding": "styles", "platform_layers": "classes"}[
        family
    ]
    behavior_keys = {
        "canvas_layout": ("selected_ids",),
        "chart_encoding": ("projected_points", "selected_ids"),
        "platform_layers": ("next",),
    }[family]
    cases = []
    spec_passed = behavior_passed = component_passed = 0
    states = canonical_states(family, scene_seed)
    for index, state in enumerate(states):
        runtime = run_program(source, state)
        valid = runtime["ok"] and validate_result(family, state, runtime["result"])
        spec = behavior = False
        if valid:
            actual, expected = runtime["result"], expected_result(family, state, goal)
            spec = _close(actual[spec_key], expected[spec_key])
            behavior = all(_close(actual[k], expected[k]) for k in behavior_keys)
            if family == "chart_encoding":
                component_passed += sum(
                    actual["styles"][k][f] == expected["styles"][k][f]
                    for k in SERIES_IDS
                    for f in ("color", "marker")
                )
            else:
                component_passed += sum(_close(v, expected[spec_key][k]) for k, v in actual[spec_key].items())
        spec_passed += int(spec)
        behavior_passed += int(behavior)
        cases.append(
            {
                "case": index,
                "schema_valid": bool(valid),
                "spec_passed": bool(spec),
                "behavior_passed": bool(behavior),
            }
        )
    passed = sum(r["spec_passed"] and r["behavior_passed"] for r in cases)
    count, unit = {
        "canvas_layout": (9, "card_rectangle"),
        "chart_encoding": (12, "series_color_or_marker"),
        "platform_layers": (6, "platform_class"),
    }[family]
    return {
        "passed": passed,
        "total": len(states),
        "spec_passed": spec_passed,
        "spec_total": len(states),
        "behavior_passed": behavior_passed,
        "behavior_total": len(states),
        "component_passed": component_passed,
        "component_total": count * len(states),
        "component_unit": unit,
        "success": passed == len(states),
        "case_diagnostics": cases,
    }


def oracle_source(family, goal):
    """Offline oracle for validation only. This code is never a policy input."""
    names = {
        "canvas_layout": ("_positions", "_apply", "_canvas_result"),
        "chart_encoding": ("_apply", "_chart_goal", "_project", "_stroke_hits", "_chart_result"),
        "platform_layers": ("_sweep", "_motion"),
    }[family]
    header = f"COLORS = {COLORS!r}\nMARKERS = {MARKERS!r}\nSERIES_IDS = {SERIES_IDS!r}\n"
    # Helpers use no module access except deep copying plain styles: JSON-safe comprehension instead.
    body = "\n\n".join(inspect.getsource(globals()[name]) for name in names)
    body = body.replace('copy.deepcopy(config["styles"])', '{k:dict(v) for k,v in config["styles"].items()}')
    if family == "canvas_layout":
        finish = f"def solve(state):\n    return _canvas_result(state,{goal!r})\n"
    elif family == "chart_encoding":
        finish = f"def solve(state):\n    return _chart_result(state,{goal!r})\n"
    else:
        finish = f'def solve(state):\n    classes={goal!r}[state["context"]]["classes"]\n    return {{"classes":classes,"next":{{b["id"]:_motion(state,b,classes) for b in state["bodies"]}}}}\n'
    return header + body + "\n\n" + finish


def starter_code(family):
    if family == "canvas_layout":
        return """def solve(state):
    b=state["board"];gap=b["gap"]
    w=(b["width"]-4*gap)/3;h=(b["height"]-4*gap)/3
    positions={key:[gap+(i%3)*(w+gap),gap+(i//3)*(h+gap),w,h] for i,key in enumerate(sorted(state["ids"]))}
    selected=[]
    for e in state["gestures"]:
        if e["kind"]=="click":
            for key,(x,y,w,h) in positions.items():
                if x<=e["x"]<=x+w and y<=e["y"]<=y+h:selected=[key]
    return {"positions":positions,"selected_ids":selected}
"""
    if family == "chart_encoding":
        return """def solve(state):
    colors=["blue","orange","green","red","purple","brown"]
    markers=["circle","square","triangle","diamond","cross","plus"]
    ids=["S1","S2","S3","S4","S5","S6"]
    p=state["plot"];projected={};styles={key:{"color":colors[i],"marker":markers[i]} for i,key in enumerate(ids)}
    for s in state["series"]:
        projected[s["id"]]=[[p["left"]+v["x"]*p["width"],p["top"]+(1-v["y"])*p["height"]] for v in s["points"]]
    selected=[];brush=state["brush"]
    if brush:
        for key in ids[state["page"]*2:state["page"]*2+2]:
            if any(brush["x0"]<=x<=brush["x1"] and brush["y0"]<=y<=brush["y1"] for x,y in projected[key]):selected.append(key)
    return {"styles":styles,"projected_points":projected,"selected_ids":selected}
"""
    return """def solve(state):
    classes={p["id"]:"decorative" for p in state["platforms"]};out={}
    for b in state["bodies"]:
        x,y,vx,vy=b["x"],b["y"],b["vx"],b["vy"];grounded=False;pid=None
        for t in range(state["ticks"]):
            vy+=state["gravity"]*state["dt"];x+=vx*state["dt"];y+=vy*state["dt"]
            for p in state["platforms"]:
                if classes[p["id"]]!="decorative" and p["x"]<=x<=p["x"]+p["width"] and p["y"]<=y+b["height"]<=p["y"]+p["height"]:
                    y=p["y"]-b["height"];vy=0;grounded=True;pid=p["id"]
        out[b["id"]]={"x":x,"y":y,"vx":vx,"vy":vy,"grounded":grounded,"platform_id":pid}
    return {"classes":classes,"next":out}
"""


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _marker(draw, x, y, marker, color, size=8):
    if marker == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=color)
    elif marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=color)
    elif marker == "triangle":
        draw.polygon([(x, y - size - 2), (x - size - 1, y + size), (x + size + 1, y + size)], fill=color)
    elif marker == "diamond":
        draw.polygon([(x, y - size - 2), (x - size - 2, y), (x, y + size + 2), (x + size + 2, y)], fill=color)
    elif marker == "cross":
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=4)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=4)
    else:
        draw.line((x - size, y, x + size, y), fill=color, width=4)
        draw.line((x, y - size, x, y + size), fill=color, width=4)


def _panel(family, state, output, reference=False, goal=None):
    image = Image.new("RGB", (PANEL_WIDTH, HEIGHT), "#f5f7fb")
    d = ImageDraw.Draw(image)
    d.text(
        (24, 20), "Approved reference" if reference else "Current candidate", fill="#15223b", font=_font(24)
    )
    d.text(
        (24, 58),
        dict((t["family"], t["title"]) for t in task_catalog())[family],
        fill="#53617a",
        font=_font(17),
    )
    valid = validate_result(family, state, output)
    if family == "canvas_layout":
        b, t = state["board"], state["transform"]
        d.text(
            (24, 100),
            f"Viewport {b['width']} x {b['height']} | {len(state['gestures'])} gestures",
            fill="#53617a",
            font=_font(17),
        )
        d.rectangle(
            (
                t["pan_x"],
                t["pan_y"],
                t["pan_x"] + b["width"] * t["scale"],
                t["pan_y"] + b["height"] * t["scale"],
            ),
            fill="#e6ebf4",
            outline="#bcc7d9",
        )
        if valid:
            for key, (px, py, pw, ph) in output["positions"].items():
                x, y, w, h = (
                    px * t["scale"] + t["pan_x"],
                    py * t["scale"] + t["pan_y"],
                    pw * t["scale"],
                    ph * t["scale"],
                )
                chosen = key in output["selected_ids"]
                d.rectangle(
                    (x, y, x + w, y + h),
                    fill="#d7e7ff" if chosen else "white",
                    outline="#2563eb" if chosen else "#7c8da7",
                    width=4 if chosen else 2,
                )
                d.text((x + w / 2, y + h / 2), key, anchor="mm", fill="#15223b", font=_font(27))
            d.rectangle((0, 632, 600, 700), fill="#f5f7fb")
            d.text(
                (24, 650),
                "Selected: " + (", ".join(output["selected_ids"]) or "none"),
                fill="#15223b",
                font=_font(18),
            )
        p = state["pointer"]
        d.line((p["x"] - 8, p["y"], p["x"] + 8, p["y"]), fill="#dd3348", width=2)
        d.line((p["x"], p["y"] - 8, p["x"], p["y"] + 8), fill="#dd3348", width=2)
    elif family == "chart_encoding":
        visible = SERIES_IDS[state["page"] * 2 : state["page"] * 2 + 2]
        p = state["plot"]
        left, top, w, h = (p[k] for k in ("left", "top", "width", "height"))
        d.text(
            (24, 100),
            f"Context {state['context']} | Page {state['page']} | {', '.join(visible)}",
            fill="#53617a",
            font=_font(18),
        )
        for frac in (0, 0.25, 0.5, 0.75, 1):
            d.line((left + frac * w, top, left + frac * w, top + h), fill="#dfe5ef")
            d.line((left, top + frac * h, left + w, top + frac * h), fill="#dfe5ef")
        d.rectangle((left, top, left + w, top + h), outline="#7c8da7", width=2)
        if reference:
            c = _chart_goal(state, goal)
            for value in (0, 0.25, 0.5, 1):
                x = value ** c["x_exponent"]
                y = value ** c["y_exponent"]
                if c["x_reversed"]:
                    x = 1 - x
                if not c["y_reversed"]:
                    y = 1 - y
                d.text((left + x * w, top + h + 12), str(value), anchor="mt", fill="#15223b", font=_font(15))
                d.text((left - 8, top + y * h), str(value), anchor="rm", fill="#15223b", font=_font(15))
        if state["brush"]:
            b = state["brush"]
            d.rectangle(
                (min(b["x0"], b["x1"]), min(b["y0"], b["y1"]), max(b["x0"], b["x1"]), max(b["y0"], b["y1"])),
                outline="#d33683",
                width=3,
            )
        if valid:
            for i, key in enumerate(visible):
                style = output["styles"][key]
                color = COLORS[style["color"]]
                _marker(d, 75 + i * 250, 160, style["marker"], color, 11)
                d.text((100 + i * 250, 148), key, fill="#15223b", font=_font(22))
                points = output["projected_points"][key]
                if len(points) > 1:
                    d.line(
                        [tuple(point) for point in points],
                        fill=color,
                        width=5 if key in output["selected_ids"] else 2,
                    )
                for x, y in points:
                    _marker(d, x, y, style["marker"], color)
                    if key in output["selected_ids"]:
                        d.ellipse((x - 13, y - 13, x + 13, y + 13), outline="#15223b", width=2)
            d.text(
                (24, 650),
                "Selected: " + (", ".join(output["selected_ids"]) or "none"),
                fill="#15223b",
                font=_font(17),
            )
    else:
        d.text(
            (24, 100),
            f"Context {state['context']} | {state['ticks']} ticks | dt {state['dt']}",
            fill="#53617a",
            font=_font(18),
        )
        d.text((24, 135), "Gray surfaces hide their collision material", fill="#53617a", font=_font(17))
        for p in state["platforms"]:
            d.rectangle((p["x"], p["y"], p["x"] + p["width"], p["y"] + p["height"]), fill="#7c8da7")
            d.text(
                (p["x"] + p["width"] / 2, p["y"] + p["height"] + 7),
                p["id"],
                anchor="mt",
                fill="#15223b",
                font=_font(15),
            )
        if valid:
            for b in state["bodies"]:
                x, y, w, h = (b[k] for k in ("x", "y", "width", "height"))
                d.rectangle((x, y, x + w, y + h), outline="#e68a1b", width=3)
                out = output["next"][b["id"]]
                nx, ny = out["x"], out["y"]
                # Clipping is visualization only; numeric outcomes are not returned to the policy.
                d.line((x + w / 2, y + h / 2, nx + w / 2, ny + h / 2), fill="#e9ba75", width=2)
                d.rectangle((nx, ny, nx + w, ny + h), fill="#e68a1b")
                d.text((nx + w / 2, ny - 7), b["id"], anchor="mb", fill="#7c4810", font=_font(14))
        d.rectangle((0, 630, 600, 700), fill="#f5f7fb")
        d.text((24, 650), "Outline: launch     Filled: final position", fill="#53617a", font=_font(17))
    if not valid:
        d.rounded_rectangle((70, 210, 530, 250), radius=7, fill="#fff3e1", outline="#b88434")
        d.text((300, 230), "No schema-valid output", anchor="mm", fill="#825817", font=_font(18))
    return np.asarray(image)


def render_reference(family, state, goal):
    return _panel(family, state, expected_result(family, state, goal), True, goal)


def render_current(family, state, output):
    return _panel(family, state, output)
