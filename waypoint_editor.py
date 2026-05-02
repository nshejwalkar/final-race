#!/usr/bin/env python3
"""
F1TENTH Interactive Waypoint Editor
====================================
Overlays waypoints from a CSV onto a PGM map image.
Two editing modes:
  - Points Mode:      drag individual waypoints, bulk edit speed/lookahead
  - Catmull-Rom Mode: fit a C-R spline through a SELECTED PORTION of the
                      raceline (highlight points first, then click Catmull-Rom).
                      Drag CPs to reshape, slide CP count to simplify.
                      Switch back to Points to commit the spline output to the
                      raceline (only the highlighted segment is replaced).

Usage:
    python waypoint_editor.py                          # defaults: pure_pursuit/waypoints/race2.csv + f1tenth_gym_ros/maps/my_map1.yaml
    python waypoint_editor.py --csv pure_pursuit/waypoints/race2.csv --map f1tenth_gym_ros/maps/my_map1.yaml
    python waypoint_editor.py --csv /abs/path/to/waypoints.csv --map /abs/path/to/map.yaml
"""

import argparse
import csv
import io
import json
import os
import struct
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# PGM → PNG conversion
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import base64


def parse_pgm(pgm_path: str):
    """Parse a P5 PGM file. Returns (width, height, maxval, pixel_bytes)."""
    with open(pgm_path, "rb") as f:
        magic = f.readline().decode().strip()
        assert magic == "P5", f"Only P5 (binary) PGM supported, got {magic}"
        tokens = []
        while len(tokens) < 3:
            line = f.readline().decode().strip()
            if line.startswith("#"):
                continue
            tokens.extend(line.split())
        width, height, maxval = int(tokens[0]), int(tokens[1]), int(tokens[2])
        data = f.read()
    return width, height, maxval, data


def _is_png(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == b"\x89PNG\r\n\x1a\n"


def parse_png_dimensions(path: str):
    """Read width/height from a PNG IHDR chunk without PIL."""
    with open(path, "rb") as f:
        f.read(8)   # PNG signature
        f.read(4)   # IHDR length
        f.read(4)   # 'IHDR'
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    return w, h


def load_pgm_as_png_base64(pgm_path: str) -> str:
    if HAS_PIL:
        img = Image.open(pgm_path).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    else:
        if _is_png(pgm_path):
            with open(pgm_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        width, height, maxval, data = parse_pgm(pgm_path)
        import zlib

        def make_png(w, h, gray_data, mv):
            raw_rows = b""
            for y in range(h):
                raw_rows += b"\x00"
                row = gray_data[y * w : (y + 1) * w]
                if mv != 255:
                    row = bytes(int(b / mv * 255) for b in row)
                raw_rows += row

            def chunk(ctype, cdata):
                c = ctype + cdata
                crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
                return struct.pack(">I", len(cdata)) + c + crc

            ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(raw_rows))
                + chunk(b"IEND", b"")
            )

        return base64.b64encode(make_png(width, height, data, maxval)).decode()


def load_yaml_simple(yaml_path: str) -> dict:
    result = {}
    with open(yaml_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    result[key.strip()] = json.loads(val)
                else:
                    try:
                        result[key.strip()] = float(val)
                    except ValueError:
                        result[key.strip()] = val
    return result


def load_csv_waypoints(csv_path: str) -> list:
    waypoints = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            vals = []
            for token in row:
                try:
                    vals.append(float(token))
                except ValueError:
                    continue
            if len(vals) >= 2:
                x = vals[0]
                y = vals[1]
                speed = vals[2] if len(vals) >= 3 else 1.0
                lookahead = vals[3] if len(vals) >= 4 else 0.85
                waypoints.append([x, y, speed, lookahead])
    return waypoints


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
MAP_PNG_B64 = ""
MAP_META = {}
WAYPOINTS = []
CSV_PATH = ""
MAP_YAML_PATH = ""


# ---------------------------------------------------------------------------
# HTML / JS frontend
# ---------------------------------------------------------------------------
def build_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>F1TENTH Waypoint Editor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
#topbar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 16px; background: #16213e;
    border-bottom: 1px solid #0f3460; flex-shrink: 0; z-index: 100;
    flex-wrap: wrap;
}
#topbar h1 { font-size: 16px; color: #e94560; white-space: nowrap; }
#topbar .info { font-size: 12px; color: #888; }
#topbar button {
    padding: 6px 14px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 13px; font-weight: 600;
}
.btn-save { background: #e94560; color: #fff; }
.btn-save:hover { background: #c73650; }
.btn-secondary { background: #0f3460; color: #e0e0e0; }
.btn-secondary:hover { background: #1a4a8a; }
.btn-undo { background: #533483; color: #e0e0e0; }
.btn-undo:hover { background: #6b44a0; }
.btn-mode { background: #0a8754; color: #fff; }
.btn-mode:hover { background: #0b9e63; }
.btn-mode.active { background: #e94560; }

#main { display: flex; flex: 1; overflow: hidden; }
#canvas-wrap { flex: 1; position: relative; overflow: hidden; cursor: crosshair; }
canvas { position: absolute; top: 0; left: 0; }

#panel {
    width: 300px; background: #16213e; border-left: 1px solid #0f3460;
    display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto;
}
#panel h2 {
    font-size: 14px; padding: 12px 14px 8px; color: #e94560;
    border-bottom: 1px solid #0f3460;
}
#panel-content { padding: 10px 14px; }
.field { margin-bottom: 10px; }
.field label {
    display: block; font-size: 11px; color: #888; margin-bottom: 3px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.field input, .field select {
    width: 100%; padding: 6px 8px; background: #1a1a2e;
    border: 1px solid #0f3460; border-radius: 4px; color: #e0e0e0; font-size: 13px;
}
.field input:focus { border-color: #e94560; outline: none; }
.field input[type=range] { padding: 0; }

.wp-list {
    max-height: 250px; overflow-y: auto; border: 1px solid #0f3460;
    border-radius: 4px; margin-top: 6px;
}
.wp-item {
    padding: 4px 8px; font-size: 11px; font-family: monospace;
    border-bottom: 1px solid #0f3460; cursor: pointer;
    display: flex; justify-content: space-between;
}
.wp-item:hover { background: #0f3460; }
.wp-item.selected { background: #533483; color: #fff; }

.section-box {
    border-top: 1px solid #0f3460; padding: 10px 14px; margin-top: 8px;
}
.multiplier-block {
    margin-top: 12px; padding-top: 10px; border-top: 1px dashed #0f3460;
}
.multiplier-block .field-label {
    display: block; font-size: 11px; color: #888; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.section-box h3 { font-size: 12px; color: #e94560; margin-bottom: 8px; }
.section-box .field { margin-bottom: 8px; }
.section-box button { width: 100%; padding: 8px; margin-top: 4px; }

#statusbar {
    padding: 4px 16px; background: #0f3460; font-size: 11px; color: #888;
    display: flex; gap: 20px; flex-shrink: 0;
}
#tooltip {
    position: fixed; background: rgba(22,33,62,0.95); border: 1px solid #e94560;
    border-radius: 4px; padding: 6px 10px; font-size: 11px;
    pointer-events: none; display: none; z-index: 1000; white-space: pre;
}
#legend { padding: 8px 14px; border-top: 1px solid #0f3460; flex-shrink: 0; }
#legend h3 { font-size: 12px; color: #e94560; margin-bottom: 6px; }
#legend-bar {
    height: 14px; border-radius: 3px;
    background: linear-gradient(to right, #00ff88, #ffff00, #ff4444);
}
#legend-labels {
    display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-top: 2px;
}
.range-row { display: flex; align-items: center; gap: 8px; }
.range-row input[type=range] { flex: 1; }
.range-row span { font-size: 12px; min-width: 30px; text-align: right; color: #e0e0e0; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a2e; }
::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 3px; }
</style>
</head>
<body>

<div id="topbar">
    <h1>F1TENTH Waypoint Editor</h1>
    <span class="info" id="file-info"></span>
    <div style="flex:1"></div>
    <button class="btn-mode" id="btn-points" onclick="setMode('points')">Points</button>
    <button class="btn-mode" id="btn-spline" onclick="setMode('spline')">Catmull-Rom</button>
    <span style="width:1px;height:24px;background:#0f3460"></span>
    <button class="btn-secondary" onclick="resetView()">Reset View</button>
    <button class="btn-undo" onclick="undo()">Undo (Ctrl+Z)</button>
    <button class="btn-undo" onclick="redo()">Redo (Ctrl+Y)</button>
    <button class="btn-secondary" onclick="saveAs()">Save As...</button>
    <button class="btn-save" onclick="save()">Save (Ctrl+S)</button>
</div>

<div id="main">
    <div id="canvas-wrap"><canvas id="c"></canvas></div>
    <div id="panel">
        <!-- ====== POINTS MODE PANEL ====== -->
        <div id="points-panel">
            <h2>Selection <span id="sel-count" style="color:#888;font-weight:normal"></span></h2>
            <div id="panel-content">
                <div id="no-sel" style="color:#666;font-size:12px;padding:10px 0;">
                    Click a waypoint to select it.<br>
                    Drag a rectangle to select multiple.<br>
                    Hold <b>Shift</b> to add to selection.<br>
                    Hold <b>Ctrl</b> to toggle individual points.
                </div>
                <div id="single-sel" style="display:none">
                    <div class="field"><label>Index</label><input type="text" id="sel-idx" readonly></div>
                    <div class="field"><label>X (meters)</label><input type="number" step="0.001" id="sel-x" onchange="updateSingleField('x')"></div>
                    <div class="field"><label>Y (meters)</label><input type="number" step="0.001" id="sel-y" onchange="updateSingleField('y')"></div>
                    <div class="field"><label>Speed (m/s)</label><input type="number" step="0.05" min="0" id="sel-speed" onchange="updateSingleField('speed')"></div>
                    <div class="field"><label>Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="sel-la" onchange="updateSingleField('lookahead')"></div>
                </div>
                <div id="multi-sel" style="display:none"><div class="wp-list" id="sel-list"></div></div>
                <div class="section-box">
                    <h3>Bulk Edit Selected</h3>
                    <div class="field"><label>Set Speed (m/s)</label><input type="number" step="0.05" min="0" id="bulk-speed" placeholder="leave blank to keep"></div>
                    <div class="field"><label>Set Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="bulk-la" placeholder="leave blank to keep"></div>
                    <button class="btn-save" onclick="applyBulk()">Apply to Selected</button>
                    <div class="multiplier-block">
                        <span class="field-label">Multiply Selected By</span>
                        <div class="field"><label>Speed Multiplier</label><input type="number" step="0.01" min="0" id="bulk-speed-mul" placeholder="e.g. 0.9 (leave blank for no change)"></div>
                        <div class="field"><label>Lookahead Multiplier</label><input type="number" step="0.01" min="0" id="bulk-la-mul" placeholder="e.g. 1.1 (leave blank for no change)"></div>
                        <button class="btn-secondary" onclick="applyBulkMultiplier()">Apply Multipliers</button>
                    </div>
                    <button class="btn-secondary" style="margin-top:8px" onclick="deleteSelected()">Delete Selected</button>
                </div>
            </div>
        </div>

        <!-- ====== SPLINE MODE PANEL ====== -->
        <div id="spline-panel" style="display:none">
            <h2>Spline Controls</h2>
            <div id="spline-panel-content" style="padding:10px 14px;">
                <div class="field">
                    <label>Control Points</label>
                    <div class="range-row">
                        <input type="range" id="cp-count" min="6" max="100" value="25" oninput="onCpCountChange()">
                        <span id="cp-count-val">25</span>
                    </div>
                </div>
                <hr style="border-color:#0f3460;margin:12px 0">
                <div id="cp-no-sel" style="color:#666;font-size:12px;padding:6px 0;">
                    Click a control point to select it.<br>
                    Drag to reshape the path.<br>
                    Hold <b>Shift+click</b> to select multiple.<br>
                    <b>Double-click</b> empty space to add a CP.<br>
                    <b>Delete</b> key to remove selected CPs.
                </div>
                <div id="cp-single-sel" style="display:none">
                    <div class="field"><label>Control Point Index</label><input type="text" id="cp-idx" readonly></div>
                    <div class="field"><label>X (meters)</label><input type="number" step="0.001" id="cp-x" onchange="updateCpField('x')"></div>
                    <div class="field"><label>Y (meters)</label><input type="number" step="0.001" id="cp-y" onchange="updateCpField('y')"></div>
                    <div class="field"><label>Speed (m/s)</label><input type="number" step="0.05" min="0" id="cp-speed" onchange="updateCpField('speed')"></div>
                    <div class="field"><label>Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="cp-la" onchange="updateCpField('lookahead')"></div>
                </div>
                <div id="cp-multi-sel" style="display:none">
                    <div class="wp-list" id="cp-list"></div>
                </div>
                <div class="section-box">
                    <h3>Bulk Edit Control Points</h3>
                    <div class="field"><label>Set Speed (m/s)</label><input type="number" step="0.05" min="0" id="cp-bulk-speed" placeholder="leave blank to keep"></div>
                    <div class="field"><label>Set Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="cp-bulk-la" placeholder="leave blank to keep"></div>
                    <button class="btn-save" onclick="applyCpBulk()">Apply to Selected</button>
                    <div class="multiplier-block">
                        <span class="field-label">Multiply Selected By</span>
                        <div class="field"><label>Speed Multiplier</label><input type="number" step="0.01" min="0" id="cp-bulk-speed-mul" placeholder="e.g. 0.9 (leave blank for no change)"></div>
                        <div class="field"><label>Lookahead Multiplier</label><input type="number" step="0.01" min="0" id="cp-bulk-la-mul" placeholder="e.g. 1.1 (leave blank for no change)"></div>
                        <button class="btn-secondary" onclick="applyCpMultiplier()">Apply Multipliers</button>
                    </div>
                    <button class="btn-secondary" style="margin-top:8px" onclick="deleteCpSelected()">Delete Selected CPs</button>
                    <button class="btn-mode" style="margin-top:8px" onclick="commitAndGoPoints()">Commit & Edit Points</button>
                </div>
            </div>
        </div>

        <div id="legend">
            <h3>Speed Color</h3>
            <div id="legend-bar"></div>
            <div id="legend-labels"><span id="leg-min">0</span><span id="leg-max">3</span></div>
        </div>
    </div>
</div>

<div id="statusbar">
    <span id="status-pos">Mouse: -</span>
    <span id="status-zoom">Zoom: 100%</span>
    <span id="status-wp">Waypoints: 0</span>
    <span id="status-mode">Mode: Points</span>
    <span id="status-dirty"></span>
</div>

<div id="tooltip"></div>

<script>
// =====================================================================
//  DATA
// =====================================================================
let mapImg = new Image();
let mapMeta = {};
let waypoints = [];
let dirty = false;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W = 0, H = 0;
let zoom = 1, panX = 0, panY = 0;

// Interaction (points mode)
let selected = new Set();
let dragStart = null;
let dragType = null;
let rectStart = null;
let hoveredIdx = -1;
let dragPointOffsets = [];
let lastMouseWorld = {x:0, y:0};
let lastMouseScreen = {x: 0, y: 0};

// Undo / Redo
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 50;

// =====================================================================
//  MODE: 'points' | 'spline'
// =====================================================================
let mode = 'points';

// Spline state
// splineRange = {start, end} indices into waypoints[] being edited
//   (inclusive both ends). null when not in spline mode.
let splineRange = null;
let splineOriginalSegment = [];   // deep copy of waypoints[start..end] at mode-entry
let splinePhantomBefore = null;   // waypoint just before start (for tangent at start)
let splinePhantomAfter = null;    // waypoint just after end   (for tangent at end)
let controlPoints = [];
let splineCurve = [];
let splineWaypoints = [];
let cpSelected = new Set();
let cpHoveredIdx = -1;

function setMode(m) {
    // ---- Spline mode requires a selection of at least 2 waypoints ----
    if (m === 'spline' && mode !== 'spline') {
        if (selected.size < 2) {
            showToast('Select at least 2 waypoints first to convert that segment to a spline.');
            return;
        }
    }

    // ---- Commit current curve → waypoints before leaving ----
    if (mode === 'spline' && m !== 'spline' && splineRange) pushUndo();
    commitCurveToWaypoints();

    mode = m;

    document.getElementById('btn-points').classList.toggle('active', m === 'points');
    document.getElementById('btn-spline').classList.toggle('active', m === 'spline');
    document.getElementById('points-panel').style.display = m === 'points' ? '' : 'none';
    document.getElementById('spline-panel').style.display = m === 'spline' ? '' : 'none';
    const names = {points: 'Points', spline: 'Catmull-Rom'};
    document.getElementById('status-mode').textContent = `Mode: ${names[m]}`;

    if (m === 'spline') {
        // Determine contiguous range from selection (min..max).
        const idxs = [...selected].sort((a, b) => a - b);
        const start = idxs[0];
        const end = idxs[idxs.length - 1];
        splineRange = {start, end};
        splineOriginalSegment = waypoints.slice(start, end + 1).map(p => ({...p}));
        // Phantom endpoints: use neighbors from the unedited portion of the closed loop
        const n = waypoints.length;
        splinePhantomBefore = {...waypoints[(start - 1 + n) % n]};
        splinePhantomAfter = {...waypoints[(end + 1) % n]};
        // Initialize control points = the entire segment (lossless on entry)
        controlPoints = splineOriginalSegment.map(p => ({...p}));
        const el = document.getElementById('cp-count');
        el.max = controlPoints.length;
        el.min = Math.min(2, controlPoints.length);
        el.value = controlPoints.length;
        document.getElementById('cp-count-val').textContent = controlPoints.length;
        regenerateSpline();
    } else {
        splineRange = null;
        splineOriginalSegment = [];
        splinePhantomBefore = null;
        splinePhantomAfter = null;
        controlPoints = [];
        splineCurve = [];
        splineWaypoints = [];
    }

    selected.clear(); cpSelected.clear();
    draw(); updateStatus();
}

// Commit current curve output to waypoints. Replaces only the edited
// portion of the raceline (waypoints[start..end]) with samples from
// the spline, preserving everything else.
function commitCurveToWaypoints() {
    if (mode !== 'spline' || !splineRange || splineWaypoints.length === 0) return;
    const {start, end} = splineRange;
    const replacement = splineWaypoints.map(p =>
        ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead}));
    waypoints.splice(start, end - start + 1, ...replacement);
    dirty = true;
}

// Get the active waypoints (what should be saved / shown in status).
// In spline mode, splice the spline output into a copy of waypoints so
// the saved file contains the unedited portion + the new spline portion.
function activeWaypoints() {
    if (mode === 'spline' && splineRange && splineWaypoints.length > 0) {
        const {start, end} = splineRange;
        const out = waypoints.slice();
        out.splice(start, end - start + 1, ...splineWaypoints.map(p =>
            ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead})));
        return out;
    }
    return waypoints;
}

// =====================================================================
//  COORDINATE TRANSFORMS
// =====================================================================
function worldToMapPx(wx, wy) {
    const ox = mapMeta.origin[0], oy = mapMeta.origin[1];
    const res = mapMeta.resolution, imgH = mapMeta.height;
    return [(wx - ox) / res, imgH - (wy - oy) / res];
}
function mapPxToWorld(px, py) {
    const ox = mapMeta.origin[0], oy = mapMeta.origin[1];
    const res = mapMeta.resolution, imgH = mapMeta.height;
    return [px * res + ox, (imgH - py) * res + oy];
}
function mapToScreen(mpx, mpy) { return [(mpx - panX) * zoom, (mpy - panY) * zoom]; }
function screenToMap(sx, sy) { return [sx / zoom + panX, sy / zoom + panY]; }
function worldToScreen(wx, wy) { return mapToScreen(...worldToMapPx(wx, wy)); }
function screenToWorld(sx, sy) { return mapPxToWorld(...screenToMap(sx, sy)); }

// =====================================================================
//  SPEED COLOR
// =====================================================================
let speedMin = 0, speedMax = 3;

function speedColor(s) {
    const t = Math.max(0, Math.min(1, (s - speedMin) / (speedMax - speedMin || 1)));
    let r, g, b;
    if (t < 0.5) {
        const u = t * 2;
        r = Math.round(u * 255); g = 255; b = Math.round((1 - u) * 136);
    } else {
        const u = (t - 0.5) * 2;
        r = 255; g = Math.round((1 - u) * 255); b = 0;
    }
    return `rgb(${r},${g},${b})`;
}

function updateSpeedRange(pts) {
    speedMin = Infinity; speedMax = -Infinity;
    for (const p of pts) {
        if (p.speed < speedMin) speedMin = p.speed;
        if (p.speed > speedMax) speedMax = p.speed;
    }
    if (speedMin === speedMax) { speedMin = Math.max(0, speedMin - 0.5); speedMax = speedMin + 1; }
    document.getElementById('leg-min').textContent = speedMin.toFixed(1);
    document.getElementById('leg-max').textContent = speedMax.toFixed(1);
}

// =====================================================================
//  CATMULL-ROM SPLINE (open, with phantom endpoints)
// =====================================================================
// Standard centripetal-form Catmull-Rom: tau = 0.5
function evalCatmullRomSegment(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    const tau = 0.5;
    return {
        x: tau * ((-t3 + 2*t2 - t) * p0.x + (3*t3 - 5*t2 + 2) * p1.x + (-3*t3 + 4*t2 + t) * p2.x + (t3 - t2) * p3.x),
        y: tau * ((-t3 + 2*t2 - t) * p0.y + (3*t3 - 5*t2 + 2) * p1.y + (-3*t3 + 4*t2 + t) * p2.y + (t3 - t2) * p3.y),
    };
}

// Generate `numSamples` points along an open Catmull-Rom curve through `pts`.
// `phantomBefore` is used as the "p0" before pts[0]; `phantomAfter` is "p3" after pts[n-1].
// First and last samples coincide with pts[0] and pts[n-1] (curve passes through every CP).
function buildSplineCurveOpen(pts, numSamples, phantomBefore, phantomAfter) {
    if (pts.length < 2 || numSamples < 2) return [];
    const n = pts.length;
    const numSegments = n - 1;
    const curve = [];
    for (let i = 0; i < numSamples; i++) {
        const t = (i / (numSamples - 1)) * numSegments;
        let segIdx = Math.floor(t);
        let frac = t - segIdx;
        if (segIdx >= numSegments) { segIdx = numSegments - 1; frac = 1; }

        const p1 = pts[segIdx];
        const p2 = pts[segIdx + 1];
        const p0 = segIdx > 0 ? pts[segIdx - 1] : (phantomBefore || p1);
        const p3 = segIdx < numSegments - 1 ? pts[segIdx + 2] : (phantomAfter || p2);

        const pt = evalCatmullRomSegment(p0, p1, p2, p3, frac);
        curve.push({
            x: pt.x, y: pt.y,
            speed: p1.speed + (p2.speed - p1.speed) * frac,
            lookahead: p1.lookahead + (p2.lookahead - p1.lookahead) * frac,
        });
    }
    return curve;
}

function regenerateSpline() {
    if (!splineRange || controlPoints.length < 2) {
        splineCurve = []; splineWaypoints = []; return;
    }
    const origN = splineOriginalSegment.length;
    // Preview: dense sampling for smooth on-screen rendering
    const numPreview = Math.max(200, origN * 4);
    splineCurve = buildSplineCurveOpen(controlPoints, numPreview, splinePhantomBefore, splinePhantomAfter);
    // Committed output: same count as original segment so the segment-replace
    // round-trip preserves waypoint indexing and minimizes information loss.
    splineWaypoints = buildSplineCurveOpen(controlPoints, origN, splinePhantomBefore, splinePhantomAfter);
}

// =====================================================================
//  GENERATE CONTROL POINTS FROM ORIGINAL SEGMENT
// =====================================================================
function generateControlPoints(count) {
    // Sample CPs from the *original segment snapshot* (not the live waypoints),
    // so toggling the slider doesn't compound any earlier resampling.
    // Endpoints are always preserved so the spline meets the unedited raceline.
    if (splineOriginalSegment.length === 0) return;
    const seg = splineOriginalSegment;
    const n = seg.length;
    count = Math.max(2, Math.min(count, n));

    if (count >= n) {
        controlPoints = seg.map(p => ({...p}));
    } else {
        const cumLen = [0];
        for (let i = 1; i < n; i++) {
            const dx = seg[i].x - seg[i-1].x;
            const dy = seg[i].y - seg[i-1].y;
            cumLen.push(cumLen[i-1] + Math.sqrt(dx*dx + dy*dy));
        }
        const totalLen = cumLen[n-1];
        controlPoints = [];
        for (let c = 0; c < count; c++) {
            const targetLen = (c / (count - 1)) * totalLen;
            let lo = 0, hi = n - 1;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (cumLen[mid] < targetLen) lo = mid + 1;
                else hi = mid;
            }
            const segIdx = Math.max(0, Math.min(n - 2, lo - 1));
            const segLen = cumLen[segIdx + 1] - cumLen[segIdx];
            const frac = segLen > 0 ? (targetLen - cumLen[segIdx]) / segLen : 0;
            const a = seg[segIdx], b = seg[segIdx + 1];
            controlPoints.push({
                x: a.x + (b.x - a.x) * frac,
                y: a.y + (b.y - a.y) * frac,
                speed: a.speed + (b.speed - a.speed) * frac,
                lookahead: a.lookahead + (b.lookahead - a.lookahead) * frac,
            });
        }
    }

    cpSelected.clear();
    regenerateSpline();
}

function onCpCountChange() {
    const v = parseInt(document.getElementById('cp-count').value);
    document.getElementById('cp-count-val').textContent = v;
    generateControlPoints(v);
    draw();
}

// Commit current curve to waypoints and switch to points mode for fine-tuning
function commitAndGoPoints() {
    const replacedCount = splineOriginalSegment.length;
    setMode('points');  // setMode already pushes undo + commits before leaving spline
    showToast(`Replaced ${replacedCount} waypoints with spline output.`);
}

// =====================================================================
//  DRAWING
// =====================================================================
function draw() {
    ctx.clearRect(0, 0, W, H);

    // Map image
    if (mapImg.complete && mapImg.naturalWidth) {
        ctx.save();
        ctx.translate(-panX * zoom, -panY * zoom);
        ctx.scale(zoom, zoom);
        ctx.drawImage(mapImg, 0, 0);
        ctx.restore();
    }

    if (mode === 'points') drawPointsMode();
    else drawSplineMode();
}

function drawPointsMode() {
    if (waypoints.length === 0) return;
    updateSpeedRange(waypoints);
    const pr = Math.max(3, 5 / Math.sqrt(zoom) * Math.min(zoom, 2));

    // Path lines
    ctx.beginPath();
    for (let i = 0; i < waypoints.length; i++) {
        const [sx, sy] = worldToScreen(waypoints[i].x, waypoints[i].y);
        i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    }
    if (waypoints.length > 1) { const [sx, sy] = worldToScreen(waypoints[0].x, waypoints[0].y); ctx.lineTo(sx, sy); }
    ctx.strokeStyle = 'rgba(100,150,255,0.3)'; ctx.lineWidth = 1.5; ctx.stroke();

    // Points
    for (let i = 0; i < waypoints.length; i++) {
        const wp = waypoints[i];
        const [sx, sy] = worldToScreen(wp.x, wp.y);
        if (sx < -20 || sy < -20 || sx > W + 20 || sy > H + 20) continue;
        const isSel = selected.has(i), isHov = i === hoveredIdx;
        ctx.beginPath();
        ctx.arc(sx, sy, isSel ? pr * 1.4 : pr, 0, Math.PI * 2);
        ctx.fillStyle = speedColor(wp.speed); ctx.fill();
        if (isSel) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
        else if (isHov) { ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5; ctx.stroke(); }
        if (zoom > 3 || isSel) {
            ctx.fillStyle = isSel ? '#fff' : 'rgba(255,255,255,0.5)';
            ctx.font = '10px monospace'; ctx.fillText(i, sx + pr + 3, sy - pr);
        }
    }

    // Lookahead radius
    const showLa = hoveredIdx >= 0 ? hoveredIdx : (selected.size === 1 ? [...selected][0] : -1);
    if (showLa >= 0) {
        const wp = waypoints[showLa];
        const [sx, sy] = worldToScreen(wp.x, wp.y);
        const laR = wp.lookahead / mapMeta.resolution * zoom;
        ctx.beginPath(); ctx.arc(sx, sy, laR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,100,0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Selection rect
    if (dragType === 'select-rect' && rectStart) {
        const [sx, sy] = rectStart;
        const [ex, ey] = [lastMouseScreen.x, lastMouseScreen.y];
        ctx.strokeStyle = '#e94560'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.strokeRect(sx, sy, ex - sx, ey - sy);
        ctx.fillStyle = 'rgba(233,69,96,0.1)'; ctx.fillRect(sx, sy, ex - sx, ey - sy);
        ctx.setLineDash([]);
    }
}

function drawSplineMode() {
    if (controlPoints.length < 2 || !splineRange) return;
    const {start, end} = splineRange;

    // Combine all visible points so the speed colormap spans the whole raceline
    updateSpeedRange(waypoints);

    // ---- Draw the unedited portion of the raceline as a faded path ----
    // Edit segment in waypoints[] is [start..end]; everything else is unchanged.
    if (waypoints.length > 1) {
        ctx.beginPath();
        const [hx, hy] = worldToScreen(waypoints[(end + 1) % waypoints.length].x,
                                       waypoints[(end + 1) % waypoints.length].y);
        ctx.moveTo(hx, hy);
        for (let i = 1; i <= waypoints.length - (end - start + 1); i++) {
            const idx = (end + 1 + i) % waypoints.length;
            const [sx, sy] = worldToScreen(waypoints[idx].x, waypoints[idx].y);
            ctx.lineTo(sx, sy);
        }
        ctx.strokeStyle = 'rgba(180,180,200,0.35)'; ctx.lineWidth = 1.5; ctx.stroke();

        // Faint dots for the unedited waypoints
        const faintR = Math.max(1.5, 2 / Math.sqrt(zoom) * Math.min(zoom, 1.5));
        for (let i = 0; i < waypoints.length; i++) {
            if (i >= start && i <= end) continue;
            const [sx, sy] = worldToScreen(waypoints[i].x, waypoints[i].y);
            if (sx < -10 || sy < -10 || sx > W + 10 || sy > H + 10) continue;
            ctx.beginPath(); ctx.arc(sx, sy, faintR, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(180,180,200,0.4)'; ctx.fill();
        }
    }

    // ---- Draw spline curve (color-coded by speed, OPEN — does not wrap) ----
    if (splineCurve.length > 1) {
        for (let i = 0; i < splineCurve.length - 1; i++) {
            const a = splineCurve[i];
            const b = splineCurve[i + 1];
            const [sx1, sy1] = worldToScreen(a.x, a.y);
            const [sx2, sy2] = worldToScreen(b.x, b.y);
            ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
            ctx.strokeStyle = speedColor(a.speed);
            ctx.lineWidth = 3; ctx.stroke();
        }
    }

    // ---- Draw output waypoint positions as small ticks ----
    if (splineWaypoints.length > 0) {
        const tickR = Math.max(2, 3 / Math.sqrt(zoom) * Math.min(zoom, 1.5));
        for (let i = 0; i < splineWaypoints.length; i++) {
            const wp = splineWaypoints[i];
            const [sx, sy] = worldToScreen(wp.x, wp.y);
            if (sx < -10 || sy < -10 || sx > W + 10 || sy > H + 10) continue;
            ctx.beginPath(); ctx.arc(sx, sy, tickR, 0, Math.PI * 2);
            ctx.fillStyle = speedColor(wp.speed);
            ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
        }
    }

    // ---- Draw control point connections (open, not closed) ----
    ctx.beginPath();
    for (let i = 0; i < controlPoints.length; i++) {
        const [sx, sy] = worldToScreen(controlPoints[i].x, controlPoints[i].y);
        i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

    // ---- Draw control points as diamonds ----
    const cpR = Math.max(5, 7 / Math.sqrt(zoom) * Math.min(zoom, 2));
    for (let i = 0; i < controlPoints.length; i++) {
        const cp = controlPoints[i];
        const [sx, sy] = worldToScreen(cp.x, cp.y);
        if (sx < -20 || sy < -20 || sx > W + 20 || sy > H + 20) continue;

        const isSel = cpSelected.has(i), isHov = i === cpHoveredIdx;
        const r = isSel ? cpR * 1.3 : cpR;

        ctx.beginPath();
        ctx.moveTo(sx, sy - r); ctx.lineTo(sx + r, sy);
        ctx.lineTo(sx, sy + r); ctx.lineTo(sx - r, sy); ctx.closePath();
        ctx.fillStyle = speedColor(cp.speed); ctx.fill();
        ctx.strokeStyle = isSel ? '#fff' : (isHov ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.5)');
        ctx.lineWidth = isSel ? 2.5 : 1.5; ctx.stroke();

        ctx.fillStyle = isSel ? '#fff' : 'rgba(255,255,255,0.6)';
        ctx.font = '11px monospace';
        ctx.fillText(`C${i}`, sx + r + 4, sy - r + 2);
    }

    // Lookahead for hovered/selected CP
    const showCp = cpHoveredIdx >= 0 ? cpHoveredIdx : (cpSelected.size === 1 ? [...cpSelected][0] : -1);
    if (showCp >= 0) {
        const cp = controlPoints[showCp];
        const [sx, sy] = worldToScreen(cp.x, cp.y);
        const laR = cp.lookahead / mapMeta.resolution * zoom;
        ctx.beginPath(); ctx.arc(sx, sy, laR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,100,0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Selection rect
    if (dragType === 'select-rect' && rectStart) {
        const [sx, sy] = rectStart;
        const [ex, ey] = [lastMouseScreen.x, lastMouseScreen.y];
        ctx.strokeStyle = '#e94560'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.strokeRect(sx, sy, ex - sx, ey - sy);
        ctx.fillStyle = 'rgba(233,69,96,0.1)'; ctx.fillRect(sx, sy, ex - sx, ey - sy);
        ctx.setLineDash([]);
    }
}

// =====================================================================
//  HIT TESTING
// =====================================================================
function hitTest(sx, sy, radius) {
    radius = radius || 10;
    let best = -1, bestDist = radius * radius;
    const arr = mode === 'points' ? waypoints : controlPoints;
    for (let i = 0; i < arr.length; i++) {
        const [wx, wy] = worldToScreen(arr[i].x, arr[i].y);
        const dx = sx - wx, dy = sy - wy;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; best = i; }
    }
    return best;
}

// =====================================================================
//  UNDO
// =====================================================================
function currentState() {
    return {
        waypoints: JSON.parse(JSON.stringify(waypoints)),
        controlPoints: JSON.parse(JSON.stringify(controlPoints)),
        splineRange: splineRange ? {...splineRange} : null,
        splineOriginalSegment: JSON.parse(JSON.stringify(splineOriginalSegment)),
        splinePhantomBefore: splinePhantomBefore ? {...splinePhantomBefore} : null,
        splinePhantomAfter: splinePhantomAfter ? {...splinePhantomAfter} : null,
    };
}

function restoreState(state) {
    waypoints = state.waypoints;
    controlPoints = state.controlPoints;
    splineRange = state.splineRange;
    splineOriginalSegment = state.splineOriginalSegment || [];
    splinePhantomBefore = state.splinePhantomBefore;
    splinePhantomAfter = state.splinePhantomAfter;
    dirty = true;
    selected.clear(); cpSelected.clear();
    if (mode === 'spline' && controlPoints.length >= 2) regenerateSpline();
    updatePanel(); updateCpPanel();
    draw(); updateStatus();
}

function pushUndo() {
    undoStack.push(currentState());
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack = []; // new action invalidates redo history
}

function undo() {
    if (undoStack.length === 0) return;
    redoStack.push(currentState());
    restoreState(undoStack.pop());
}

function redo() {
    if (redoStack.length === 0) return;
    undoStack.push(currentState());
    restoreState(redoStack.pop());
}

// =====================================================================
//  POINTS MODE PANEL
// =====================================================================
function updatePanel() {
    if (mode !== 'points') return;
    const n = selected.size;
    document.getElementById('sel-count').textContent = n > 0 ? `(${n})` : '';
    document.getElementById('no-sel').style.display = n === 0 ? '' : 'none';
    document.getElementById('single-sel').style.display = n === 1 ? '' : 'none';
    document.getElementById('multi-sel').style.display = n > 1 ? '' : 'none';

    if (n === 1) {
        const idx = [...selected][0]; const wp = waypoints[idx];
        document.getElementById('sel-idx').value = idx;
        document.getElementById('sel-x').value = wp.x.toFixed(6);
        document.getElementById('sel-y').value = wp.y.toFixed(6);
        document.getElementById('sel-speed').value = wp.speed;
        document.getElementById('sel-la').value = wp.lookahead;
    } else if (n > 1) {
        const list = document.getElementById('sel-list');
        list.innerHTML = '';
        for (const idx of [...selected].sort((a, b) => a - b)) {
            const wp = waypoints[idx];
            const div = document.createElement('div');
            div.className = 'wp-item selected';
            div.innerHTML = `<span>#${idx}</span><span>v=${wp.speed} la=${wp.lookahead}</span>`;
            div.onclick = () => { selected.clear(); selected.add(idx); updatePanel(); draw(); };
            list.appendChild(div);
        }
    }
}

function updateSingleField(field) {
    if (selected.size !== 1) return;
    const idx = [...selected][0];
    pushUndo();
    if (field === 'x') waypoints[idx].x = parseFloat(document.getElementById('sel-x').value);
    if (field === 'y') waypoints[idx].y = parseFloat(document.getElementById('sel-y').value);
    if (field === 'speed') waypoints[idx].speed = parseFloat(document.getElementById('sel-speed').value);
    if (field === 'lookahead') waypoints[idx].lookahead = parseFloat(document.getElementById('sel-la').value);
    dirty = true; draw(); updateStatus();
}

function applyBulk() {
    if (selected.size === 0) return;
    const sv = document.getElementById('bulk-speed').value;
    const lv = document.getElementById('bulk-la').value;
    if (!sv && !lv) return;
    pushUndo();
    for (const idx of selected) {
        if (sv) waypoints[idx].speed = parseFloat(sv);
        if (lv) waypoints[idx].lookahead = parseFloat(lv);
    }
    dirty = true; updatePanel(); draw(); updateStatus();
}

function deleteSelected() {
    if (selected.size === 0) return;
    if (!confirm(`Delete ${selected.size} waypoint(s)?`)) return;
    pushUndo();
    for (const idx of [...selected].sort((a, b) => b - a)) waypoints.splice(idx, 1);
    selected.clear(); dirty = true; updatePanel(); draw(); updateStatus();
}

// =====================================================================
//  SPLINE MODE PANEL
// =====================================================================
function updateCpPanel() {
    if (mode !== 'spline') return;
    const n = cpSelected.size;
    document.getElementById('cp-no-sel').style.display = n === 0 ? '' : 'none';
    document.getElementById('cp-single-sel').style.display = n === 1 ? '' : 'none';
    document.getElementById('cp-multi-sel').style.display = n > 1 ? '' : 'none';

    if (n === 1) {
        const idx = [...cpSelected][0]; const cp = controlPoints[idx];
        document.getElementById('cp-idx').value = `C${idx}`;
        document.getElementById('cp-x').value = cp.x.toFixed(6);
        document.getElementById('cp-y').value = cp.y.toFixed(6);
        document.getElementById('cp-speed').value = cp.speed;
        document.getElementById('cp-la').value = cp.lookahead;
    } else if (n > 1) {
        const list = document.getElementById('cp-list');
        list.innerHTML = '';
        for (const idx of [...cpSelected].sort((a, b) => a - b)) {
            const cp = controlPoints[idx];
            const div = document.createElement('div');
            div.className = 'wp-item selected';
            div.innerHTML = `<span>C${idx}</span><span>v=${cp.speed.toFixed(2)} la=${cp.lookahead.toFixed(2)}</span>`;
            div.onclick = () => { cpSelected.clear(); cpSelected.add(idx); updateCpPanel(); draw(); };
            list.appendChild(div);
        }
    }
}

function updateCpField(field) {
    if (cpSelected.size !== 1) return;
    const idx = [...cpSelected][0];
    pushUndo();
    if (field === 'x') controlPoints[idx].x = parseFloat(document.getElementById('cp-x').value);
    if (field === 'y') controlPoints[idx].y = parseFloat(document.getElementById('cp-y').value);
    if (field === 'speed') controlPoints[idx].speed = parseFloat(document.getElementById('cp-speed').value);
    if (field === 'lookahead') controlPoints[idx].lookahead = parseFloat(document.getElementById('cp-la').value);
    dirty = true; regenerateSpline(); draw(); updateStatus();
}

function applyCpBulk() {
    if (cpSelected.size === 0) return;
    const sv = document.getElementById('cp-bulk-speed').value;
    const lv = document.getElementById('cp-bulk-la').value;
    if (!sv && !lv) return;
    pushUndo();
    for (const idx of cpSelected) {
        if (sv) controlPoints[idx].speed = parseFloat(sv);
        if (lv) controlPoints[idx].lookahead = parseFloat(lv);
    }
    dirty = true; regenerateSpline(); updateCpPanel(); draw(); updateStatus();
}

function applyBulkMultiplier() {
    if (selected.size === 0) return;
    const sm = parseFloat(document.getElementById('bulk-speed-mul').value);
    const lm = parseFloat(document.getElementById('bulk-la-mul').value);
    if (!sm && !lm) return;
    pushUndo();
    for (const idx of selected) {
        if (sm) waypoints[idx].speed *= sm;
        if (lm) waypoints[idx].lookahead *= lm;
    }
    dirty = true; updatePanel(); draw(); updateStatus();
}

function applyCpMultiplier() {
    if (cpSelected.size === 0) return;
    const sm = parseFloat(document.getElementById('cp-bulk-speed-mul').value);
    const lm = parseFloat(document.getElementById('cp-bulk-la-mul').value);
    if (!sm && !lm) return;
    pushUndo();
    for (const idx of cpSelected) {
        if (sm) controlPoints[idx].speed *= sm;
        if (lm) controlPoints[idx].lookahead *= lm;
    }
    dirty = true; regenerateSpline(); updateCpPanel(); draw(); updateStatus();
}

function deleteCpSelected() {
    if (cpSelected.size === 0) return;
    // Don't let user delete the first or last CP (they anchor the spline to the unedited raceline).
    if (cpSelected.has(0) || cpSelected.has(controlPoints.length - 1)) {
        alert('Cannot delete the first or last control point — they anchor the spline to the rest of the raceline.');
        return;
    }
    if (controlPoints.length - cpSelected.size < 2) {
        alert('Need at least 2 control points for a spline.');
        return;
    }
    if (!confirm(`Delete ${cpSelected.size} control point(s)?`)) return;
    pushUndo();
    for (const idx of [...cpSelected].sort((a, b) => b - a)) controlPoints.splice(idx, 1);
    cpSelected.clear(); dirty = true;
    regenerateSpline(); updateCpPanel(); draw(); updateStatus();
}

function addControlPointNear(wx, wy) {
    // Insert a new CP into the closest open segment between two existing CPs.
    if (controlPoints.length < 2) return;
    let bestIdx = 0, bestDist = Infinity;
    for (let i = 0; i < controlPoints.length - 1; i++) {
        const a = controlPoints[i], b = controlPoints[i + 1];
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const d = (wx - mx) * (wx - mx) + (wy - my) * (wy - my);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    const a = controlPoints[bestIdx], b = controlPoints[bestIdx + 1];
    const newCp = {
        x: wx, y: wy,
        speed: (a.speed + b.speed) / 2,
        lookahead: (a.lookahead + b.lookahead) / 2,
    };
    pushUndo();
    controlPoints.splice(bestIdx + 1, 0, newCp);
    document.getElementById('cp-count').value = controlPoints.length;
    document.getElementById('cp-count-val').textContent = controlPoints.length;
    cpSelected.clear(); cpSelected.add(bestIdx + 1);
    dirty = true; regenerateSpline(); updateCpPanel(); draw(); updateStatus();
}

// =====================================================================
//  STATUS
// =====================================================================
function updateStatus() {
    const awp = activeWaypoints();
    document.getElementById('status-wp').textContent =
        mode === 'spline' ? `CPs: ${controlPoints.length} | Editing ${splineOriginalSegment.length} of ${awp.length} WPs` :
        `Waypoints: ${awp.length}`;
    document.getElementById('status-zoom').textContent = `Zoom: ${(zoom * 100).toFixed(0)}%`;
    document.getElementById('status-dirty').textContent = dirty ? '● Unsaved changes' : '';
    document.getElementById('status-dirty').style.color = dirty ? '#e94560' : '#888';
}

// =====================================================================
//  MOUSE HANDLERS
// =====================================================================
function onMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    // Pan
    if (e.button === 1 || e.button === 2 || (e.button === 0 && e.altKey)) {
        dragType = 'pan';
        dragStart = {x: sx, y: sy, panX, panY};
        canvas.style.cursor = 'grabbing';
        return;
    }
    if (e.button !== 0) return;

    const hitRadius = mode === 'spline' ? 16 : 12;
    const hit = hitTest(sx, sy, hitRadius);
    const sel = mode === 'points' ? selected : cpSelected;
    const panelFn = mode === 'points' ? updatePanel : updateCpPanel;

    if (hit >= 0 && (e.ctrlKey || e.metaKey)) {
        if (sel.has(hit)) sel.delete(hit); else sel.add(hit);
        panelFn(); draw(); return;
    }

    if (hit >= 0) {
        if (!sel.has(hit) && !e.shiftKey) sel.clear();
        sel.add(hit);
        panelFn();

        dragType = 'point';
        const [wx, wy] = screenToWorld(sx, sy);
        const arr = mode === 'points' ? waypoints : controlPoints;
        dragPointOffsets = [];
        for (const idx of sel) {
            dragPointOffsets.push({ idx, dx: arr[idx].x - wx, dy: arr[idx].y - wy });
        }
        pushUndo();
        draw(); return;
    }

    if (!e.shiftKey) sel.clear();
    dragType = 'select-rect';
    rectStart = [sx, sy];
    panelFn(); draw();
}

function onMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    lastMouseScreen = {x: sx, y: sy};
    const [wx, wy] = screenToWorld(sx, sy);
    lastMouseWorld = {x: wx, y: wy};

    document.getElementById('status-pos').textContent = `Mouse: (${wx.toFixed(3)}, ${wy.toFixed(3)})`;

    // Tooltip
    const tip = document.getElementById('tooltip');
    const hitRadius = mode === 'spline' ? 16 : 12;
    const hov = hitTest(sx, sy, hitRadius);

    if (mode === 'points') hoveredIdx = hov;
    else cpHoveredIdx = hov;

    if (hov >= 0 && dragType !== 'point') {
        const arr = mode === 'points' ? waypoints : controlPoints;
        const prefix = mode === 'points' ? `#${hov}` : `C${hov}`;
        const p = arr[hov];
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY + 14) + 'px';
        tip.textContent = `${prefix}\nx: ${p.x.toFixed(4)}\ny: ${p.y.toFixed(4)}\nspeed: ${p.speed.toFixed(2)}\nlookahead: ${p.lookahead.toFixed(2)}`;
    } else {
        tip.style.display = 'none';
    }

    if (dragType === 'pan') {
        panX = dragStart.panX - (sx - dragStart.x) / zoom;
        panY = dragStart.panY - (sy - dragStart.y) / zoom;
        draw(); updateStatus(); return;
    }

    if (dragType === 'point') {
        const arr = mode === 'points' ? waypoints : controlPoints;
        for (const off of dragPointOffsets) {
            arr[off.idx].x = wx + off.dx;
            arr[off.idx].y = wy + off.dy;
        }
        dirty = true;
        if (mode === 'spline') regenerateSpline();
        const panelFn = mode === 'points' ? updatePanel : updateCpPanel;
        panelFn();
        draw(); updateStatus(); return;
    }

    if (dragType === 'select-rect') { draw(); return; }
    draw();
}

function onMouseUp(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    if (dragType === 'select-rect' && rectStart) {
        const x1 = Math.min(rectStart[0], sx), y1 = Math.min(rectStart[1], sy);
        const x2 = Math.max(rectStart[0], sx), y2 = Math.max(rectStart[1], sy);
        if (Math.abs(x2 - x1) > 3 || Math.abs(y2 - y1) > 3) {
            const arr = mode === 'points' ? waypoints : controlPoints;
            const sel = mode === 'points' ? selected : cpSelected;
            for (let i = 0; i < arr.length; i++) {
                const [wx, wy] = worldToScreen(arr[i].x, arr[i].y);
                if (wx >= x1 && wx <= x2 && wy >= y1 && wy <= y2) sel.add(i);
            }
        }
        const panelFn = mode === 'points' ? updatePanel : updateCpPanel;
        panelFn();
    }

    if (dragType === 'point') {
        const panelFn = mode === 'points' ? updatePanel : updateCpPanel;
        panelFn();
    }

    dragType = null; rectStart = null;
    canvas.style.cursor = 'crosshair';
    draw();
}

function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const [mpx, mpy] = screenToMap(sx, sy);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.1, Math.min(100, zoom * factor));
    panX = mpx - sx / newZoom; panY = mpy - sy / newZoom;
    zoom = newZoom;
    draw(); updateStatus();
}

// =====================================================================
//  KEYBOARD
// =====================================================================
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); return; }
    if (e.ctrlKey && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) { e.preventDefault(); redo(); return; }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); save(); }
    if (e.ctrlKey && e.key === 'a') {
        e.preventDefault();
        const arr = mode === 'points' ? waypoints : controlPoints;
        const sel = mode === 'points' ? selected : cpSelected;
        for (let i = 0; i < arr.length; i++) sel.add(i);
        const panelFn = mode === 'points' ? updatePanel : updateCpPanel;
        panelFn(); draw();
    }
    if (e.key === 'Escape') {
        selected.clear(); cpSelected.clear();
        updatePanel(); updateCpPanel(); draw();
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && document.activeElement.tagName !== 'INPUT') {
        if (mode === 'points' && selected.size > 0) deleteSelected();
        if (mode === 'spline' && cpSelected.size > 0) deleteCpSelected();
    }
});

// =====================================================================
//  SAVE
// =====================================================================
async function save() {
    const wps = activeWaypoints();
    const resp = await fetch('/api/save', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({waypoints: wps}),
    });
    const result = await resp.json();
    if (result.ok) {
        // Also sync waypoints array so everything is consistent
        waypoints = wps.map(p => ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead}));
        dirty = false; updateStatus();
        showToast(`Saved ${wps.length} waypoints (from ${mode} mode)`);
    }
    else alert('Save failed: ' + result.error);
}

async function saveAs() {
    const name = prompt('Filename (saved next to original):', 'wp-neel-tuned.csv');
    if (!name) return;
    const wps = activeWaypoints();
    const resp = await fetch('/api/save', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({waypoints: wps, filename: name}),
    });
    const result = await resp.json();
    if (result.ok) {
        waypoints = wps.map(p => ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead}));
        dirty = false; updateStatus();
        showToast(`Saved as ${result.path}`);
    }
    else alert('Save failed: ' + result.error);
}

function showToast(msg) {
    const tip = document.getElementById('tooltip');
    tip.style.display = 'block'; tip.style.left = '50%'; tip.style.top = '40px';
    tip.style.transform = 'translateX(-50%)'; tip.textContent = msg;
    setTimeout(() => { tip.style.display = 'none'; tip.style.transform = ''; }, 2000);
}

// =====================================================================
//  VIEW
// =====================================================================
function resetView() {
    if (!mapImg.complete || !mapImg.naturalWidth) return;
    const iw = mapMeta.width, ih = mapMeta.height, pad = 30;
    zoom = Math.min((W - pad * 2) / iw, (H - pad * 2) / ih);
    const msw = iw * zoom, msh = ih * zoom;
    panX = -(W - msw) / (2 * zoom); panY = -(H - msh) / (2 * zoom);
    draw(); updateStatus();
}

function resize() {
    const wrap = document.getElementById('canvas-wrap');
    W = wrap.clientWidth; H = wrap.clientHeight;
    canvas.width = W; canvas.height = H;
    draw();
}

// =====================================================================
//  INIT
// =====================================================================
async function init() {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    mapMeta = data.mapMeta;
    waypoints = data.waypoints;

    document.getElementById('file-info').textContent =
        `CSV: ${data.csvPath} | Map: ${data.mapYaml} (${mapMeta.width}x${mapMeta.height}px, ${mapMeta.resolution}m/px)`;

    mapImg.onload = () => { resize(); resetView(); };
    mapImg.src = 'data:image/png;base64,' + data.mapPng;

    window.addEventListener('resize', resize);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, {passive: false});
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('dblclick', (e) => {
        if (mode !== 'spline') return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
        if (hitTest(sx, sy, 16) >= 0) return;
        const [wx, wy] = screenToWorld(sx, sy);
        addControlPointNear(wx, wy);
    });
    window.addEventListener('beforeunload', (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

    setMode('points');
    updateStatus();
}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = build_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/data":
            data = {
                "mapPng": MAP_PNG_B64,
                "mapMeta": MAP_META,
                "waypoints": [{"x": w[0], "y": w[1], "speed": w[2], "lookahead": w[3]} for w in WAYPOINTS],
                "csvPath": os.path.basename(CSV_PATH),
                "mapYaml": os.path.basename(MAP_YAML_PATH),
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            wps = body["waypoints"]
            filename = body.get("filename")
            save_path = os.path.join(os.path.dirname(CSV_PATH), filename) if filename else CSV_PATH

            try:
                with open(save_path, "w", newline="") as f:
                    f.write("x,y,speed,lookahead\n")
                    for wp in wps:
                        f.write(f"{wp['x']},{wp['y']},{wp['speed']},{wp['lookahead']}\n")

                global WAYPOINTS
                WAYPOINTS = [[wp["x"], wp["y"], wp["speed"], wp["lookahead"]] for wp in wps]

                resp = json.dumps({"ok": True, "path": save_path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp)
                print(f"  Saved {len(wps)} waypoints -> {save_path}")
            except Exception as e:
                resp = json.dumps({"ok": False, "error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp)
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global MAP_PNG_B64, MAP_META, WAYPOINTS, CSV_PATH, MAP_YAML_PATH

    parser = argparse.ArgumentParser(description="F1TENTH Interactive Waypoint Editor")
    parser.add_argument("--csv", default="pure_pursuit/waypoints/race2.csv", help="Path to waypoint CSV file")
    parser.add_argument("--map", default="f1tenth_gym_ros/maps/my_map1.yaml", help="Path to map YAML file")
    parser.add_argument("--port", type=int, default=8766, help="HTTP server port")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    waypoints_dir = os.path.join(script_dir, "pure_pursuit", "waypoints")
    # Bare filename (no directory component) → resolve into waypoints/
    if not os.path.isabs(args.csv) and os.sep not in args.csv and '/' not in args.csv:
        CSV_PATH = os.path.join(waypoints_dir, args.csv)
    else:
        CSV_PATH = args.csv if os.path.isabs(args.csv) else os.path.join(script_dir, args.csv)
    MAP_YAML_PATH = args.map if os.path.isabs(args.map) else os.path.join(script_dir, args.map)

    if not os.path.isfile(CSV_PATH):
        # Auto-create from the most recently modified CSV in the waypoints dir
        # waypoints_dir already set above
        candidates = [
            os.path.join(waypoints_dir, f)
            for f in os.listdir(waypoints_dir)
            if f.endswith(".csv") and os.path.isfile(os.path.join(waypoints_dir, f))
        ] if os.path.isdir(waypoints_dir) else []
        if not candidates:
            print(f"ERROR: CSV file not found and no existing CSVs in {waypoints_dir} to copy from.")
            sys.exit(1)
        latest = max(candidates, key=os.path.getmtime)
        import shutil
        shutil.copy2(latest, CSV_PATH)
        print(f"  Created  : {CSV_PATH}  (copied from {os.path.basename(latest)})")
    if not os.path.isfile(MAP_YAML_PATH):
        print(f"ERROR: Map YAML not found: {MAP_YAML_PATH}")
        sys.exit(1)

    meta = load_yaml_simple(MAP_YAML_PATH)
    pgm_path = meta.get("image", "")
    if not os.path.isabs(pgm_path):
        pgm_path = os.path.join(os.path.dirname(MAP_YAML_PATH), pgm_path)

    if not os.path.isfile(pgm_path):
        print(f"ERROR: PGM file not found: {pgm_path}")
        sys.exit(1)

    print(f"  Map YAML : {MAP_YAML_PATH}")
    print(f"  Map image: {pgm_path}")
    print(f"  CSV      : {CSV_PATH}")

    MAP_PNG_B64 = load_pgm_as_png_base64(pgm_path)

    if HAS_PIL:
        _img = Image.open(pgm_path)
        pgm_w, pgm_h = _img.size
    elif _is_png(pgm_path):
        pgm_w, pgm_h = parse_png_dimensions(pgm_path)
    else:
        pgm_w, pgm_h, _, _ = parse_pgm(pgm_path)
    MAP_META = {
        "resolution": float(meta.get("resolution", 0.05)),
        "origin": meta.get("origin", [0, 0, 0]),
        "width": pgm_w,
        "height": pgm_h,
    }
    print(f"  Map size : {pgm_w} x {pgm_h} px  ({MAP_META['resolution']} m/px)")
    print(f"  Origin   : {MAP_META['origin']}")

    WAYPOINTS = load_csv_waypoints(CSV_PATH)
    print(f"  Waypoints: {len(WAYPOINTS)}")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Editor running at: {url}")
    print(f"  Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
