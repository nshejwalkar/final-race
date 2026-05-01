#!/usr/bin/env python3
"""
F1TENTH Interactive Waypoint Editor
====================================
Overlays waypoints from a CSV onto a PGM map image.
Three editing modes (seamlessly linked — edits carry between modes):
  - Points Mode:     drag individual waypoints, bulk edit speed/lookahead
  - Catmull-Rom Mode: fit a C-R spline through control points, drag to reshape
  - Bezier Mode:     cubic Bezier with tangent handles for precise curve shaping

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


def load_pgm_as_png_base64(pgm_path: str) -> str:
    if HAS_PIL:
        img = Image.open(pgm_path).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    else:
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
    display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden;
}
#panel h2 {
    font-size: 14px; padding: 12px 14px 8px; color: #e94560;
    border-bottom: 1px solid #0f3460;
}
#panel-content { flex: 1; overflow-y: auto; padding: 10px 14px; }
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
    border-top: 1px solid #0f3460; padding: 10px 14px; flex-shrink: 0;
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
    <button class="btn-mode" id="btn-bezier" onclick="setMode('bezier')">Bezier</button>
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
            </div>
            <div class="section-box">
                <h3>Bulk Edit Selected</h3>
                <div class="field"><label>Set Speed (m/s)</label><input type="number" step="0.05" min="0" id="bulk-speed" placeholder="leave blank to keep"></div>
                <div class="field"><label>Set Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="bulk-la" placeholder="leave blank to keep"></div>
                <button class="btn-save" onclick="applyBulk()">Apply to Selected</button>
                <button class="btn-secondary" style="margin-top:4px" onclick="deleteSelected()">Delete Selected</button>
            </div>
        </div>

        <!-- ====== SPLINE MODE PANEL ====== -->
        <div id="spline-panel" style="display:none">
            <h2>Spline Controls</h2>
            <div id="spline-panel-content" style="flex:1;overflow-y:auto;padding:10px 14px;">
                <div class="field">
                    <label>Control Points</label>
                    <div class="range-row">
                        <input type="range" id="cp-count" min="6" max="100" value="25" oninput="onCpCountChange()">
                        <span id="cp-count-val">25</span>
                    </div>
                </div>
                <div class="field">
                    <label>Output Waypoints</label>
                    <div class="range-row">
                        <input type="range" id="wp-density" min="50" max="1000" step="10" value="400" oninput="onWpDensityChange()">
                        <span id="wp-density-val">400</span>
                    </div>
                </div>
                <div class="field">
                    <label>Spline Tension (0=loose, 1=tight)</label>
                    <div class="range-row">
                        <input type="range" id="spline-tension" min="0" max="100" value="50" oninput="onTensionChange()">
                        <span id="tension-val">0.50</span>
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
            </div>
            <div class="section-box">
                <h3>Bulk Edit Control Points</h3>
                <div class="field"><label>Set Speed (m/s)</label><input type="number" step="0.05" min="0" id="cp-bulk-speed" placeholder="leave blank to keep"></div>
                <div class="field"><label>Set Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="cp-bulk-la" placeholder="leave blank to keep"></div>
                <button class="btn-save" onclick="applyCpBulk()">Apply to Selected</button>
                <button class="btn-secondary" style="margin-top:4px" onclick="deleteCpSelected()">Delete Selected CPs</button>
                <button class="btn-mode" style="margin-top:8px" onclick="commitAndGoPoints()">Commit & Edit Points</button>
            </div>
        </div>

        <!-- ====== BEZIER MODE PANEL ====== -->
        <div id="bezier-panel" style="display:none">
            <h2>Bezier Controls</h2>
            <div style="flex:1;overflow-y:auto;padding:10px 14px;">
                <div class="field">
                    <label>Anchor Points</label>
                    <div class="range-row">
                        <input type="range" id="bz-count" min="4" max="80" value="20" oninput="onBzCountChange()">
                        <span id="bz-count-val">20</span>
                    </div>
                </div>
                <div class="field">
                    <label>Output Waypoints</label>
                    <div class="range-row">
                        <input type="range" id="bz-density" min="50" max="1000" step="10" value="400" oninput="onBzDensityChange()">
                        <span id="bz-density-val">400</span>
                    </div>
                </div>
                <div class="field" style="margin-top:4px">
                    <label><input type="checkbox" id="bz-smooth" checked onchange="onBzSmoothToggle()"> Smooth handles (mirror tangents)</label>
                </div>
                <hr style="border-color:#0f3460;margin:12px 0">
                <div id="bz-no-sel" style="color:#666;font-size:12px;padding:6px 0;">
                    Click an anchor (square) to select.<br>
                    Drag <b>handles</b> (circles) to adjust tangents.<br>
                    <b>Double-click</b> to add an anchor.<br>
                    <b>Delete</b> to remove selected anchors.
                </div>
                <div id="bz-single-sel" style="display:none">
                    <div class="field"><label>Anchor Index</label><input type="text" id="bz-idx" readonly></div>
                    <div class="field"><label>X (meters)</label><input type="number" step="0.001" id="bz-x" onchange="updateBzField('x')"></div>
                    <div class="field"><label>Y (meters)</label><input type="number" step="0.001" id="bz-y" onchange="updateBzField('y')"></div>
                    <div class="field"><label>Speed (m/s)</label><input type="number" step="0.05" min="0" id="bz-speed" onchange="updateBzField('speed')"></div>
                    <div class="field"><label>Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="bz-la" onchange="updateBzField('lookahead')"></div>
                </div>
                <div id="bz-multi-sel" style="display:none"><div class="wp-list" id="bz-list"></div></div>
            </div>
            <div class="section-box">
                <h3>Bulk Edit Anchors</h3>
                <div class="field"><label>Set Speed (m/s)</label><input type="number" step="0.05" min="0" id="bz-bulk-speed" placeholder="leave blank to keep"></div>
                <div class="field"><label>Set Lookahead (m)</label><input type="number" step="0.05" min="0.1" id="bz-bulk-la" placeholder="leave blank to keep"></div>
                <button class="btn-save" onclick="applyBzBulk()">Apply to Selected</button>
                <button class="btn-secondary" style="margin-top:4px" onclick="deleteBzSelected()">Delete Selected Anchors</button>
                <button class="btn-mode" style="margin-top:8px" onclick="commitAndGoPoints()">Commit & Edit Points</button>
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
//  MODE: 'points' | 'spline' | 'bezier'
// =====================================================================
let mode = 'points';

// Spline state
let controlPoints = [];
let splineCurve = [];
let splineWaypoints = [];
let cpSelected = new Set();
let cpHoveredIdx = -1;
let splineTension = 0.5;
let wpDensity = 400;

// Bezier state
// Each anchor: {x, y, hix, hiy, hox, hoy, speed, lookahead}
//   hix/hiy = handle-in offset (relative to anchor)
//   hox/hoy = handle-out offset (relative to anchor)
let bezierAnchors = [];
let bezierCurve = [];       // dense preview
let bezierWaypoints = [];   // output
let bzSelected = new Set();
let bzHoveredIdx = -1;
let bzHoveredType = null;   // 'anchor' | 'handle-in' | 'handle-out'
let bzDragType = null;      // what we're dragging in bezier mode
let bzDragIdx = -1;
let bzSmooth = true;
let bzDensity = 400;

function setMode(m) {
    // ---- Commit current curve → waypoints before leaving ----
    commitCurveToWaypoints();

    const prevMode = mode;
    mode = m;

    document.getElementById('btn-points').classList.toggle('active', m === 'points');
    document.getElementById('btn-spline').classList.toggle('active', m === 'spline');
    document.getElementById('btn-bezier').classList.toggle('active', m === 'bezier');
    document.getElementById('points-panel').style.display = m === 'points' ? '' : 'none';
    document.getElementById('spline-panel').style.display = m === 'spline' ? '' : 'none';
    document.getElementById('bezier-panel').style.display = m === 'bezier' ? '' : 'none';
    const names = {points: 'Points', spline: 'Catmull-Rom', bezier: 'Bezier'};
    document.getElementById('status-mode').textContent = `Mode: ${names[m]}`;

    // ---- Re-fit curve from (possibly updated) waypoints ----
    if (m === 'spline') {
        generateControlPoints(parseInt(document.getElementById('cp-count').value));
    }
    if (m === 'bezier') {
        generateBezierAnchors(parseInt(document.getElementById('bz-count').value));
    }

    selected.clear(); cpSelected.clear(); bzSelected.clear();
    draw(); updateStatus();
}

// Commit current curve output to waypoints (lossless for the curve representation)
function commitCurveToWaypoints() {
    if (mode === 'spline' && splineWaypoints.length > 0) {
        waypoints = splineWaypoints.map(p => ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead}));
        dirty = true;
    }
    if (mode === 'bezier' && bezierWaypoints.length > 0) {
        waypoints = bezierWaypoints.map(p => ({x:p.x, y:p.y, speed:p.speed, lookahead:p.lookahead}));
        dirty = true;
    }
}

// Get the active waypoints (what should be saved / shown in status)
function activeWaypoints() {
    if (mode === 'spline' && splineWaypoints.length > 0) return splineWaypoints;
    if (mode === 'bezier' && bezierWaypoints.length > 0) return bezierWaypoints;
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
//  CATMULL-ROM SPLINE
// =====================================================================
function catmullRom(p0, p1, p2, p3, t, alpha) {
    // alpha: 0 = uniform, 0.5 = centripetal, 1 = chordal
    // Using matrix form with tension parameter
    const t2 = t * t, t3 = t2 * t;
    const a = alpha; // tension factor (0.5 = standard Catmull-Rom)
    return {
        x: a * ((-t3 + 2*t2 - t) * p0.x + (3*t3 - 5*t2 + 2) * p1.x + (-3*t3 + 4*t2 + t) * p2.x + (t3 - t2) * p3.x),
        y: a * ((-t3 + 2*t2 - t) * p0.y + (3*t3 - 5*t2 + 2) * p1.y + (-3*t3 + 4*t2 + t) * p2.y + (t3 - t2) * p3.y),
    };
}

function evalCatmullRomClosed(pts, t, tension) {
    // t in [0, pts.length), wraps around
    const n = pts.length;
    const i = Math.floor(t) % n;
    const frac = t - Math.floor(t);
    const p0 = pts[(i - 1 + n) % n];
    const p1 = pts[i % n];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];

    const s = 1 - tension; // s=1 → loose (alpha=0.5), s=0 → tight (alpha=1)
    const alpha = 0.5 + s * 0.5; // range [0.5, 1.0]

    // Standard Catmull-Rom with adjustable tension via scaling
    const tt = frac, tt2 = tt * tt, tt3 = tt2 * tt;
    // Tension-adjusted matrix (tau = 0.5 * (1 - tension_user) ... but let's use simple approach)
    const tau = 0.5 * (1 + (1 - tension) * 0.5); // range roughly [0.5, 0.75]
    return {
        x: tau * ((-tt3 + 2*tt2 - tt) * p0.x + (3*tt3 - 5*tt2 + 2) * p1.x + (-3*tt3 + 4*tt2 + tt) * p2.x + (tt3 - tt2) * p3.x),
        y: tau * ((-tt3 + 2*tt2 - tt) * p0.y + (3*tt3 - 5*tt2 + 2) * p1.y + (-3*tt3 + 4*tt2 + tt) * p2.y + (tt3 - tt2) * p3.y),
    };
}

function buildSplineCurve(pts, numSamples, tension) {
    // Generate dense samples along the closed Catmull-Rom spline
    if (pts.length < 3) return [];
    const n = pts.length;
    const curve = [];
    for (let i = 0; i < numSamples; i++) {
        const t = (i / numSamples) * n;
        const pt = evalCatmullRomClosed(pts, t, tension);
        // Interpolate speed/lookahead
        const idx = Math.floor(t) % n;
        const frac = t - Math.floor(t);
        const sp1 = pts[idx % n].speed, sp2 = pts[(idx + 1) % n].speed;
        const la1 = pts[idx % n].lookahead, la2 = pts[(idx + 1) % n].lookahead;
        curve.push({
            x: pt.x, y: pt.y,
            speed: sp1 + (sp2 - sp1) * frac,
            lookahead: la1 + (la2 - la1) * frac,
        });
    }
    return curve;
}

function regenerateSpline() {
    const tension = splineTension;
    const numPreview = Math.max(500, wpDensity * 2);
    splineCurve = buildSplineCurve(controlPoints, numPreview, tension);
    splineWaypoints = buildSplineCurve(controlPoints, wpDensity, tension);
}

// =====================================================================
//  GENERATE CONTROL POINTS FROM WAYPOINTS
// =====================================================================
function generateControlPoints(count) {
    if (waypoints.length === 0) return;
    count = Math.min(count, waypoints.length);
    controlPoints = [];

    // Compute cumulative arc length
    const n = waypoints.length;
    const cumLen = [0];
    for (let i = 1; i <= n; i++) {
        const a = waypoints[(i - 1) % n], b = waypoints[i % n];
        const dx = b.x - a.x, dy = b.y - a.y;
        cumLen.push(cumLen[i - 1] + Math.sqrt(dx * dx + dy * dy));
    }
    const totalLen = cumLen[n];

    // Sample control points at equal arc-length intervals
    for (let c = 0; c < count; c++) {
        const targetLen = (c / count) * totalLen;
        // Binary search for segment
        let lo = 0, hi = n;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (cumLen[mid] < targetLen) lo = mid + 1;
            else hi = mid;
        }
        const segIdx = Math.max(0, lo - 1);
        const segLen = cumLen[segIdx + 1] - cumLen[segIdx];
        const frac = segLen > 0 ? (targetLen - cumLen[segIdx]) / segLen : 0;

        const a = waypoints[segIdx % n], b = waypoints[(segIdx + 1) % n];
        controlPoints.push({
            x: a.x + (b.x - a.x) * frac,
            y: a.y + (b.y - a.y) * frac,
            speed: a.speed + (b.speed - a.speed) * frac,
            lookahead: a.lookahead + (b.lookahead - a.lookahead) * frac,
        });
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

function onWpDensityChange() {
    wpDensity = parseInt(document.getElementById('wp-density').value);
    document.getElementById('wp-density-val').textContent = wpDensity;
    regenerateSpline();
    draw();
}

function onTensionChange() {
    splineTension = parseInt(document.getElementById('spline-tension').value) / 100;
    document.getElementById('tension-val').textContent = splineTension.toFixed(2);
    regenerateSpline();
    draw();
}

// Commit current curve to waypoints and switch to points mode for fine-tuning
function commitAndGoPoints() {
    pushUndo();
    commitCurveToWaypoints();
    showToast(`Committed ${waypoints.length} waypoints from ${mode} mode`);
    setMode('points');
}

// =====================================================================
//  CUBIC BEZIER
// =====================================================================
function cubicBezier(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t) {
    const u = 1 - t, u2 = u * u, u3 = u2 * u;
    const t2 = t * t, t3 = t2 * t;
    return {
        x: u3*p0x + 3*u2*t*p1x + 3*u*t2*p2x + t3*p3x,
        y: u3*p0y + 3*u2*t*p1y + 3*u*t2*p2y + t3*p3y,
    };
}

function generateBezierAnchors(count) {
    if (waypoints.length === 0) return;
    count = Math.min(count, waypoints.length);
    bezierAnchors = [];

    // Sample anchor positions at equal arc-length (same as spline)
    const n = waypoints.length;
    const cumLen = [0];
    for (let i = 1; i <= n; i++) {
        const a = waypoints[(i-1)%n], b = waypoints[i%n];
        cumLen.push(cumLen[i-1] + Math.hypot(b.x-a.x, b.y-a.y));
    }
    const totalLen = cumLen[n];
    const sampled = [];
    for (let c = 0; c < count; c++) {
        const target = (c / count) * totalLen;
        let lo = 0, hi = n;
        while (lo < hi) { const mid = (lo+hi)>>1; if (cumLen[mid]<target) lo=mid+1; else hi=mid; }
        const si = Math.max(0, lo-1);
        const segL = cumLen[si+1]-cumLen[si];
        const f = segL > 0 ? (target-cumLen[si])/segL : 0;
        const a = waypoints[si%n], b = waypoints[(si+1)%n];
        sampled.push({
            x: a.x+(b.x-a.x)*f, y: a.y+(b.y-a.y)*f,
            speed: a.speed+(b.speed-a.speed)*f,
            lookahead: a.lookahead+(b.lookahead-a.lookahead)*f,
        });
    }

    // Convert to anchors with auto-computed handles (Catmull-Rom → Bezier)
    for (let i = 0; i < count; i++) {
        const prev = sampled[(i-1+count)%count];
        const curr = sampled[i];
        const next = sampled[(i+1)%count];
        // Tangent = (next - prev) / 2, handle length = tangent / 3
        const tx = (next.x - prev.x) / 6;
        const ty = (next.y - prev.y) / 6;
        bezierAnchors.push({
            x: curr.x, y: curr.y,
            hix: -tx, hiy: -ty,  // handle-in (arriving)
            hox: tx, hoy: ty,    // handle-out (departing)
            speed: curr.speed, lookahead: curr.lookahead,
        });
    }

    bzSelected.clear();
    regenerateBezier();
}

function regenerateBezier() {
    const anchors = bezierAnchors;
    if (anchors.length < 2) { bezierCurve = []; bezierWaypoints = []; return; }
    const n = anchors.length;
    const samplesPerSeg = Math.max(10, Math.ceil(bzDensity * 2 / n));

    // Dense preview
    bezierCurve = [];
    for (let i = 0; i < n; i++) {
        const a0 = anchors[i], a1 = anchors[(i+1)%n];
        const p0x = a0.x, p0y = a0.y;
        const p1x = a0.x + a0.hox, p1y = a0.y + a0.hoy;
        const p2x = a1.x + a1.hix, p2y = a1.y + a1.hiy;
        const p3x = a1.x, p3y = a1.y;
        for (let j = 0; j < samplesPerSeg; j++) {
            const t = j / samplesPerSeg;
            const pt = cubicBezier(p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y,t);
            pt.speed = a0.speed + (a1.speed - a0.speed) * t;
            pt.lookahead = a0.lookahead + (a1.lookahead - a0.lookahead) * t;
            bezierCurve.push(pt);
        }
    }

    // Output waypoints at equal arc-length
    if (bezierCurve.length < 2) { bezierWaypoints = []; return; }
    const cLen = [0];
    for (let i = 1; i < bezierCurve.length; i++) {
        cLen.push(cLen[i-1] + Math.hypot(bezierCurve[i].x-bezierCurve[i-1].x, bezierCurve[i].y-bezierCurve[i-1].y));
    }
    // Add closing segment
    const closeDist = Math.hypot(bezierCurve[0].x-bezierCurve[bezierCurve.length-1].x, bezierCurve[0].y-bezierCurve[bezierCurve.length-1].y);
    const totalArc = cLen[cLen.length-1] + closeDist;

    bezierWaypoints = [];
    for (let w = 0; w < bzDensity; w++) {
        const target = (w / bzDensity) * totalArc;
        let lo = 0, hi = cLen.length - 1;
        while (lo < hi) { const mid = (lo+hi)>>1; if (cLen[mid]<target) lo=mid+1; else hi=mid; }
        const si = Math.max(0, lo-1);
        if (si >= bezierCurve.length - 1) {
            // In closing segment
            const f = totalArc > 0 ? (target - cLen[cLen.length-1]) / closeDist : 0;
            const a = bezierCurve[bezierCurve.length-1], b = bezierCurve[0];
            bezierWaypoints.push({
                x: a.x+(b.x-a.x)*f, y: a.y+(b.y-a.y)*f,
                speed: a.speed+(b.speed-a.speed)*f,
                lookahead: a.lookahead+(b.lookahead-a.lookahead)*f,
            });
        } else {
            const segL = cLen[si+1]-cLen[si];
            const f = segL > 0 ? (target-cLen[si])/segL : 0;
            const a = bezierCurve[si], b = bezierCurve[si+1];
            bezierWaypoints.push({
                x: a.x+(b.x-a.x)*f, y: a.y+(b.y-a.y)*f,
                speed: a.speed+(b.speed-a.speed)*f,
                lookahead: a.lookahead+(b.lookahead-a.lookahead)*f,
            });
        }
    }
}

function onBzCountChange() {
    const v = parseInt(document.getElementById('bz-count').value);
    document.getElementById('bz-count-val').textContent = v;
    generateBezierAnchors(v);
    draw();
}

function onBzDensityChange() {
    bzDensity = parseInt(document.getElementById('bz-density').value);
    document.getElementById('bz-density-val').textContent = bzDensity;
    regenerateBezier();
    draw();
}

function onBzSmoothToggle() {
    bzSmooth = document.getElementById('bz-smooth').checked;
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
    else if (mode === 'spline') drawSplineMode();
    else drawBezierMode();
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
    if (controlPoints.length < 3) return;

    // Determine color range from control points
    updateSpeedRange(controlPoints);

    // Draw the existing waypoints faintly in background
    if (waypoints.length > 1) {
        ctx.beginPath();
        for (let i = 0; i < waypoints.length; i++) {
            const [sx, sy] = worldToScreen(waypoints[i].x, waypoints[i].y);
            i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
        }
        const [sx0, sy0] = worldToScreen(waypoints[0].x, waypoints[0].y);
        ctx.lineTo(sx0, sy0);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.stroke();

        // Faint dots
        const faintR = Math.max(1.5, 2 / Math.sqrt(zoom) * Math.min(zoom, 1.5));
        for (let i = 0; i < waypoints.length; i++) {
            const [sx, sy] = worldToScreen(waypoints[i].x, waypoints[i].y);
            if (sx < -10 || sy < -10 || sx > W + 10 || sy > H + 10) continue;
            ctx.beginPath(); ctx.arc(sx, sy, faintR, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,0.15)'; ctx.fill();
        }
    }

    // Draw spline curve (color-coded by speed)
    if (splineCurve.length > 1) {
        for (let i = 0; i < splineCurve.length; i++) {
            const a = splineCurve[i];
            const b = splineCurve[(i + 1) % splineCurve.length];
            const [sx1, sy1] = worldToScreen(a.x, a.y);
            const [sx2, sy2] = worldToScreen(b.x, b.y);
            ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
            ctx.strokeStyle = speedColor(a.speed);
            ctx.lineWidth = 3; ctx.stroke();
        }
    }

    // Draw output waypoint positions as small ticks
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

    // Draw control point connections
    ctx.beginPath();
    for (let i = 0; i < controlPoints.length; i++) {
        const [sx, sy] = worldToScreen(controlPoints[i].x, controlPoints[i].y);
        i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    }
    const [csx, csy] = worldToScreen(controlPoints[0].x, controlPoints[0].y);
    ctx.lineTo(csx, csy);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

    // Draw control points as diamonds
    const cpR = Math.max(5, 7 / Math.sqrt(zoom) * Math.min(zoom, 2));
    for (let i = 0; i < controlPoints.length; i++) {
        const cp = controlPoints[i];
        const [sx, sy] = worldToScreen(cp.x, cp.y);
        if (sx < -20 || sy < -20 || sx > W + 20 || sy > H + 20) continue;

        const isSel = cpSelected.has(i), isHov = i === cpHoveredIdx;
        const r = isSel ? cpR * 1.3 : cpR;

        // Diamond shape
        ctx.beginPath();
        ctx.moveTo(sx, sy - r); ctx.lineTo(sx + r, sy);
        ctx.lineTo(sx, sy + r); ctx.lineTo(sx - r, sy); ctx.closePath();
        ctx.fillStyle = speedColor(cp.speed); ctx.fill();
        ctx.strokeStyle = isSel ? '#fff' : (isHov ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.5)');
        ctx.lineWidth = isSel ? 2.5 : 1.5; ctx.stroke();

        // Label
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

function drawBezierMode() {
    if (bezierAnchors.length < 2) return;
    updateSpeedRange(bezierAnchors);

    // Faint original waypoints
    if (waypoints.length > 1) {
        ctx.beginPath();
        for (let i = 0; i < waypoints.length; i++) {
            const [sx,sy] = worldToScreen(waypoints[i].x, waypoints[i].y);
            i===0 ? ctx.moveTo(sx,sy) : ctx.lineTo(sx,sy);
        }
        ctx.lineTo(...worldToScreen(waypoints[0].x, waypoints[0].y));
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.stroke();
    }

    // Draw bezier curve (color-coded)
    if (bezierCurve.length > 1) {
        for (let i = 0; i < bezierCurve.length; i++) {
            const a = bezierCurve[i], b = bezierCurve[(i+1)%bezierCurve.length];
            const [sx1,sy1] = worldToScreen(a.x,a.y);
            const [sx2,sy2] = worldToScreen(b.x,b.y);
            ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2);
            ctx.strokeStyle = speedColor(a.speed); ctx.lineWidth = 3; ctx.stroke();
        }
    }

    // Output waypoint ticks
    if (bezierWaypoints.length > 0) {
        const tr = Math.max(2, 3/Math.sqrt(zoom)*Math.min(zoom,1.5));
        for (const wp of bezierWaypoints) {
            const [sx,sy] = worldToScreen(wp.x,wp.y);
            if (sx<-10||sy<-10||sx>W+10||sy>H+10) continue;
            ctx.beginPath(); ctx.arc(sx,sy,tr,0,Math.PI*2);
            ctx.fillStyle = speedColor(wp.speed); ctx.globalAlpha=0.4; ctx.fill(); ctx.globalAlpha=1;
        }
    }

    const aR = Math.max(5, 7/Math.sqrt(zoom)*Math.min(zoom,2));
    const hR = Math.max(3, 5/Math.sqrt(zoom)*Math.min(zoom,2));

    // Draw handles and tangent lines for selected/hovered anchors
    for (let i = 0; i < bezierAnchors.length; i++) {
        const a = bezierAnchors[i];
        const isSel = bzSelected.has(i);
        const isHov = i === bzHoveredIdx;
        if (!isSel && !isHov) continue;

        const [ax,ay] = worldToScreen(a.x, a.y);
        const [hix,hiy] = worldToScreen(a.x+a.hix, a.y+a.hiy);
        const [hox,hoy] = worldToScreen(a.x+a.hox, a.y+a.hoy);

        // Tangent lines
        ctx.beginPath(); ctx.moveTo(hix,hiy); ctx.lineTo(ax,ay); ctx.lineTo(hox,hoy);
        ctx.strokeStyle = 'rgba(100,200,255,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();

        // Handle-in circle
        ctx.beginPath(); ctx.arc(hix,hiy,hR,0,Math.PI*2);
        ctx.fillStyle = '#4488ff'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();

        // Handle-out circle
        ctx.beginPath(); ctx.arc(hox,hoy,hR,0,Math.PI*2);
        ctx.fillStyle = '#ff8844'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    // Draw anchor points as squares
    for (let i = 0; i < bezierAnchors.length; i++) {
        const a = bezierAnchors[i];
        const [sx,sy] = worldToScreen(a.x, a.y);
        if (sx<-20||sy<-20||sx>W+20||sy>H+20) continue;
        const isSel = bzSelected.has(i), isHov = i === bzHoveredIdx;
        const r = isSel ? aR*1.3 : aR;

        ctx.fillStyle = speedColor(a.speed);
        ctx.fillRect(sx-r, sy-r, r*2, r*2);
        ctx.strokeStyle = isSel ? '#fff' : (isHov ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.5)');
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.strokeRect(sx-r, sy-r, r*2, r*2);

        // Label
        ctx.fillStyle = isSel ? '#fff' : 'rgba(255,255,255,0.6)';
        ctx.font = '11px monospace';
        ctx.fillText(`B${i}`, sx+r+4, sy-r+2);
    }

    // Lookahead for selected
    const showBz = bzHoveredIdx >= 0 ? bzHoveredIdx : (bzSelected.size===1 ? [...bzSelected][0] : -1);
    if (showBz >= 0) {
        const a = bezierAnchors[showBz];
        const [sx,sy] = worldToScreen(a.x,a.y);
        const laR = a.lookahead / mapMeta.resolution * zoom;
        ctx.beginPath(); ctx.arc(sx,sy,laR,0,Math.PI*2);
        ctx.strokeStyle='rgba(255,255,100,0.4)'; ctx.lineWidth=1;
        ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Selection rect
    if (dragType === 'select-rect' && rectStart) {
        const [sx,sy] = rectStart;
        const [ex,ey] = [lastMouseScreen.x, lastMouseScreen.y];
        ctx.strokeStyle='#e94560'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        ctx.strokeRect(sx,sy,ex-sx,ey-sy);
        ctx.fillStyle='rgba(233,69,96,0.1)'; ctx.fillRect(sx,sy,ex-sx,ey-sy);
        ctx.setLineDash([]);
    }
}

// =====================================================================
//  HIT TESTING
// =====================================================================
function hitTest(sx, sy, radius) {
    radius = radius || 10;
    let best = -1, bestDist = radius * radius;
    const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
    for (let i = 0; i < arr.length; i++) {
        const [wx, wy] = worldToScreen(arr[i].x, arr[i].y);
        const dx = sx - wx, dy = sy - wy;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; best = i; }
    }
    return best;
}

// Hit test bezier handles (returns {idx, type} or null)
function hitTestBzHandle(sx, sy, radius) {
    radius = radius || 14;
    const r2 = radius * radius;
    // Only check handles of selected/hovered anchors
    for (let i = 0; i < bezierAnchors.length; i++) {
        if (!bzSelected.has(i) && i !== bzHoveredIdx) continue;
        const a = bezierAnchors[i];
        // Handle-in
        const [hix,hiy] = worldToScreen(a.x+a.hix, a.y+a.hiy);
        if ((sx-hix)**2+(sy-hiy)**2 < r2) return {idx:i, type:'handle-in'};
        // Handle-out
        const [hox,hoy] = worldToScreen(a.x+a.hox, a.y+a.hoy);
        if ((sx-hox)**2+(sy-hoy)**2 < r2) return {idx:i, type:'handle-out'};
    }
    return null;
}

// =====================================================================
//  UNDO
// =====================================================================
function currentState() {
    return {
        waypoints: JSON.parse(JSON.stringify(waypoints)),
        controlPoints: JSON.parse(JSON.stringify(controlPoints)),
        bezierAnchors: JSON.parse(JSON.stringify(bezierAnchors)),
    };
}

function restoreState(state) {
    waypoints = state.waypoints;
    controlPoints = state.controlPoints;
    bezierAnchors = state.bezierAnchors;
    dirty = true;
    selected.clear(); cpSelected.clear(); bzSelected.clear();
    if (mode === 'spline' && controlPoints.length >= 3) regenerateSpline();
    if (mode === 'bezier' && bezierAnchors.length >= 2) regenerateBezier();
    updatePanel(); updateCpPanel(); updateBzPanel();
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

// =====================================================================
//  BEZIER MODE PANEL
// =====================================================================
function updateBzPanel() {
    if (mode !== 'bezier') return;
    const n = bzSelected.size;
    document.getElementById('bz-no-sel').style.display = n===0 ? '' : 'none';
    document.getElementById('bz-single-sel').style.display = n===1 ? '' : 'none';
    document.getElementById('bz-multi-sel').style.display = n>1 ? '' : 'none';

    if (n === 1) {
        const idx = [...bzSelected][0]; const a = bezierAnchors[idx];
        document.getElementById('bz-idx').value = `B${idx}`;
        document.getElementById('bz-x').value = a.x.toFixed(6);
        document.getElementById('bz-y').value = a.y.toFixed(6);
        document.getElementById('bz-speed').value = a.speed;
        document.getElementById('bz-la').value = a.lookahead;
    } else if (n > 1) {
        const list = document.getElementById('bz-list');
        list.innerHTML = '';
        for (const idx of [...bzSelected].sort((a,b)=>a-b)) {
            const a = bezierAnchors[idx];
            const div = document.createElement('div');
            div.className = 'wp-item selected';
            div.innerHTML = `<span>B${idx}</span><span>v=${a.speed.toFixed(2)} la=${a.lookahead.toFixed(2)}</span>`;
            div.onclick = () => { bzSelected.clear(); bzSelected.add(idx); updateBzPanel(); draw(); };
            list.appendChild(div);
        }
    }
}

function updateBzField(field) {
    if (bzSelected.size !== 1) return;
    const idx = [...bzSelected][0]; pushUndo();
    if (field==='x') bezierAnchors[idx].x = parseFloat(document.getElementById('bz-x').value);
    if (field==='y') bezierAnchors[idx].y = parseFloat(document.getElementById('bz-y').value);
    if (field==='speed') bezierAnchors[idx].speed = parseFloat(document.getElementById('bz-speed').value);
    if (field==='lookahead') bezierAnchors[idx].lookahead = parseFloat(document.getElementById('bz-la').value);
    dirty = true; regenerateBezier(); draw(); updateStatus();
}

function applyBzBulk() {
    if (bzSelected.size === 0) return;
    const sv = document.getElementById('bz-bulk-speed').value;
    const lv = document.getElementById('bz-bulk-la').value;
    if (!sv && !lv) return; pushUndo();
    for (const idx of bzSelected) {
        if (sv) bezierAnchors[idx].speed = parseFloat(sv);
        if (lv) bezierAnchors[idx].lookahead = parseFloat(lv);
    }
    dirty = true; regenerateBezier(); updateBzPanel(); draw(); updateStatus();
}

function deleteBzSelected() {
    if (bzSelected.size === 0) return;
    if (bezierAnchors.length - bzSelected.size < 2) { alert('Need at least 2 anchors.'); return; }
    if (!confirm(`Delete ${bzSelected.size} anchor(s)?`)) return;
    pushUndo();
    for (const idx of [...bzSelected].sort((a,b)=>b-a)) bezierAnchors.splice(idx,1);
    bzSelected.clear(); dirty = true;
    regenerateBezier(); updateBzPanel(); draw(); updateStatus();
}

function addBezierAnchorNear(wx, wy) {
    if (bezierAnchors.length < 2) return;
    let bestIdx = 0, bestDist = Infinity;
    for (let i = 0; i < bezierAnchors.length; i++) {
        const a = bezierAnchors[i], b = bezierAnchors[(i+1)%bezierAnchors.length];
        const mx = (a.x+b.x)/2, my = (a.y+b.y)/2;
        const d = (wx-mx)**2+(wy-my)**2;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    const a = bezierAnchors[bestIdx], b = bezierAnchors[(bestIdx+1)%bezierAnchors.length];
    const newA = {
        x: wx, y: wy,
        hix: (a.x-wx)*0.3, hiy: (a.y-wy)*0.3,
        hox: (b.x-wx)*0.3, hoy: (b.y-wy)*0.3,
        speed: (a.speed+b.speed)/2,
        lookahead: (a.lookahead+b.lookahead)/2,
    };
    pushUndo();
    bezierAnchors.splice(bestIdx+1, 0, newA);
    document.getElementById('bz-count').value = bezierAnchors.length;
    document.getElementById('bz-count-val').textContent = bezierAnchors.length;
    bzSelected.clear(); bzSelected.add(bestIdx+1);
    dirty = true; regenerateBezier(); updateBzPanel(); draw(); updateStatus();
}

function deleteCpSelected() {
    if (cpSelected.size === 0) return;
    if (controlPoints.length - cpSelected.size < 3) {
        alert('Need at least 3 control points for a spline.');
        return;
    }
    if (!confirm(`Delete ${cpSelected.size} control point(s)?`)) return;
    pushUndo();
    for (const idx of [...cpSelected].sort((a, b) => b - a)) controlPoints.splice(idx, 1);
    cpSelected.clear(); dirty = true;
    regenerateSpline(); updateCpPanel(); draw(); updateStatus();
}

function addControlPointNear(wx, wy) {
    // Find the spline segment closest to the click and insert a new CP there
    if (controlPoints.length < 3) return;
    let bestIdx = 0, bestDist = Infinity;
    for (let i = 0; i < controlPoints.length; i++) {
        const a = controlPoints[i], b = controlPoints[(i + 1) % controlPoints.length];
        // Project click onto segment a→b
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const d = (wx - mx) * (wx - mx) + (wy - my) * (wy - my);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    const a = controlPoints[bestIdx], b = controlPoints[(bestIdx + 1) % controlPoints.length];
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
        mode === 'spline' ? `CPs: ${controlPoints.length} | WPs: ${awp.length}` :
        mode === 'bezier' ? `Anchors: ${bezierAnchors.length} | WPs: ${awp.length}` :
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
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
        dragType = 'pan';
        dragStart = {x: sx, y: sy, panX, panY};
        canvas.style.cursor = 'grabbing';
        return;
    }
    if (e.button !== 0) return;

    // Bezier handle hit test (check before anchor hit test)
    if (mode === 'bezier') {
        const hh = hitTestBzHandle(sx, sy, 14);
        if (hh) {
            bzDragType = hh.type; bzDragIdx = hh.idx;
            dragType = 'bz-handle';
            pushUndo();
            draw(); return;
        }
    }

    const hitRadius = mode === 'bezier' ? 16 : (mode === 'spline' ? 16 : 12);
    const hit = hitTest(sx, sy, hitRadius);
    const sel = mode === 'points' ? selected : (mode === 'spline' ? cpSelected : bzSelected);
    const panelFn = mode === 'points' ? updatePanel : (mode === 'spline' ? updateCpPanel : updateBzPanel);

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
        const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
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
    const hitRadius = mode === 'bezier' ? 16 : (mode === 'spline' ? 16 : 12);
    const hov = hitTest(sx, sy, hitRadius);

    if (mode === 'points') hoveredIdx = hov;
    else if (mode === 'spline') cpHoveredIdx = hov;
    else { bzHoveredIdx = hov; bzHoveredType = hov >= 0 ? 'anchor' : null; }

    if (hov >= 0 && dragType !== 'point' && dragType !== 'bz-handle') {
        const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
        const prefix = mode === 'points' ? `#${hov}` : (mode === 'spline' ? `C${hov}` : `B${hov}`);
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

    if (dragType === 'bz-handle') {
        const a = bezierAnchors[bzDragIdx];
        const hdx = wx - a.x, hdy = wy - a.y;
        if (bzDragType === 'handle-out') {
            a.hox = hdx; a.hoy = hdy;
            if (bzSmooth) { const len = Math.hypot(a.hix,a.hiy); const nlen = Math.hypot(hdx,hdy)||1; a.hix = -hdx*len/nlen; a.hiy = -hdy*len/nlen; }
        } else {
            a.hix = hdx; a.hiy = hdy;
            if (bzSmooth) { const len = Math.hypot(a.hox,a.hoy); const nlen = Math.hypot(hdx,hdy)||1; a.hox = -hdx*len/nlen; a.hoy = -hdy*len/nlen; }
        }
        dirty = true; regenerateBezier(); draw(); updateStatus(); return;
    }

    if (dragType === 'point') {
        const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
        for (const off of dragPointOffsets) {
            arr[off.idx].x = wx + off.dx;
            arr[off.idx].y = wy + off.dy;
        }
        dirty = true;
        if (mode === 'spline') regenerateSpline();
        if (mode === 'bezier') regenerateBezier();
        const panelFn = mode === 'points' ? updatePanel : (mode === 'spline' ? updateCpPanel : updateBzPanel);
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
            const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
            const sel = mode === 'points' ? selected : (mode === 'spline' ? cpSelected : bzSelected);
            for (let i = 0; i < arr.length; i++) {
                const [wx, wy] = worldToScreen(arr[i].x, arr[i].y);
                if (wx >= x1 && wx <= x2 && wy >= y1 && wy <= y2) sel.add(i);
            }
        }
        const panelFn = mode === 'points' ? updatePanel : (mode === 'spline' ? updateCpPanel : updateBzPanel);
        panelFn();
    }

    if (dragType === 'point') {
        const panelFn = mode === 'points' ? updatePanel : (mode === 'spline' ? updateCpPanel : updateBzPanel);
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
        const arr = mode === 'points' ? waypoints : (mode === 'spline' ? controlPoints : bezierAnchors);
        const sel = mode === 'points' ? selected : (mode === 'spline' ? cpSelected : bzSelected);
        for (let i = 0; i < arr.length; i++) sel.add(i);
        const panelFn = mode === 'points' ? updatePanel : (mode === 'spline' ? updateCpPanel : updateBzPanel);
        panelFn(); draw();
    }
    if (e.key === 'Escape') {
        selected.clear(); cpSelected.clear(); bzSelected.clear();
        updatePanel(); updateCpPanel(); updateBzPanel(); draw();
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && document.activeElement.tagName !== 'INPUT') {
        if (mode === 'points' && selected.size > 0) deleteSelected();
        if (mode === 'spline' && cpSelected.size > 0) deleteCpSelected();
        if (mode === 'bezier' && bzSelected.size > 0) deleteBzSelected();
    }
    if (e.key === 'Tab') {
        e.preventDefault();
        const modes = ['points', 'spline', 'bezier'];
        setMode(modes[(modes.indexOf(mode) + 1) % 3]);
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

    // Set initial wp density to match loaded waypoint count
    wpDensity = waypoints.length;
    bzDensity = waypoints.length;
    document.getElementById('wp-density').value = wpDensity;
    document.getElementById('wp-density-val').textContent = wpDensity;
    document.getElementById('bz-density').value = bzDensity;
    document.getElementById('bz-density-val').textContent = bzDensity;

    window.addEventListener('resize', resize);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, {passive: false});
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('dblclick', (e) => {
        if (mode !== 'spline' && mode !== 'bezier') return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
        if (hitTest(sx, sy, 16) >= 0) return;
        const [wx, wy] = screenToWorld(sx, sy);
        if (mode === 'spline') addControlPointNear(wx, wy);
        else addBezierAnchorNear(wx, wy);
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
            html = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        elif parsed.path == "/api/data":
            data = {
                "mapPng": MAP_PNG_B64,
                "mapMeta": MAP_META,
                "waypoints": [{"x": w[0], "y": w[1], "speed": w[2], "lookahead": w[3]} for w in WAYPOINTS],
                "csvPath": os.path.basename(CSV_PATH),
                "mapYaml": os.path.basename(MAP_YAML_PATH),
            }
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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
