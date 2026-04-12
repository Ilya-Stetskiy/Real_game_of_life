
"""
Run:
  python -m pip install -r requirements_pywebview.txt
  python gen_cells_app.py
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trackmate_to_cells import convert as trackmate_convert


DEFAULT_SPOTS = "H2BmCherry_timelapse_spots.csv"
DEFAULT_EDGES = "H2BmCherry_timelapse_edges.csv"
DEFAULT_TRACKS = "H2BmCherry_timelapse_tracks.csv"
DEFAULT_OUTDIR = "out_cells"


def _stable_color_rgba(cell_id: str) -> str:
    # FNV-1a 32-bit -> hue
    h = 2166136261
    for b in cell_id.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    hue = (h % 360) / 360.0

    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"


def _load_cell_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame", "x", "y", "area", "cell_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Expected cell_series.csv")

    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype("int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.dropna(subset=["x", "y", "area"]).reset_index(drop=True)

    # radius in microns
    if "radius" in df.columns:
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
        df["radius"] = df["radius"].fillna(np.sqrt(df["area"].clip(lower=0) / math.pi))
    else:
        df["radius"] = np.sqrt(df["area"].clip(lower=0) / math.pi)

    # top-left coord system: shift to (0,0)
    x0 = float(df["x"].min())
    y0 = float(df["y"].min())
    df["x_plot"] = df["x"] - x0
    df["y_plot"] = df["y"] - y0
    return df.sort_values(["frame", "cell_id"]).reset_index(drop=True)


@dataclass
class Dataset:
    series_path: str
    frames: List[int]
    bounds: Dict[str, float]
    by_frame: Dict[int, List[Dict[str, Any]]]


def _build_dataset(series_path: str) -> Dataset:
    df = _load_cell_series(series_path)
    frames = sorted(df["frame"].unique().astype(int).tolist())
    bounds = {
        "x_min": float(df["x_plot"].min()),
        "x_max": float(df["x_plot"].max()),
        "y_min": float(df["y_plot"].min()),
        "y_max": float(df["y_plot"].max()),
    }

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for fr, g in df.groupby("frame", sort=False):
        fr_i = int(fr)
        pts = []
        for row in g.itertuples(index=False):
            cell_id = str(row.cell_id)
            pts.append(
                {
                    "cell_id": cell_id,
                    "x": float(row.x_plot),
                    "y": float(row.y_plot),
                    "r": float(row.radius),
                    "area": float(row.area),
                    "color": _stable_color_rgba(cell_id),
                }
            )
        by_frame[fr_i] = pts

    return Dataset(series_path=series_path, frames=frames, bounds=bounds, by_frame=by_frame)

def _rgba_to_rgba_tuple(rgba: str) -> tuple[int, int, int, int]:
    # rgba(12,34,56,0.85)
    s = rgba.strip().lower()
    if not s.startswith("rgba"):
        return (255, 255, 255, 255)
    inside = s[s.find("(") + 1 : s.rfind(")")]
    parts = [p.strip() for p in inside.split(",")]
    if len(parts) != 4:
        return (255, 255, 255, 255)
    r = int(float(parts[0]))
    g = int(float(parts[1]))
    b = int(float(parts[2]))
    a = float(parts[3])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, min(255, int(a * 255))))


def _render_gif(dataset: Dataset, out_path: str, fps: int = 15, width: int = 1000, height: int = 800) -> None:
    """
    Render a GIF from the same data the viewer uses.
    Note: this is a *data-driven* renderer (PIL), not a screenshot of the UI.
    """
    from PIL import Image, ImageDraw

    bounds = dataset.bounds
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    pad = 0.04
    span_x = (x_max - x_min) * (1 + pad) or 1.0
    span_y = (y_max - y_min) * (1 + pad) or 1.0
    s = min(width / span_x, height / span_y)
    ox = 0.5 * (width - s * (x_max - x_min))
    oy = 0.5 * (height - s * (y_max - y_min))

    frames: List[Image.Image] = []
    for fr in dataset.frames:
        img = Image.new("RGBA", (width, height), (11, 13, 18, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        pts = dataset.by_frame.get(fr, [])
        for p in pts:
            x = ox + s * (p["x"] - x_min)
            y = oy + s * (p["y"] - y_min)
            r = max(1.5, s * p["r"])
            fill = _rgba_to_rgba_tuple(p["color"])
            outline = (0, 0, 0, 120)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=1)

        # small header
        draw.text((10, 10), f"frame {fr}", fill=(233, 236, 245, 200))
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

    duration_ms = int(round(1000 / max(int(fps), 1)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


HTML = r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Gen Cells App</title>
    <style>
      :root{
        --bg: #0b0d12;
        --panel: #121623;
        --panel2: #0f1420;
        --text: #e9ecf5;
        --muted: #aab2c5;
        --border: #222a3d;
        --accent: #7aa2ff;
      }
      html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }
      .wrap { display: grid; grid-template-columns: 380px 1fr; height: 100%; }
      .left { padding: 14px; border-right: 1px solid var(--border); background: linear-gradient(180deg, var(--panel), var(--panel2)); overflow:auto; }
      .title { font-size: 18px; font-weight: 650; margin: 0 0 10px 0; }
      .card { border:1px solid var(--border); border-radius: 14px; padding: 12px; background: rgba(255,255,255,0.03); margin-bottom: 12px;}
      .status { font-size: 12px; color: var(--muted); padding: 10px; border: 1px solid var(--border); border-radius: 12px; background: rgba(255,255,255,0.03); word-break: break-word; }
      .row { display:flex; gap:10px; align-items:center; margin: 10px 0; flex-wrap: wrap;}
      button { background: rgba(255,255,255,0.06); color: var(--text); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; cursor:pointer; }
      button:hover { border-color: rgba(122,162,255,0.55); }
      button.primary { border-color: rgba(122,162,255,0.85); }
      input[type="text"] { width: 100%; background: rgba(255,255,255,0.06); color: var(--text); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
      select, input[type="range"] { width: 100%; }
      select { background: rgba(255,255,255,0.06); color: var(--text); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
      .label { font-size: 12px; color: var(--muted); margin-top: 10px;}
      .frameText { font-size: 13px; margin-top: 6px; color: var(--text); }
      .right { position: relative; }
      canvas { width: 100%; height: 100%; display:block; background: #0b0d12; }
      .tooltip { position:absolute; pointer-events:none; padding: 8px 10px; border-radius: 10px; border:1px solid rgba(255,255,255,0.16); background: rgba(10,12,18,0.92); color: var(--text); font-size: 12px; display:none; white-space: pre; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="left">
        <div class="title">Gen Cells App</div>

        <div class="card">
          <div class="label">Convert TrackMate CSV → out_cells</div>
          <div id="convStatus" class="status">Ready.</div>
          <div class="label">Input folder (contains spots/edges/tracks)</div>
          <input id="inDir" type="text" class="mono" placeholder="(auto / choose folder)"/>
          <div class="row">
            <button id="btnPickDir">Choose folder…</button>
            <button id="btnConvert" class="primary">Convert</button>
          </div>
          <div class="label">Output folder</div>
          <input id="outDir" type="text" class="mono" value="out_cells"/>
          <div class="status">Expected filenames inside folder: <span class="mono">*_spots.csv</span>, <span class="mono">*_edges.csv</span>, <span class="mono">*_tracks.csv</span></div>
        </div>

        <div class="card">
          <div class="label">Viewer</div>
          <div id="viewStatus" class="status">Loading…</div>
          <div class="row">
            <button id="btnLoadSeries">Load cell_series.csv…</button>
            <button id="btnExportGif">Export GIF…</button>
            <button id="btnPlay">Play</button>
            <button id="btnPause">Pause</button>
          </div>

          <div class="label">Speed</div>
          <select id="speed">
            <option value="1">x1</option>
            <option value="2">x2</option>
            <option value="3">x3</option>
            <option value="4">x4</option>
            <option value="8">x8</option>
          </select>

          <div class="label">Frame</div>
          <input id="slider" type="range" min="0" max="0" step="1" value="0"/>
          <div id="frameText" class="frameText"></div>
          <div class="status">Hover a cell to see id. Top-left is (0,0). Y grows down.</div>
        </div>
      </div>

      <div class="right">
        <canvas id="c"></canvas>
        <div id="tip" class="tooltip"></div>
      </div>
    </div>

    <script>
      const canvas = document.getElementById('c');
      const ctx = canvas.getContext('2d');
      const slider = document.getElementById('slider');
      const frameText = document.getElementById('frameText');
      const convStatus = document.getElementById('convStatus');
      const viewStatus = document.getElementById('viewStatus');
      const tipEl = document.getElementById('tip');
      const inDir = document.getElementById('inDir');
      const outDir = document.getElementById('outDir');

      let dataset = null;   // {frames, bounds, by_frame, series_path}
      let playing = false;
      let timer = null;

      function resizeCanvas(){
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr,0,0,dpr,0,0); // draw in CSS pixels
        draw();
      }
      window.addEventListener('resize', resizeCanvas);

      function scaleForBounds(bounds){
        const w = canvas.getBoundingClientRect().width;
        const h = canvas.getBoundingClientRect().height;
        const pad = 0.04;
        const spanX = (bounds.x_max - bounds.x_min) * (1 + pad);
        const spanY = (bounds.y_max - bounds.y_min) * (1 + pad);
        const s = Math.min(w / (spanX || 1), h / (spanY || 1));
        const ox = 0.5 * (w - s * (bounds.x_max - bounds.x_min));
        const oy = 0.5 * (h - s * (bounds.y_max - bounds.y_min));
        return {s, ox, oy};
      }

      function currentFrameIndex(){
        return parseInt(slider.value || '0', 10);
      }

      function draw(){
        const w = canvas.getBoundingClientRect().width;
        const h = canvas.getBoundingClientRect().height;
        ctx.clearRect(0,0,w,h);
        if(!dataset){ return; }
        const idx = currentFrameIndex();
        const frames = dataset.frames;
        const fr = frames[Math.max(0, Math.min(idx, frames.length-1))];
        frameText.textContent = `Frame ${fr} (${idx+1}/${frames.length})`;

        const pts = dataset.by_frame[String(fr)] || [];
        const tr = scaleForBounds(dataset.bounds);

        // draw grid light
        ctx.save();
        ctx.globalAlpha = 0.10;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        const step = 80;
        for(let x=0; x<w; x+=step){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
        for(let y=0; y<h; y+=step){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
        ctx.restore();

        for(const p of pts){
          const x = tr.ox + tr.s * (p.x - dataset.bounds.x_min);
          const y = tr.oy + tr.s * (p.y - dataset.bounds.y_min);
          const r = Math.max(1.5, tr.s * p.r);
          ctx.beginPath();
          ctx.arc(x,y,r,0,Math.PI*2);
          ctx.fillStyle = p.color;
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = 'rgba(0,0,0,0.45)';
          ctx.stroke();
        }
      }

      function stopTimer(){
        playing = false;
        if(timer){ clearInterval(timer); timer=null; }
      }

      function startTimer(){
        stopTimer();
        playing = true;
        const speed = parseInt(document.getElementById('speed').value, 10) || 1;
        const intervalMs = Math.max(10, Math.floor(1000/15/speed));
        timer = setInterval(() => {
          if(!dataset) return;
          let v = currentFrameIndex() + 1;
          if(v >= dataset.frames.length) v = 0;
          slider.value = String(v);
          draw();
        }, intervalMs);
      }

      slider.addEventListener('input', () => { draw(); });
      document.getElementById('speed').addEventListener('change', () => { if(playing) startTimer(); });
      document.getElementById('btnPlay').addEventListener('click', () => startTimer());
      document.getElementById('btnPause').addEventListener('click', () => stopTimer());

      canvas.addEventListener('mousemove', (ev) => {
        if(!dataset){ return; }
        const rect = canvas.getBoundingClientRect();
        const mx = ev.clientX - rect.left;
        const my = ev.clientY - rect.top;
        const idx = currentFrameIndex();
        const frames = dataset.frames;
        const fr = frames[Math.max(0, Math.min(idx, frames.length-1))];
        const pts = dataset.by_frame[String(fr)] || [];
        const tr = scaleForBounds(dataset.bounds);

        let hit = null;
        for(const p of pts){
          const x = tr.ox + tr.s * (p.x - dataset.bounds.x_min);
          const y = tr.oy + tr.s * (p.y - dataset.bounds.y_min);
          const r = Math.max(1.5, tr.s * p.r);
          const dx = mx - x, dy = my - y;
          if(dx*dx + dy*dy <= r*r){
            hit = p;
            break;
          }
        }
        if(hit){
          tipEl.style.display = 'block';
          tipEl.style.left = (mx + 14) + 'px';
          tipEl.style.top = (my + 14) + 'px';
          tipEl.textContent = `cell_id: ${hit.cell_id}\\narea: ${hit.area.toFixed(2)} µm²`;
        } else {
          tipEl.style.display = 'none';
        }
      });
      canvas.addEventListener('mouseleave', () => { tipEl.style.display='none'; });

      document.getElementById('btnPickDir').addEventListener('click', async () => {
        const d = await pywebview.api.pick_input_dir();
        if(d){ inDir.value = d; convStatus.textContent = 'Selected: ' + d; }
      });

      document.getElementById('btnConvert').addEventListener('click', async () => {
        try{
          convStatus.textContent = 'Converting…';
          const inputDir = inDir.value || '';
          const out = outDir.value || 'out_cells';
          const res = await pywebview.api.convert_trackmate(inputDir, out);
          convStatus.textContent = res.message;
          if(res.ok && res.cell_series_path){
            dataset = await pywebview.api.load_series(res.cell_series_path);
            slider.max = String(Math.max(0, dataset.frames.length-1));
            slider.value = '0';
            viewStatus.textContent = `Loaded: ${dataset.series_path} | frames=${dataset.frames.length}`;
            stopTimer();
            draw();
          }
        } catch(e){
          convStatus.textContent = 'Convert failed: ' + e;
        }
      });

      document.getElementById('btnLoadSeries').addEventListener('click', async () => {
        try{
          const path = await pywebview.api.pick_cell_series();
          if(!path){ return; }
          dataset = await pywebview.api.load_series(path);
          slider.max = String(Math.max(0, dataset.frames.length-1));
          slider.value = '0';
          viewStatus.textContent = `Loaded: ${dataset.series_path} | frames=${dataset.frames.length}`;
          stopTimer();
          draw();
        } catch(e){
          viewStatus.textContent = 'Load failed: ' + e;
        }
      });

      document.getElementById('btnExportGif').addEventListener('click', async () => {
        try{
          if(!dataset){ viewStatus.textContent = 'No data loaded yet.'; return; }
          viewStatus.textContent = 'Exporting GIF…';
          const ok = await pywebview.api.export_gif();
          viewStatus.textContent = ok ? 'GIF exported.' : 'GIF export cancelled.';
        } catch(e){
          viewStatus.textContent = 'Export failed: ' + e;
        }
      });

      async function init(){
        try{
          const state = await pywebview.api.get_initial_state();
          inDir.value = state.input_dir || '';
          outDir.value = state.out_dir || 'out_cells';
          if(state.cell_series_path){
            dataset = await pywebview.api.load_series(state.cell_series_path);
            slider.max = String(Math.max(0, dataset.frames.length-1));
            slider.value = '0';
            viewStatus.textContent = `Loaded: ${dataset.series_path} | frames=${dataset.frames.length}`;
          } else {
            viewStatus.textContent = 'No cell_series.csv yet. Convert or load.';
          }
          resizeCanvas();
          draw();
        }catch(e){
          viewStatus.textContent = 'Init failed: ' + e;
          resizeCanvas();
        }
      }
      init();
    </script>
  </body>
</html>
"""


class Api:
    def __init__(self) -> None:
        self._dataset: Optional[Dataset] = None
        self._last_series_path: Optional[str] = None

    def get_initial_state(self) -> dict:
        inp = os.path.abspath(".")
        out_dir = DEFAULT_OUTDIR
        cand = os.path.join(out_dir, "cell_series.csv")
        cell_series = os.path.abspath(cand) if os.path.exists(cand) else None
        return {"input_dir": inp, "out_dir": out_dir, "cell_series_path": cell_series}

    def pick_input_dir(self) -> Optional[str]:
        import webview

        res = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if not res:
            return None
        return res[0]

    def pick_cell_series(self) -> Optional[str]:
        import webview

        res = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("cell_series (*.csv)", "CSV Files (*.csv)", "All files (*.*)"),
        )
        if not res:
            return None
        path = res[0]
        base = os.path.basename(path).lower()
        if "cell_lineage" in base:
            cand = os.path.join(os.path.dirname(path), "cell_series.csv")
            if os.path.exists(cand):
                path = cand
        return path

    def convert_trackmate(self, input_dir: str, out_dir: str) -> dict:
        # Resolve inputs
        input_dir = os.path.abspath(input_dir or ".")
        out_dir = os.path.abspath(out_dir or DEFAULT_OUTDIR)
        os.makedirs(out_dir, exist_ok=True)

        # Find TrackMate csv files in input dir
        spots = None
        edges = None
        tracks = None
        for name in os.listdir(input_dir):
            low = name.lower()
            full = os.path.join(input_dir, name)
            if not os.path.isfile(full) or not low.endswith(".csv"):
                continue
            if low.endswith("_spots.csv"):
                spots = full
            elif low.endswith("_edges.csv"):
                edges = full
            elif low.endswith("_tracks.csv"):
                tracks = full

        # Fallback to default names
        spots = spots or os.path.join(input_dir, DEFAULT_SPOTS)
        edges = edges or os.path.join(input_dir, DEFAULT_EDGES)
        tracks = tracks or os.path.join(input_dir, DEFAULT_TRACKS)

        if not os.path.exists(spots) or not os.path.exists(edges):
            return {
                "ok": False,
                "message": f"Missing input CSV. Expected spots/edges in {input_dir}. Found spots={os.path.exists(spots)} edges={os.path.exists(edges)}",
                "cell_series_path": None,
            }

        # Run conversion
        trackmate_convert(spots_csv=spots, edges_csv=edges, tracks_csv=tracks if os.path.exists(tracks) else None, outdir=out_dir)

        cell_series = os.path.join(out_dir, "cell_series.csv")
        if not os.path.exists(cell_series):
            return {"ok": False, "message": "Conversion finished but cell_series.csv not found.", "cell_series_path": None}

        return {
            "ok": True,
            "message": f"Converted. Saved to {out_dir}",
            "cell_series_path": os.path.abspath(cell_series),
        }

    def load_series(self, path: str) -> dict:
        self._dataset = _build_dataset(path)
        self._last_series_path = os.path.abspath(path)
        return {
            "series_path": self._dataset.series_path,
            "frames": self._dataset.frames,
            "bounds": self._dataset.bounds,
            "by_frame": self._dataset.by_frame,
        }

    def export_gif(self) -> bool:
        """
        Export current dataset as GIF (asks user for save location).
        """
        if self._dataset is None:
            raise ValueError("No dataset loaded")

        import webview

        default_name = "cell_motion.gif"
        if self._last_series_path:
            base = os.path.basename(self._last_series_path)
            default_name = os.path.splitext(base)[0] + ".gif"

        res = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_name,
            file_types=("GIF Files (*.gif)", "All files (*.*)"),
        )
        if not res:
            return False
        out_path = res
        if isinstance(out_path, list):
            out_path = out_path[0]

        _render_gif(self._dataset, str(out_path), fps=15, width=1000, height=800)
        return True


def main() -> None:
    try:
        import webview
    except Exception as e:
        raise SystemExit(
            "pywebview is not installed. Install with:\n"
            "  python -m pip install -r requirements_pywebview.txt\n"
            f"Error: {e}"
        )

    api = Api()
    webview.create_window("Gen Cells App", html=HTML, js_api=api, width=1280, height=860)
    webview.start(debug=False)


if __name__ == "__main__":
    main()

