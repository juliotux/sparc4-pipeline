"""
sparc4_plot_saturation.py
=========================
SPARC4 real-time saturation monitor — ZeroMQ subscriber.

Plots peak counts vs local time for the target and each comparison star.
A horizontal line marks the saturation level. In POLAR mode each beam
is plotted separately (beam-0 solid, beam-1 dashed).

Usage
-----
::

    python sparc4_plot_saturation.py --db /data/SPARC4/reduced
    python sparc4_plot_saturation.py --host 192.168.1.10 --db /data/SPARC4/reduced

Author
------
Eder Martioli <martioli@lna.br>
Laboratório Nacional de Astrofísica — LNA/MCTI
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as _mticker

try:
    import zmq
except ImportError:
    raise SystemExit("pyzmq is required:  pip install pyzmq")

try:
    from sparc4_astro_utils import (compute_sun_events, add_twilight_lines,
                                     night_xlim, date_from_jd,
                                     OPD_LON, OPD_LAT, OPD_ALT)
    _HAS_ASTRO = True
except ImportError:
    _HAS_ASTRO = False
    OPD_LON = -45.5825
    OPD_LAT = -22.5344
    OPD_ALT = 1864.0

CH_COLOR     = {1: "darkblue", 2: "darkgreen", 3: "darkorange", 4: "darkred"}
CH_BAND      = {1: "g",        2: "r",         3: "i",          4: "z"}
TARGET_COLOR = "green"
COMP_COLORS  = ["red", "darkorange", "purple", "saddlebrown", "crimson", "teal"]


def _jd_to_hours(jd_arr: np.ndarray, utc_offset_h: float) -> np.ndarray:
    if len(jd_arr) == 0:
        return np.array([])
    return ((jd_arr + 0.5 + utc_offset_h / 24.0) % 1.0) * 24.0

def _format_time_axis(ax) -> None:
    def _fmt(x, pos):
        x = x % 24.0; h = int(x); m = int(round((x-h)*60)) % 60
        return f"{h:02d}:{m:02d}"
    ax.xaxis.set_major_formatter(_mticker.FuncFormatter(_fmt))
    ax.xaxis.set_major_locator(_mticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(_mticker.MultipleLocator(1/12))
    ax.tick_params(axis="x", which="minor", length=3)

def _sync_xlim(axes_list: list) -> None:
    """Set x-axis limits from the first panel with data, apply to all others."""
    ref_xlim = None
    for ax in axes_list:
        lines_with_data = [l for l in ax.get_lines()
                           if len(l.get_xdata()) > 1
                           and not getattr(l, "_is_twilight", False)]
        if lines_with_data:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
            ref_xlim = ax.get_xlim()
            break
    if ref_xlim is None:
        return
    for ax in axes_list:
        ax.set_xlim(ref_xlim)


def _load_channel_data(db_path: str, history: int) -> tuple:
    """Load selection and peak history from one channel's own DB.

    In POLAR mode both beam-0 and beam-1 peaks are returned separately.
    Returns (target_id, comp_ids, beam_map, data_b0, data_b1).
    """
    if not os.path.isfile(db_path):
        return None, [], {}, {}, {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        sel_rows = conn.execute(
            "SELECT role, ref_id, beam FROM selection ORDER BY rank"
        ).fetchall()
        target_id, comp_ids, beam_map = None, [], {}
        for row in sel_rows:
            rid  = row["ref_id"]
            beam = row["beam"]
            if beam is not None:
                beam_map[rid] = beam
            if row["role"] == "target": target_id = rid
            else: comp_ids.append(rid)

        is_polar = bool(beam_map)
        all_ids  = ([target_id] if target_id is not None else []) + comp_ids

        def _fetch_peaks(rid):
            pts = conn.execute(
                "SELECT f.jd, s.peak FROM sources s "
                "JOIN frames f ON f.id = s.frame_id "
                "WHERE s.ref_id = ? AND f.jd IS NOT NULL "
                "  AND f.bad = 0 AND s.peak IS NOT NULL "
                "ORDER BY f.jd", (rid,)
            ).fetchall()[-history:]
            if pts:
                return (np.array([r["jd"]   for r in pts]),
                        np.array([r["peak"] for r in pts]))
            return None

        if is_polar:
            b0_ids = [r for r in all_ids if beam_map.get(r) == 0]
            b1_ids = [r for r in all_ids if beam_map.get(r) == 1]
            b1_for_b0 = {b0: b1_ids[i] for i, b0 in enumerate(b0_ids)
                         if i < len(b1_ids)}
            data_b0, data_b1 = {}, {}
            for b0 in b0_ids:
                res = _fetch_peaks(b0)
                if res: data_b0[b0] = res
                b1 = b1_for_b0.get(b0)
                if b1 is not None:
                    res1 = _fetch_peaks(b1)
                    if res1: data_b1[b0] = res1
            target_id = b0_ids[0] if b0_ids else target_id
            comp_ids  = b0_ids[1:]
        else:
            data_b0, data_b1 = {}, {}
            for rid in all_ids:
                res = _fetch_peaks(rid)
                if res: data_b0[rid] = res

        conn.close()
        n = sum(len(v[0]) for v in data_b0.values())
        mode = "POLAR" if is_polar else "PHOT"
        print(f"  {os.path.basename(db_path)} [{mode}]: "
              f"target={target_id}  comps={comp_ids}  pts={n}")
        return target_id, comp_ids, beam_map, data_b0, data_b1
    except Exception as exc:
        print(f"  [warn] {db_path}: {exc}")
        return None, [], {}, {}, {}


def run(host: str, port: int, channels: List[int],
        db_dir: Optional[str], sat_level: float, history: int,
        interval: float, utc_offset: float,
        twilight: bool = False,
        night_xlim_flag: bool = False,
        obs_lon: float = OPD_LON,
        obs_lat: float = OPD_LAT,
        obs_alt: float = OPD_ALT,
        obs_date: str = "",
        save_png: str = "") -> None:

    ref_channel = channels[0]

    # Per-channel selections — each channel has its own ref_ids
    per_ch_target: Dict[int, Optional[int]] = {ch: None for ch in channels}
    per_ch_comps:  Dict[int, List[int]]     = {ch: []   for ch in channels}

    # Buffers keyed by (ref_id, beam): 0=ordinary, 1=extraordinary
    bufs:      Dict[int, Dict] = {ch: {} for ch in channels}
    peak_bufs: Dict[int, Dict] = {ch: {} for ch in channels}

    def _ensure(ch: int, rid: int, beam: int = 0) -> None:
        key = (rid, beam)
        if key not in bufs[ch]:
            bufs[ch][key]      = deque(maxlen=history)
            peak_bufs[ch][key] = deque(maxlen=history)

    if db_dir:
        print("[sat] Loading historical data...")
        for ch in channels:
            path = os.path.join(db_dir, f"monitor_ch{ch}.db")
            t, comps, _bm, data_b0, data_b1 = _load_channel_data(path, history)
            per_ch_target[ch] = t
            per_ch_comps[ch]  = comps
            for rid, (jds, peaks) in data_b0.items():
                _ensure(ch, rid, beam=0)
                bufs[ch][(rid, 0)].extend(jds)
                peak_bufs[ch][(rid, 0)].extend(peaks)
            for rid, (jds, peaks) in data_b1.items():
                _ensure(ch, rid, beam=1)
                bufs[ch][(rid, 1)].extend(jds)
                peak_bufs[ch][(rid, 1)].extend(peaks)

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    for ch in channels:
        sock.setsockopt_string(zmq.SUBSCRIBE, f"sparc4.ch{ch}")
    sock.setsockopt(zmq.RCVTIMEO, int(interval * 1000))
    print(f"[sat] Subscribed  tcp://{host}:{port}  ch={channels}")

    nch     = len(channels)
    panel_h = min(3.5, 12.0 / nch)
    fig, axs = plt.subplots(nch, 1, figsize=(13, panel_h * nch), squeeze=False)
    axes: Dict[int, plt.Axes] = {ch: axs[i][0] for i, ch in enumerate(channels)}

    fig.patch.set_facecolor("white")
    fig.suptitle(f"SPARC4  —  Peak Counts  (sat={sat_level:.0f})",
                 fontsize=14, fontweight="bold", color="black", y=0.99)
    try:
        fig.canvas.manager.set_window_title("SPARC4 — Saturation")
    except Exception:
        pass

    for idx, ch in enumerate(channels):
        ax = axes[ch]
        ax.set_facecolor("white")
        ax.set_ylabel("Peak (ADU)", fontsize=12)
        ax.axhline(sat_level,       color="darkred",   lw=1.4, ls="--",
                   label=f"Saturation ({sat_level:.0f})")
        ax.axhline(sat_level * 0.7, color="darkorange", lw=1.0, ls=":",
                   label=f"70%  ({sat_level*0.7:.0f})")
        ax.set_ylim(0, sat_level * 1.15)
        ax.grid(True, color="lightgray", ls="--", lw=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors="black", labelsize=11)
        ax.set_title(f"{CH_BAND[ch]}  (ch{ch})", fontsize=12,
                     fontweight="bold", color=CH_COLOR[ch], loc="left")
        _format_time_axis(ax)
        if idx == nch - 1:
            ax.set_xlabel(f"Local time  (UTC{utc_offset:+.0f}h)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    lines: Dict[int, Dict] = {ch: {} for ch in channels}
    last_draw = 0.0
    new_data  = False

    plt.ion()
    fig.show()

    def _redraw() -> None:
        for ch in channels:
            tid  = per_ch_target[ch]
            cids = per_ch_comps[ch]
            ordered = ([tid] if tid is not None else []) + \
                      [r for r in cids if r != tid]
            ax = axes[ch]
            if not ordered:
                ax.set_title(f"{CH_BAND[ch]}  (ch{ch})  — waiting for selection",
                             fontsize=11, fontweight="bold",
                             color=CH_COLOR[ch], loc="left")
                continue

            latest = []
            for rid in ordered:
                beams_present = [b for b in (0, 1)
                                 if (rid, b) in bufs[ch]
                                 and len(bufs[ch][(rid, b)]) > 0]
                if not beams_present:
                    continue
                for b in beams_present:
                    key   = (rid, b)
                    jds   = np.array(bufs[ch][key])
                    peaks = np.array(peak_bufs[ch][key])
                    ht    = _jd_to_hours(jds, utc_offset)

                    is_target = (rid == tid)
                    beam_sfx  = f" B{b}" if len(beams_present) > 1 or b == 1 else ""
                    if is_target:
                        col, marker, lw, ms = TARGET_COLOR, "o", 2.0, 7
                        mfc = TARGET_COLOR
                        lbl = f"Target{beam_sfx}  (src {rid})"
                    else:
                        ci  = cids.index(rid) if rid in cids else 0
                        col = COMP_COLORS[ci % len(COMP_COLORS)]
                        marker, lw, ms, mfc = "s", 1.5, 5, "white"
                        lbl = f"C{ci+1}{beam_sfx}  (src {rid})"

                    line_key = (rid, b)
                    if line_key not in lines[ch]:
                        ls_ = "-" if b == 0 else "--"
                        l, = ax.plot(ht, peaks, marker=marker, ls=ls_,
                                     color=col, markerfacecolor=mfc,
                                     markeredgecolor=col,
                                     lw=lw, ms=ms, zorder=3, label=lbl)
                        lines[ch][line_key] = l
                    else:
                        lines[ch][line_key].set_data(ht, peaks)

                    if len(peaks):
                        pk   = float(peaks[-1])
                        warn = " ⚠SAT" if pk >= sat_level else ""
                        pfx  = "T" if is_target else f"C{cids.index(rid)+1 if rid in cids else 0}"
                        latest.append(f"{pfx}B{b}={pk:.0f}{warn}")

            ax.relim(); ax.autoscale_view(scaley=False)
            handles, lbls = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, lbls, loc="upper left", fontsize=9,
                          framealpha=0.9, edgecolor="gray")
            suffix = ("  |  " + "  ".join(latest)) if latest else ""
            ax.set_title(f"{CH_BAND[ch]}  (ch{ch}){suffix}",
                         fontsize=11, fontweight="bold",
                         color=CH_COLOR[ch], loc="left")

        _sync_xlim(list(axes.values()))
        if _HAS_ASTRO and twilight and sun_evts is not None:
            for _ax in axes.values():
                for _l in list(_ax.get_lines()):
                    if getattr(_l, "_is_twilight", False):
                        _l.remove()
            add_twilight_lines(list(axes.values()), sun_evts, legend_ax_index=0)
            for _ax in axes.values():
                for _l in _ax.get_lines():
                    if _l.get_label() in ("Sunset", "Sunrise",
                        "Civil twil.", "Nautical twil.", "Astron. twil."):
                        _l._is_twilight = True

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if save_png:
            import pathlib
            pathlib.Path(save_png).mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(str(pathlib.Path(save_png) / "saturation.png"),
                            dpi=110, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception:
                pass

    # Connect xlim_changed callbacks for interactive pan/zoom sync
    _syncing = [False]
    def _on_xlim_changed(ax_changed):
        if _syncing[0]: return
        _syncing[0] = True
        try:
            xl = ax_changed.get_xlim()
            for ax in axes.values():
                if ax is not ax_changed:
                    ax.set_xlim(xl)
            fig.canvas.draw_idle()
        finally:
            _syncing[0] = False
    for _ax in axes.values():
        _ax.callbacks.connect("xlim_changed", _on_xlim_changed)

    # Sun events
    sun_evts = None
    if _HAS_ASTRO and (twilight or night_xlim_flag):
        _jd0 = next(
            (jd for ch_bufs in bufs.values()
             for dq in ch_bufs.values()
             for jd in dq if np.isfinite(jd)), None)
        _date = obs_date or (date_from_jd(_jd0, utc_offset) if _jd0 else "")
        if not _date:
            from datetime import date as _d
            _date = _d.today().isoformat()
        sun_evts = compute_sun_events(_date, obs_lon, obs_lat, obs_alt, utc_offset)
        if night_xlim_flag and sun_evts is not None:
            xlo, xhi = night_xlim(sun_evts)
            if xlo is not None:
                for _ax in axes.values():
                    _ax.set_xlim(xlo, xhi)
                print(f"[sat] Night x-lim: {xlo:.3f} – {xhi:.3f} h")

    _redraw()

    try:
        while True:
            try:
                parts = sock.recv_multipart()
                msg   = json.loads(parts[1].decode())
            except zmq.Again:
                if new_data and time.time() - last_draw >= interval:
                    _redraw(); last_draw = time.time(); new_data = False
                time.sleep(0.05); fig.canvas.flush_events(); continue

            ch = msg.get("channel")
            if ch not in channels: continue
            jd      = msg.get("jd")
            sources = msg.get("sources", [])
            if jd is None or not sources: continue

            # Update this channel's own selection from its message
            msg_target = msg.get("target_id")
            msg_comps  = msg.get("comp_ids") or []
            if msg_target is not None:
                if (per_ch_target[ch] != msg_target
                        or per_ch_comps[ch] != list(msg_comps)):
                    per_ch_target[ch] = msg_target
                    per_ch_comps[ch]  = list(msg_comps)
                    print(f"[sat] ch{ch}: "
                          f"target={msg_target}  comps={msg_comps}")

            tid  = per_ch_target[ch]
            cids = per_ch_comps[ch]

            # Store peak per source per beam
            for s in sources:
                rid  = s.get("ref_id")
                pk   = s.get("peak")
                beam = s.get("beam")
                if rid is None or pk is None: continue
                b = beam if beam is not None else 0
                _ensure(ch, rid, beam=b)
                bufs[ch][(rid, b)].append(float(jd))
                peak_bufs[ch][(rid, b)].append(float(pk))
            new_data = True

            if tid is not None:
                def _last_peak(rid):
                    for b in (0, 1):
                        pk_dq = peak_bufs[ch].get((rid, b))
                        if pk_dq: return float(pk_dq[-1])
                    return 0.0
                track = [tid] + [r for r in cids if r != tid]
                pstr  = "  ".join(
                    [f"T={_last_peak(track[0]):.0f}"] +
                    [f"C{i}={_last_peak(r):.0f}"
                     for i, r in enumerate(track[1:], 1)])
                warn = "  ⚠SAT" if any(
                    _last_peak(r) >= sat_level for r in track) else ""
                print(f"  ch{ch} ({CH_BAND[ch]})  {pstr}{warn}")
            else:
                print(f"  ch{ch} ({CH_BAND[ch]})  jd={jd:.6f}  "
                      f"{len(sources)} sources stored (no selection yet)")

            if time.time() - last_draw >= interval:
                _redraw(); last_draw = time.time(); new_data = False
            time.sleep(0.01); fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[sat] Stopped.")
    finally:
        plt.ioff(); plt.show(block=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="SPARC4 saturation monitor — ZeroMQ subscriber",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int,   default=5556)
    p.add_argument("--channels",   default="1,2,3,4")
    p.add_argument("--db",         default="")
    p.add_argument("--sat",        type=float, default=65000.0,
                   help="Saturation level in ADU  [default: 65000]")
    p.add_argument("--history",    type=int,   default=300)
    p.add_argument("--interval",   type=float, default=2.0)
    p.add_argument("--utc-offset", type=float, default=-3.0)
    g = p.add_argument_group("Night / twilight options")
    g.add_argument("--twilight",   action="store_true")
    g.add_argument("--night-xlim", action="store_true")
    g.add_argument("--obs-lon",    type=float, default=OPD_LON)
    g.add_argument("--obs-lat",    type=float, default=OPD_LAT)
    g.add_argument("--obs-alt",    type=float, default=OPD_ALT)
    g.add_argument("--obs-date",   default="")
    p.add_argument("--save-png",   default="",
                   help="Directory to save PNG for the dashboard")
    opts     = p.parse_args()
    channels = [int(c) for c in opts.channels.split(",")]
    run(opts.host, opts.port, channels,
        opts.db or None, opts.sat, opts.history,
        opts.interval, opts.utc_offset,
        twilight=opts.twilight,
        night_xlim_flag=opts.night_xlim,
        obs_lon=opts.obs_lon, obs_lat=opts.obs_lat,
        obs_alt=opts.obs_alt, obs_date=opts.obs_date,
        save_png=opts.save_png)


if __name__ == "__main__":
    main()
