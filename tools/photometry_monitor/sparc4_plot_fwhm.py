"""
sparc4_plot_fwhm.py
===================
SPARC4 real-time FWHM / seeing monitor — ZeroMQ subscriber.

Plots median FWHM in arcsec vs local time for all four channels
and the number of detected sources per frame.

Usage
-----
::

    python sparc4_plot_fwhm.py --db /data/SPARC4/reduced
    python sparc4_plot_fwhm.py --host 192.168.1.10 --db /data/SPARC4/reduced

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
from typing import Dict, Deque, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

CH_COLOR = {1: "darkblue", 2: "darkgreen", 3: "darkorange", 4: "darkred"}
CH_BAND  = {1: "g",        2: "r",         3: "i",          4: "z"}


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


def run(host: str, port: int, channels: List[int],
        db_dir: Optional[str], history: int, interval: float,
        platescale: Optional[float], utc_offset: float,
        twilight: bool = False,
        night_xlim_flag: bool = False,
        obs_lon: float = OPD_LON,
        obs_lat: float = OPD_LAT,
        obs_alt: float = OPD_ALT,
        obs_date: str = "",
        save_png: str = "") -> None:

    # Per-channel circular buffers: jd and fwhm_arcsec
    buf_jd:   Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}
    buf_fwhm: Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}
    buf_nsrc: Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}

    # Load historical data from DB
    if db_dir:
        print("[fwhm] Loading historical data...")
        for ch in channels:
            db_path = os.path.join(db_dir, f"monitor_ch{ch}.db")
            if not os.path.isfile(db_path):
                continue
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT jd, fwhm, n_sources FROM frames "
                    "WHERE bad=0 AND jd IS NOT NULL AND fwhm IS NOT NULL "
                    "ORDER BY jd"
                ).fetchall()[-history:]
                ps = platescale or 0.335
                for r in rows:
                    buf_jd[ch].append(r["jd"])
                    buf_fwhm[ch].append(r["fwhm"] * ps)
                    buf_nsrc[ch].append(r["n_sources"] or 0)
                conn.close()
                print(f"  monitor_ch{ch}.db: {len(rows)} frames")
            except Exception as exc:
                print(f"  [warn] ch{ch}: {exc}")

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    for ch in channels:
        sock.setsockopt_string(zmq.SUBSCRIBE, f"sparc4.ch{ch}")
    sock.setsockopt(zmq.RCVTIMEO, int(interval * 1000))
    print(f"[fwhm] Subscribed  tcp://{host}:{port}  ch={channels}")

    # Figure: top panel = FWHM per channel, bottom panel = N sources
    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_fwhm = fig.add_subplot(gs[0])
    ax_nsrc = fig.add_subplot(gs[1], sharex=ax_fwhm)

    fig.patch.set_facecolor("white")
    fig.suptitle("SPARC4  —  FWHM / Seeing", fontsize=14,
                 fontweight="bold", color="black", y=0.99)
    try:
        fig.canvas.manager.set_window_title("SPARC4 — FWHM")
    except Exception:
        pass

    for ax in (ax_fwhm, ax_nsrc):
        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", ls="--", lw=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors="black", labelsize=11)

    ax_fwhm.set_ylabel('FWHM (")', fontsize=12)
    ax_fwhm.set_ylim(0, 5)
    plt.setp(ax_fwhm.get_xticklabels(), visible=False)

    ax_nsrc.set_ylabel("N src", fontsize=11)
    ax_nsrc.set_xlabel(f"Local time  (UTC{utc_offset:+.0f}h)", fontsize=12)
    _format_time_axis(ax_nsrc)

    lines_fwhm: Dict[int, plt.Line2D] = {}
    lines_nsrc: Dict[int, plt.Line2D] = {}
    last_draw = 0.0
    new_data  = False

    plt.ion()
    fig.show()

    def _redraw() -> None:
        for ch in channels:
            if not buf_jd[ch]:
                continue
            jds  = np.array(buf_jd[ch])
            fwhm = np.array(buf_fwhm[ch])
            nsrc = np.array(buf_nsrc[ch])
            ht   = _jd_to_hours(jds, utc_offset)
            col  = CH_COLOR[ch]

            if ch not in lines_fwhm:
                l, = ax_fwhm.plot(ht, fwhm, "-o", color=col,
                                  ms=4, lw=1.5, label=f"ch{ch} ({CH_BAND[ch]})")
                lines_fwhm[ch] = l
            else:
                lines_fwhm[ch].set_data(ht, fwhm)

            if ch not in lines_nsrc:
                l, = ax_nsrc.plot(ht, nsrc, "-", color=col,
                                  lw=1.2, alpha=0.7)
                lines_nsrc[ch] = l
            else:
                lines_nsrc[ch].set_data(ht, nsrc)

        ax_fwhm.relim(); ax_fwhm.autoscale_view(scaley=True)
        ax_nsrc.relim(); ax_nsrc.autoscale_view(scaley=True)
        handles, lbls = ax_fwhm.get_legend_handles_labels()
        if handles:
            ax_fwhm.legend(handles, lbls, loc="upper right", fontsize=10,
                           framealpha=0.9)

        # Shared x-axis (sharex already links them; sync ensures twilight lines)
        _sync_xlim([ax_fwhm, ax_nsrc])

        if _HAS_ASTRO and twilight and sun_evts is not None:
            for _ax in [ax_fwhm, ax_nsrc]:
                for _l in list(_ax.get_lines()):
                    if getattr(_l, "_is_twilight", False):
                        _l.remove()
            add_twilight_lines([ax_fwhm, ax_nsrc], sun_evts, legend_ax_index=0)
            for _ax in [ax_fwhm, ax_nsrc]:
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
                fig.savefig(str(pathlib.Path(save_png) / "fwhm.png"),
                            dpi=110, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception:
                pass

    # Sun events
    sun_evts = None
    if _HAS_ASTRO and (twilight or night_xlim_flag):
        _jd0 = next((jd for dq in buf_jd.values() for jd in dq
                     if np.isfinite(jd)), None)
        _date = obs_date or (date_from_jd(_jd0, utc_offset) if _jd0 else "")
        if not _date:
            from datetime import date as _d
            _date = _d.today().isoformat()
        sun_evts = compute_sun_events(_date, obs_lon, obs_lat, obs_alt, utc_offset)
        if night_xlim_flag and sun_evts is not None:
            xlo, xhi = night_xlim(sun_evts)
            if xlo is not None:
                ax_fwhm.set_xlim(xlo, xhi)
                print(f"[fwhm] Night x-lim: {xlo:.3f} – {xhi:.3f} h")

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
            if ch not in channels:
                continue
            jd      = msg.get("jd")
            fwhm_as = msg.get("fwhm_arcsec")
            ps      = platescale or msg.get("platescale", 0.335)
            n_src   = msg.get("n_sources", 0)
            if jd is None or fwhm_as is None:
                continue

            buf_jd[ch].append(float(jd))
            buf_fwhm[ch].append(float(fwhm_as))
            buf_nsrc[ch].append(int(n_src))
            new_data = True
            print(f"  ch{ch} ({CH_BAND[ch]})  "
                  f"FWHM={fwhm_as:.2f}\"  n_src={n_src}")

            if time.time() - last_draw >= interval:
                _redraw(); last_draw = time.time(); new_data = False
            time.sleep(0.01); fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[fwhm] Stopped.")
    finally:
        plt.ioff(); plt.show(block=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="SPARC4 FWHM monitor — ZeroMQ subscriber",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int,   default=5556)
    p.add_argument("--channels",   default="1,2,3,4")
    p.add_argument("--db",         default="")
    p.add_argument("--history",    type=int,   default=300)
    p.add_argument("--interval",   type=float, default=2.0)
    p.add_argument("--platescale", type=float, default=None)
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
        opts.db or None, opts.history, opts.interval,
        opts.platescale, opts.utc_offset,
        twilight=opts.twilight,
        night_xlim_flag=opts.night_xlim,
        obs_lon=opts.obs_lon, obs_lat=opts.obs_lat,
        obs_alt=opts.obs_alt, obs_date=opts.obs_date,
        save_png=opts.save_png)


if __name__ == "__main__":
    main()
