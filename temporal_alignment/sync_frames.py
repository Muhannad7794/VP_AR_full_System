#!/usr/bin/env python3
"""
sync_frames.py
==============
Temporal alignment, step 1: maps every Sony frame to the ZED frame captured at
the same moment.

Method "time" (default)
-----------------------
Both recordings carry their own time axis:
  * ZED: the camera timestamp of every SVO frame (zed_frames.csv, written by
    extract_frames.py). Frames dropped during recording appear as gaps in the
    timestamps and are therefore handled exactly.
  * Sony: the MP4 written by OBS has a constant frame rate, so frame j is at
    j / fps seconds from the start of the file (sony_frames.csv).
Both clocks run in real time, so the two time axes differ by an offset b:

    t_zed = t_sony + b      (seconds, each measured from its own first frame)

A difference in clock rate between the two machines would make b change
slowly over the recording. It is measured (drift check) and applied only
when it is significant.

The offset is estimated twice, independently:
  1. Cover events. Covering the lenses with an opaque card makes the image
     dark in both streams. Dark segments are detected from the luma signal,
     their falling and rising edges are timed with sub-frame interpolation and
     matched between the two streams.
  2. Motion correlation. The frame-to-frame motion signals of both streams
     are correlated over the whole overlap; the lag with the highest
     correlation is the offset. It uses thousands of frames and is the more
     precise estimate.
Both estimates must agree (--agree-tol). Otherwise the script stops; the
choice can then be forced with --offset-source or --offset.

Active window
-------------
Frames before the end of the first cover event and after the start of the
last one (walking to and from the rig, the cover itself) are excluded from
frame_mapping.json, with a guard margin. Frames inside any dark segment are
excluded too. The mapping of the whole overlap is kept in
frame_mapping_full.json; no file is deleted.

Outputs (data/json_output/<dataset>/)
-------------------------------------
    frame_mapping.json        {sony frame: zed frame} for the active window, both
                              1-based indices into the sorted sony_rgb / zed_rgb
                              files (the format read by smooth_frames_mapping.py)
    frame_mapping_full.json   the same for the whole overlap
    frame_mapping_detail.csv  one row per saved Sony frame: times, matched ZED
                              frame, residual and flags
    sync_report.json          offsets, events, drift check and decisions
Plots: data/plots/<dataset>/sync/

Method "dtw" (legacy)
---------------------
The earlier ResNet18 feature + FastDTW alignment, kept unchanged for
comparison. It needs only the extracted images.

Usage
-----
    python3 temporal_alignment/sync_frames.py --dataset dataset_05
    python3 temporal_alignment/sync_frames.py --dataset dataset_05 --window 30.0 262.0
    python3 temporal_alignment/sync_frames.py --dataset dataset_05 --method dtw
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sony frames whose motion to the previous frame is below this value (mean
# absolute grey-level difference, 0-255) are treated as repeated frames.
DUPLICATE_MOTION = 0.01
# A ZED frame interval longer than this many nominal periods is a gap.
GAP_FACTOR = 1.5
# Grid step of the coarse correlation search, seconds.
COARSE_STEP_S = 0.01
# Step of the fine correlation search, seconds.
FINE_STEP_S = 0.0005
# Windows of the drift check with a correlation peak below this are ignored.
DRIFT_MIN_R = 0.3


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _float(value):
    return float(value) if value not in ("", None) else np.nan


def load_zed(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[ERROR] {path} is empty")
    ts = np.array([int(r["camera_timestamp_ns"]) for r in rows], dtype=np.int64)
    t = (ts - ts[0]) / 1e9
    period = float(np.median(np.diff(t)))
    return {
        "name": "ZED",
        "t": t,
        "ts0_ns": int(ts[0]),
        "saved": np.array([int(r["saved_index"]) for r in rows]),
        "luma": np.array([_float(r["luma"]) for r in rows]),
        "motion": np.array([_float(r["motion"]) for r in rows]),
        "period": period,
    }


def load_sony(path, fps):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[ERROR] {path} is empty")
    idx = np.array([int(r["frame_index"]) for r in rows])
    container_ms = np.array([_float(r["container_ms"]) for r in rows])
    if not fps or fps <= 0:
        fps = 1000.0 / float(np.nanmedian(np.diff(container_ms)))
    return {
        "name": "Sony",
        "t": idx / fps,
        "fps": float(fps),
        "container_ms": container_ms,
        "saved": np.array([int(r["saved_index"]) for r in rows]),
        "luma": np.array([_float(r["luma"]) for r in rows]),
        "motion": np.array([_float(r["motion"]) for r in rows]),
        "period": 1.0 / fps,
    }


# ---------------------------------------------------------------------------
# Motion signal and correlation
# ---------------------------------------------------------------------------


def motion_samples(stream, invalid=None):
    """Motion samples at the midpoint between consecutive frames, log-scaled and standardised.

    A sample spans the interval between two frames; it is discarded when that
    interval is a gap (dropped frames) or when the frame is flagged invalid.
    """
    t, m = stream["t"], stream["motion"]
    mid = 0.5 * (t[1:] + t[:-1])
    val = m[1:]
    ok = np.isfinite(val) & (np.diff(t) <= GAP_FACTOR * stream["period"])
    if invalid is not None:
        ok &= ~invalid[1:]
    x = np.log1p(np.maximum(val[ok], 0.0))
    x = (x - x.mean()) / (x.std() + 1e-12)
    return mid[ok], x


def resample(t_samples, x_samples, grid_t, max_gap):
    """Linear interpolation onto a grid; grid points inside gaps or outside the data are masked."""
    values = np.interp(grid_t, t_samples, x_samples)
    idx = np.searchsorted(t_samples, grid_t)
    inside = (idx > 0) & (idx < len(t_samples))
    idx_c = np.clip(idx, 1, len(t_samples) - 1)
    small_gap = (t_samples[idx_c] - t_samples[idx_c - 1]) <= max_gap
    mask = (inside & small_gap).astype(np.float64)
    return values * mask, mask


def masked_pearson(y, my, x, mx, min_n):
    """Pearson correlation of y and x for every integer lag, using only unmasked samples.

    lag k pairs y[n] with x[n - k]. Returns (lags, r, n).
    """
    from scipy.signal import correlate, correlation_lags

    def c(a, b):
        return correlate(a, b, mode="full", method="fft")

    n = c(my, mx)
    sy, sx = c(y * my, mx), c(my, x * mx)
    syy, sxx = c(y * y * my, mx), c(my, x * x * mx)
    sxy = c(y * my, x * mx)
    lags = correlation_lags(len(y), len(x), mode="full")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        var = (sxx - sx * sx / n) * (syy - sy * sy / n)
        r = cov / np.sqrt(var)
    r[(n < min_n) | ~np.isfinite(r)] = np.nan
    return lags, r, n


class ZedMotion:
    """ZED motion samples with gap-aware interpolation at arbitrary times."""

    def __init__(self, t, x, max_gap):
        self.t, self.x, self.max_gap = t, x, max_gap

    def at(self, times):
        idx = np.searchsorted(self.t, times)
        ok = (idx > 0) & (idx < len(self.t))
        idx_c = np.clip(idx, 1, len(self.t) - 1)
        ok &= (self.t[idx_c] - self.t[idx_c - 1]) <= self.max_gap
        return np.interp(times, self.t, self.x), ok


def pearson_at(offsets, s_t, s_x, zed):
    """Exact Pearson correlation between Sony samples and ZED motion at s_t + offset."""
    r = np.full(len(offsets), np.nan)
    n = np.zeros(len(offsets), dtype=int)
    for i, b in enumerate(offsets):
        y, ok = zed.at(s_t + b)
        if ok.sum() < 50:
            continue
        r[i] = np.corrcoef(s_x[ok], y[ok])[0, 1]
        n[i] = int(ok.sum())
    return r, n


def refine_peak(offsets, r):
    """Offset of the correlation maximum with parabolic sub-step interpolation."""
    if np.all(~np.isfinite(r)):
        return np.nan, np.nan
    i = int(np.nanargmax(r))
    b, peak = offsets[i], r[i]
    if 0 < i < len(r) - 1 and np.isfinite(r[i - 1]) and np.isfinite(r[i + 1]):
        denom = r[i - 1] - 2 * r[i] + r[i + 1]
        if denom < 0:
            b += 0.5 * (r[i - 1] - r[i + 1]) / denom * (offsets[1] - offsets[0])
    return float(b), float(peak)


def motion_offset(sony, zed_stream, max_offset, sony_invalid, zed_invalid):
    """Coarse (grid, FFT) and fine (exact) search of the offset from motion correlation.

    Frames around cover events are excluded by the caller, so this estimate is
    independent of the event-based one: the few large brightness steps of a
    cover would otherwise dominate the correlation and carry their own
    frame-quantisation error into it.
    """
    s_t, s_x = motion_samples(sony, invalid=sony_invalid)
    z_t, z_x = motion_samples(zed_stream, invalid=zed_invalid)
    max_gap = GAP_FACTOR * zed_stream["period"]
    h = COARSE_STEP_S
    grid_s = np.arange(0.0, sony["t"][-1] + h, h)
    grid_z = np.arange(0.0, zed_stream["t"][-1] + h, h)
    x, mx = resample(s_t, s_x, grid_s, GAP_FACTOR * sony["period"] * 2)
    y, my = resample(z_t, z_x, grid_z, max_gap)
    min_n = int(20.0 / h)  # at least 20 s of overlap
    lags, r, n = masked_pearson(y, my, x, mx, min_n)
    lag_s = lags * h
    keep = np.abs(lag_s) <= max_offset
    lag_s, r, n = lag_s[keep], r[keep], n[keep]
    if np.all(~np.isfinite(r)):
        return None
    i = int(np.nanargmax(r))
    b_coarse, r_coarse = float(lag_s[i]), float(r[i])
    # Second-best peak at least 0.5 s away, as a measure of ambiguity.
    far = np.abs(lag_s - b_coarse) > 0.5
    r_second = float(np.nanmax(r[far])) if np.any(far & np.isfinite(r)) else np.nan

    zed = ZedMotion(z_t, z_x, max_gap)
    fine_offsets = np.arange(
        b_coarse - 0.03, b_coarse + 0.03 + FINE_STEP_S / 2, FINE_STEP_S
    )
    r_fine, n_fine = pearson_at(fine_offsets, s_t, s_x, zed)
    b_fine, r_peak = refine_peak(fine_offsets, r_fine)
    return {
        "offset_s": b_fine,
        "peak_r": r_peak,
        "coarse_offset_s": b_coarse,
        "coarse_peak_r": r_coarse,
        "second_peak_r": r_second,
        "samples": (
            int(n_fine[np.nanargmax(r_fine)]) if np.any(np.isfinite(r_fine)) else 0
        ),
        "_curves": {
            "coarse_lag": lag_s,
            "coarse_r": r,
            "fine_lag": fine_offsets,
            "fine_r": r_fine,
        },
        "_samples": (s_t, s_x, zed),
    }


def drift_check(s_t, s_x, zed, b_global, window_s, t_range):
    """Offset in sliding windows; a linear fit measures a clock-rate difference."""
    step = window_s / 2.0
    offsets = np.arange(b_global - 0.15, b_global + 0.15 + 0.0005, 0.001)
    windows = []
    start = t_range[0]
    while start + window_s <= t_range[1] + 1e-9:
        sel = (s_t >= start) & (s_t < start + window_s)
        if sel.sum() >= 200:
            r, n = pearson_at(offsets, s_t[sel], s_x[sel], zed)
            b, peak = refine_peak(offsets, r)
            windows.append(
                {
                    "t_center_s": start + window_s / 2,
                    "offset_s": b,
                    "peak_r": peak,
                    "samples": int(sel.sum()),
                }
            )
        start += step
    good = [
        w for w in windows if np.isfinite(w["peak_r"]) and w["peak_r"] >= DRIFT_MIN_R
    ]
    fit = {"windows_used": len(good), "apply": False}
    if len(good) >= 4:
        tc = np.array([w["t_center_s"] for w in good])
        bo = np.array([w["offset_s"] for w in good])
        coef, cov = np.polyfit(tc, bo, 1, cov=True)
        slope, slope_se = float(coef[0]), float(np.sqrt(cov[0, 0]))
        resid = bo - np.polyval(coef, tc)
        span = float(tc.max() - tc.min())
        fit.update(
            {
                "slope_s_per_s": slope,
                "slope_ppm": slope * 1e6,
                "slope_se_ppm": slope_se * 1e6,
                "change_over_span_ms": slope * span * 1000.0,
                "intercept_s": float(coef[1]),
                "residual_max_ms": float(np.max(np.abs(resid)) * 1000.0),
                "residual_std_ms": float(np.std(resid) * 1000.0),
                "offset_spread_ms": float((bo.max() - bo.min()) * 1000.0),
            }
        )
        # Applied only when clearly resolved and larger than a few milliseconds.
        fit["apply"] = bool(abs(slope) > 3 * slope_se and abs(slope * span) > 0.004)
    return windows, fit


# ---------------------------------------------------------------------------
# Cover events
# ---------------------------------------------------------------------------


def dark_segments(stream, threshold, min_dur, merge_gap):
    """Dark segments of a stream with sub-frame timed falling and rising edges."""
    t, luma, period = stream["t"], stream["luma"], stream["period"]
    n = len(luma)
    dark = luma < threshold
    runs, i = [], 0
    while i < n:
        if dark[i]:
            j = i
            while j + 1 < n and dark[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and t[a] - t[merged[-1][1]] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    segments = []
    for a, b in merged:
        if t[b] - t[a] + period < min_dur:
            continue
        level_dark = float(np.median(luma[a : b + 1]))
        seg = {
            "i0": int(a),
            "i1": int(b),
            "t0": float(t[a]),
            "t1": float(t[b]),
            "dark_luma": level_dark,
            "fall": None,
            "rise": None,
            "fall_reliable": False,
            "rise_reliable": False,
        }
        if a > 0:
            pre = (t >= t[a] - 0.5) & (t < t[a]) & ~dark
            bright = float(np.median(luma[pre])) if pre.any() else float(luma[a - 1])
            mid = 0.5 * (bright + level_dark)
            k = a - 1
            while k > 0 and luma[k] < mid:
                k -= 1
            l0, l1 = luma[k], luma[k + 1]
            frac = (l0 - mid) / (l0 - l1) if l0 != l1 else 0.5
            seg["fall"] = float(t[k] + np.clip(frac, 0, 1) * (t[k + 1] - t[k]))
            seg["fall_reliable"] = bool(t[k + 1] - t[k] <= GAP_FACTOR * period)
        if b < n - 1:
            post = (t > t[b]) & (t <= t[b] + 0.5) & ~dark
            bright = float(np.median(luma[post])) if post.any() else float(luma[b + 1])
            mid = 0.5 * (bright + level_dark)
            k = b + 1
            while k < n - 1 and luma[k] < mid:
                k += 1
            l0, l1 = luma[k - 1], luma[k]
            frac = (mid - l0) / (l1 - l0) if l1 != l0 else 0.5
            seg["rise"] = float(t[k - 1] + np.clip(frac, 0, 1) * (t[k] - t[k - 1]))
            seg["rise_reliable"] = bool(t[k] - t[k - 1] <= GAP_FACTOR * period)
        segments.append(seg)
    return segments


def _edges(seg):
    out = []
    for kind in ("fall", "rise"):
        if seg[kind] is not None and seg[f"{kind}_reliable"]:
            out.append((kind, seg[kind]))
    return out


def match_events(seg_s, seg_z, tol, max_offset):
    """Offset that pairs the most Sony and ZED dark-segment edges."""
    candidates = []
    for ss in seg_s:
        for kind, ts in _edges(ss):
            for sz in seg_z:
                for kind_z, tz in _edges(sz):
                    if kind == kind_z and abs(tz - ts) <= max_offset:
                        candidates.append(tz - ts)
    best = None
    for b in candidates:
        pairs, used = [], set()
        for i, ss in enumerate(seg_s):
            for j, sz in enumerate(seg_z):
                if j in used:
                    continue
                common = [
                    (k, ts, dict(_edges(sz))[k])
                    for k, ts in _edges(ss)
                    if k in dict(_edges(sz))
                ]
                if common and all(abs(tz - ts - b) <= tol for _, ts, tz in common):
                    pairs.append((i, j, common))
                    used.add(j)
                    break
        n_edges = sum(len(c) for _, _, c in pairs)
        resid = (
            np.mean([abs(tz - ts - b) for _, _, c in pairs for _, ts, tz in c])
            if pairs
            else np.inf
        )
        key = (n_edges, -resid)
        if best is None or key > best[0]:
            best = (key, b, pairs)
    if best is None or not best[2]:
        return None
    _, _, pairs = best
    edge_offsets = [tz - ts for _, _, c in pairs for _, ts, tz in c]
    return {
        "offset_s": float(np.median(edge_offsets)),
        "edges_used": len(edge_offsets),
        "edge_offsets_s": [float(v) for v in edge_offsets],
        "edge_spread_ms": float((max(edge_offsets) - min(edge_offsets)) * 1000.0),
        "pairs": [(int(i), int(j)) for i, j, _ in pairs],
    }


# ---------------------------------------------------------------------------
# Main time-based method
# ---------------------------------------------------------------------------


def run_time_method(args):
    ds = args.dataset
    ext_dir = os.path.join(BASE_DIR, "data", "extracted", ds)
    out_dir = os.path.join(BASE_DIR, "data", "json_output", ds)
    plot_dir = os.path.join(BASE_DIR, "data", "plots", ds, "sync")
    os.makedirs(out_dir, exist_ok=True)
    zed_csv, sony_csv = os.path.join(ext_dir, "zed_frames.csv"), os.path.join(
        ext_dir, "sony_frames.csv"
    )
    if not (os.path.isfile(zed_csv) and os.path.isfile(sony_csv)):
        sys.exit(
            f"[ERROR] {zed_csv} or {sony_csv} not found. Run the current extract_frames.py "
            "first (it writes the timing files), or use --method dtw."
        )
    info_path = os.path.join(ext_dir, "extraction_info.json")
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {}
    fps_sony = info.get("sony", {}).get("fps")
    zed_skip = int(info.get("skip", 1))

    zed = load_zed(zed_csv)
    sony = load_sony(sony_csv, fps_sony)

    print(f"\n{'=' * 60}\n  Time-based temporal alignment - {ds}\n{'=' * 60}\n")
    z_dt = np.diff(zed["t"])
    gaps = np.flatnonzero(z_dt > GAP_FACTOR * zed["period"])
    lost = int(np.sum(np.round(z_dt[gaps] / zed["period"]) - 1)) if len(gaps) else 0
    print(
        f"  ZED : {len(zed['t'])} frames, {zed['t'][-1]:.2f} s, {1.0 / zed['period']:.3f} fps (median), "
        f"{len(gaps)} gaps, about {lost} frames missing"
    )
    duplicates = np.isfinite(sony["motion"]) & (sony["motion"] < DUPLICATE_MOTION)
    dup_idx = np.flatnonzero(duplicates)
    print(
        f"  Sony: {len(sony['t'])} frames, {sony['t'][-1]:.2f} s, {sony['fps']:.3f} fps (container), "
        f"{len(dup_idx)} repeated frames"
    )
    cms = sony["container_ms"]
    if np.isfinite(cms).sum() > 10:
        dev = cms / 1000.0 - sony["t"]
        dev = dev[np.isfinite(dev)]
        spread = float((np.percentile(dev, 99) - np.percentile(dev, 1)) * 1000)
        if spread > 2.0:
            print(
                f"  [WARN] Sony container timestamps deviate from a constant frame rate "
                f"(1-99 % spread {spread:.1f} ms). The file may have a variable frame rate."
            )

    report = {
        "dataset": ds,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "time",
        "zed": {
            "frames": len(zed["t"]),
            "duration_s": float(zed["t"][-1]),
            "fps_median": 1.0 / zed["period"],
            "gaps": int(len(gaps)),
            "frames_missing": lost,
            "gap_list": [
                {
                    "after_frame": int(g),
                    "t_s": float(zed["t"][g]),
                    "length_ms": float(z_dt[g] * 1000),
                }
                for g in gaps
            ],
        },
        "sony": {
            "frames": len(sony["t"]),
            "duration_s": float(sony["t"][-1]),
            "fps": sony["fps"],
            "repeated_frames": int(len(dup_idx)),
            "repeated_frame_indices": [int(i) for i in dup_idx[:200]],
        },
    }

    # --- cover events -------------------------------------------------------
    thr_s = args.dark_threshold_sony or min(
        40.0, 0.35 * float(np.nanmedian(sony["luma"]))
    )
    thr_z = args.dark_threshold_zed or min(
        40.0, 0.35 * float(np.nanmedian(zed["luma"]))
    )
    seg_s = dark_segments(sony, thr_s, args.min_dark, 0.2)
    seg_z = dark_segments(zed, thr_z, args.min_dark, 0.2)
    print(
        f"\n  Dark segments (cover events): Sony {len(seg_s)} (threshold {thr_s:.1f}), "
        f"ZED {len(seg_z)} (threshold {thr_z:.1f})"
    )
    for name, segs in (("Sony", seg_s), ("ZED", seg_z)):
        for s in segs:
            fall = f"{s['fall']:.3f}" if s["fall"] is not None else "start"
            rise = f"{s['rise']:.3f}" if s["rise"] is not None else "end"
            print(
                f"    {name:4s} {fall:>9s} -> {rise:>9s} s   dark luma {s['dark_luma']:.1f}"
            )
    ev = match_events(seg_s, seg_z, args.event_tol, args.max_offset)
    if ev:
        print(
            f"  Event offset : {ev['offset_s'] * 1000:+.1f} ms from {ev['edges_used']} edges "
            f"(spread {ev['edge_spread_ms']:.1f} ms)"
        )
    else:
        print("  Event offset : no cover events matched between the two streams")

    # --- motion correlation ---------------------------------------------------
    def near_dark(stream, segments, pad):
        t = stream["t"]
        out = np.zeros(len(t), dtype=bool)
        for seg in segments:
            lo = (seg["fall"] if seg["fall"] is not None else seg["t0"]) - pad
            hi = (seg["rise"] if seg["rise"] is not None else seg["t1"]) + pad
            out |= (t >= lo) & (t <= hi)
        return out

    mo = motion_offset(
        sony,
        zed,
        args.max_offset,
        duplicates | near_dark(sony, seg_s, 0.5),
        near_dark(zed, seg_z, 0.5),
    )
    if mo:
        ambiguity = mo["second_peak_r"] / mo["peak_r"] if mo["peak_r"] > 0 else np.nan
        print(
            f"  Motion offset: {mo['offset_s'] * 1000:+.1f} ms, peak r {mo['peak_r']:.3f}, "
            f"next peak (>0.5 s away) r {mo['second_peak_r']:.3f}, {mo['samples']} samples"
        )
    else:
        ambiguity = np.nan
        print("  Motion offset: correlation could not be computed")

    report["events"] = {
        "threshold_sony": thr_s,
        "threshold_zed": thr_z,
        "sony_segments": seg_s,
        "zed_segments": seg_z,
        "match": {k: v for k, v in (ev or {}).items()},
    }
    report["motion"] = {k: v for k, v in (mo or {}).items() if not k.startswith("_")}

    # --- decision ---------------------------------------------------------------
    # The motion correlation can have several similar peaks when the board was
    # moved in a regular rhythm. Such a result is "ambiguous" on its own; a
    # cover event then decides which peak is the right one.
    motion_strong = bool(mo and np.isfinite(mo["peak_r"]) and mo["peak_r"] >= 0.3)
    motion_unambiguous = motion_strong and (
        not np.isfinite(ambiguity) or ambiguity < 0.8
    )
    decision, b0 = None, None
    if args.offset is not None:
        decision, b0 = "manual (--offset)", args.offset
    elif args.offset_source == "events":
        if not ev:
            sys.exit(
                "[ERROR] --offset-source events, but no cover events were matched."
            )
        decision, b0 = "events (forced)", ev["offset_s"]
    elif args.offset_source == "motion":
        if not mo:
            sys.exit("[ERROR] --offset-source motion, but the correlation failed.")
        decision, b0 = "motion (forced)", mo["offset_s"]
    else:
        if ev and motion_strong:
            diff = abs(ev["offset_s"] - mo["offset_s"])
            if diff > args.agree_tol:
                # Highest motion peak near the event offset (another periodic peak may be the global one).
                s_t, s_x, zm = mo["_samples"]
                offs = np.arange(
                    ev["offset_s"] - args.agree_tol,
                    ev["offset_s"] + args.agree_tol + 1e-9,
                    FINE_STEP_S,
                )
                r_loc, _ = pearson_at(offs, s_t, s_x, zm)
                b_loc, r_peak_loc = refine_peak(offs, r_loc)
                interior = (
                    np.isfinite(b_loc)
                    and abs(b_loc - ev["offset_s"]) < 0.9 * args.agree_tol
                )
                if interior and r_peak_loc >= 0.3 and r_peak_loc >= 0.8 * mo["peak_r"]:
                    print(
                        f"  Motion peak near the event offset: {b_loc * 1000:+.1f} ms, r {r_peak_loc:.3f} "
                        "(the global motion peak was a repetition of the movement rhythm)"
                    )
                    mo["offset_s"], mo["peak_r"] = b_loc, r_peak_loc
                    diff = abs(ev["offset_s"] - b_loc)
            report["agreement_ms"] = diff * 1000
            if diff > args.agree_tol:
                print(
                    f"\n[ERROR] Event and motion offsets disagree by {diff * 1000:.1f} ms "
                    f"(tolerance {args.agree_tol * 1000:.0f} ms). Check the plots in {plot_dir} and "
                    "re-run with --offset-source events|motion or --offset <seconds>."
                )
                write_plots(
                    plot_dir,
                    sony,
                    zed,
                    seg_s,
                    seg_z,
                    mo,
                    None,
                    [],
                    {},
                    None,
                    ev["offset_s"],
                    args.no_plots,
                )
                save_json(os.path.join(out_dir, "sync_report.json"), report)
                sys.exit(1)
            decision, b0 = "motion (confirmed by cover events)", mo["offset_s"]
        elif motion_unambiguous:
            decision, b0 = "motion only (no matched cover events)", mo["offset_s"]
            print(
                "  [WARN] Offset from motion correlation only; no cover event confirms it."
            )
        elif ev:
            decision, b0 = "events only (motion correlation weak)", ev["offset_s"]
            print(
                "  [WARN] Offset from cover events only; the motion correlation is weak."
            )
        else:
            write_plots(
                plot_dir,
                sony,
                zed,
                seg_s,
                seg_z,
                mo,
                None,
                [],
                {},
                None,
                mo["offset_s"] if mo else 0.0,
                args.no_plots,
            )
            sys.exit(
                "[ERROR] Neither cover events nor an unambiguous motion correlation give an offset. "
                "Inspect the plots and use --offset <seconds> or --offset-source motion."
            )
    report["motion"] = {k: v for k, v in (mo or {}).items() if not k.startswith("_")}
    print(f"\n  Offset used  : {b0 * 1000:+.1f} ms  ({decision})")
    print("                 t_zed = t_sony + offset, each from its own first frame")

    # --- overlap and drift check ----------------------------------------------
    s_lo = max(0.0, -b0)
    s_hi = min(float(sony["t"][-1]), float(zed["t"][-1]) - b0)
    if s_hi - s_lo < 10.0:
        sys.exit(
            f"[ERROR] The recordings overlap for only {max(0.0, s_hi - s_lo):.1f} s at this offset."
        )
    windows, fit = [], {"apply": False}
    if mo:
        s_t, s_x, zm = mo["_samples"]
        windows, fit = drift_check(s_t, s_x, zm, b0, args.drift_window, (s_lo, s_hi))
    if fit.get("windows_used", 0) >= 4:
        print(
            f"  Drift check  : {fit['windows_used']} windows, rate {fit['slope_ppm']:+.1f} ppm "
            f"(+-{fit['slope_se_ppm']:.1f}), {fit['change_over_span_ms']:+.1f} ms over the recording, "
            f"residual max {fit['residual_max_ms']:.1f} ms -> {'applied' if fit['apply'] else 'not applied'}"
        )
        if fit["residual_max_ms"] > 0.5 * 1000.0 / sony["fps"]:
            print(
                "  [WARN] Windowed offsets deviate from a straight line by more than half a frame; "
                "inspect sync_drift.png (possible capture stall on one side)."
            )
    else:
        print("  Drift check  : too few reliable windows; constant offset used")
    s_ref = 0.5 * (s_lo + s_hi)
    if fit["apply"]:
        b_ref = float(np.polyval([fit["slope_s_per_s"], fit["intercept_s"]], s_ref))

        def offset_at(s):
            return b_ref + fit["slope_s_per_s"] * (s - s_ref)

    else:

        def offset_at(s):
            return b0 + 0.0 * s

    # --- active window --------------------------------------------------------------
    window, window_source = None, None
    guard = args.guard
    if args.window:
        window, window_source = (
            float(args.window[0]),
            float(args.window[1]),
        ), "manual (--window)"
    else:
        segs_for_window = []
        if ev:
            segs_for_window = [seg_s[i] for i, _ in ev["pairs"]]
            window_source = "matched cover events"
        elif seg_s:
            segs_for_window = seg_s
            window_source = "Sony dark segments only"
        elif seg_z:
            segs_for_window = [
                {
                    **s,
                    "t0": s["t0"] - b0,
                    "t1": s["t1"] - b0,
                    "fall": None if s["fall"] is None else s["fall"] - b0,
                    "rise": None if s["rise"] is None else s["rise"] - b0,
                }
                for s in seg_z
            ]
            window_source = "ZED dark segments only"
        segs_for_window = sorted(segs_for_window, key=lambda s: s["t0"])
        middle = 0.5 * (s_lo + s_hi)
        if len(segs_for_window) >= 2:
            first, last = segs_for_window[0], segs_for_window[-1]
            start = (
                first["rise"] if first["rise"] is not None else first["t1"]
            ) + guard
            end = (last["fall"] if last["fall"] is not None else last["t0"]) - guard
            window = (start, end)
        elif len(segs_for_window) == 1:
            seg = segs_for_window[0]
            centre = 0.5 * (seg["t0"] + seg["t1"])
            if centre < middle:
                window = (
                    (seg["rise"] if seg["rise"] is not None else seg["t1"]) + guard,
                    s_hi,
                )
            else:
                window = (
                    s_lo,
                    (seg["fall"] if seg["fall"] is not None else seg["t0"]) - guard,
                )
            print(
                "  [WARN] Only one cover event; the other end of the window is the end of the overlap."
            )
        elif args.allow_full_overlap:
            window, window_source = (s_lo, s_hi), "full overlap (--allow-full-overlap)"
        else:
            write_plots(
                plot_dir,
                sony,
                zed,
                seg_s,
                seg_z,
                mo,
                None,
                [],
                {},
                None,
                b0,
                args.no_plots,
            )
            sys.exit(
                "[ERROR] No cover events found, so the active window is unknown. Inspect "
                f"{plot_dir}/sync_luma.png and re-run with --window START END (Sony seconds) "
                "or --allow-full-overlap."
            )
    window = (max(window[0], s_lo), min(window[1], s_hi))
    if window[1] - window[0] < 5.0:
        sys.exit(f"[ERROR] Active window too short: {window[0]:.2f}-{window[1]:.2f} s.")
    print(
        f"  Active window: Sony {window[0]:.2f}-{window[1]:.2f} s "
        f"({window[1] - window[0]:.1f} s, {window_source}, guard {guard:.2f} s)"
    )

    # --- mapping ------------------------------------------------------------------
    max_dt = (
        (args.max_dt_ms / 1000.0) if args.max_dt_ms else 0.6 * zed["period"] * zed_skip
    )
    z_saved = np.flatnonzero(zed["saved"] >= 0)
    z_saved_t = zed["t"][z_saved]

    def in_dark(times, segments, pad):
        out = np.zeros(len(times), dtype=bool)
        for s in segments:
            lo = (s["fall"] if s["fall"] is not None else s["t0"]) - pad
            hi = (s["rise"] if s["rise"] is not None else s["t1"]) + pad
            out |= (times >= lo) & (times <= hi)
        return out

    s_saved = np.flatnonzero(sony["saved"] >= 0)
    s_times = sony["t"][s_saved]
    target = s_times + offset_at(s_times)
    pos = np.clip(np.searchsorted(z_saved_t, target), 1, len(z_saved_t) - 1)
    nearer_left = np.abs(z_saved_t[pos - 1] - target) <= np.abs(z_saved_t[pos] - target)
    pos = np.where(nearer_left, pos - 1, pos)
    residual = z_saved_t[pos] - target
    matched = np.abs(residual) <= max_dt
    in_window = (s_times >= window[0]) & (s_times <= window[1])
    dark = in_dark(s_times, seg_s, guard) | in_dark(z_saved_t[pos], seg_z, guard)
    dup = duplicates[s_saved]
    use = matched & in_window & ~dark

    mapping, mapping_full = {}, {}
    detail_path = os.path.join(out_dir, "frame_mapping_detail.csv")
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sony_frame",
                "sony_saved_index",
                "sony_time_s",
                "zed_target_time_s",
                "zed_saved_index",
                "zed_grab_index",
                "zed_time_s",
                "residual_ms",
                "matched",
                "in_window",
                "dark",
                "sony_repeated",
                "used",
            ]
        )
        for k in range(len(s_saved)):
            zi = int(z_saved[pos[k]])
            sony_key, zed_val = (
                int(sony["saved"][s_saved[k]]) + 1,
                int(zed["saved"][zi]) + 1,
            )
            if matched[k]:
                mapping_full[str(sony_key)] = zed_val
            if use[k]:
                mapping[str(sony_key)] = zed_val
            w.writerow(
                [
                    int(s_saved[k]),
                    sony_key - 1,
                    f"{s_times[k]:.6f}",
                    f"{target[k]:.6f}",
                    zed_val - 1,
                    zi,
                    f"{zed['t'][zi]:.6f}",
                    f"{residual[k] * 1000:.3f}",
                    int(matched[k]),
                    int(in_window[k]),
                    int(dark[k]),
                    int(dup[k]),
                    int(use[k]),
                ]
            )

    unmatched_in_window = int(np.sum(in_window & ~matched))
    res_used = residual[use] * 1000
    print(
        f"\n  Mapping      : {len(mapping)} Sony frames in the active window "
        f"({len(mapping_full)} in the whole overlap)"
    )
    if len(res_used):
        print(
            f"  Residual     : |median| {np.median(np.abs(res_used)):.1f} ms, max {np.max(np.abs(res_used)):.1f} ms "
            f"(limit {max_dt * 1000:.1f} ms)"
        )
    print(
        f"  Excluded in window: {unmatched_in_window} without a ZED frame within the limit "
        f"(dropped ZED frames), {int(np.sum(in_window & matched & dark))} in dark segments"
    )
    print(
        f"  Repeated Sony frames mapped: {int(np.sum(use & dup))} (content one frame older; kept)"
    )

    report.update(
        {
            "decision": decision,
            "offset_s": float(b0),
            "event_offset_s": ev["offset_s"] if ev else None,
            "motion_offset_s": mo["offset_s"] if mo else None,
            "drift": fit,
            "drift_windows": windows,
            "overlap_sony_s": [s_lo, s_hi],
            "active_window_sony_s": list(window),
            "active_window_source": window_source,
            "guard_s": guard,
            "max_residual_ms_allowed": max_dt * 1000,
            "mapped_frames_window": len(mapping),
            "mapped_frames_overlap": len(mapping_full),
            "unmatched_in_window": unmatched_in_window,
            "residual_ms": {
                "median_abs": (
                    float(np.median(np.abs(res_used))) if len(res_used) else None
                ),
                "max_abs": float(np.max(np.abs(res_used))) if len(res_used) else None,
            },
            "zed_t0_ns": zed["ts0_ns"],
        }
    )
    save_json(os.path.join(out_dir, "frame_mapping.json"), mapping)
    save_json(os.path.join(out_dir, "frame_mapping_full.json"), mapping_full)
    save_json(os.path.join(out_dir, "sync_report.json"), report)
    write_plots(
        plot_dir,
        sony,
        zed,
        seg_s,
        seg_z,
        mo,
        window,
        windows,
        fit,
        residual[use],
        b0,
        args.no_plots,
    )

    print(
        f"\n[SUCCESS] Frame mapping saved -> {os.path.join(out_dir, 'frame_mapping.json')}"
    )
    print(
        f"          Detail: {detail_path}\n          Report: {os.path.join(out_dir, 'sync_report.json')}"
    )
    print(f"          Plots : {plot_dir}\n")


def save_json(path, obj):
    def convert(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def write_plots(
    plot_dir, sony, zed, seg_s, seg_z, mo, window, windows, fit, residual, b0, disabled
):
    if disabled:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(sony["t"], sony["luma"], lw=0.8, label="Sony luma")
    ax.plot(
        zed["t"] - b0,
        zed["luma"],
        lw=0.8,
        alpha=0.8,
        label="ZED luma (shifted by the offset)",
    )
    for s in seg_s:
        ax.axvspan(s["t0"], s["t1"], color="k", alpha=0.15)
    if window:
        for x in window:
            ax.axvline(x, color="g", ls="--")
    ax.set_xlabel("Sony time (s)")
    ax.set_ylabel("mean grey level")
    ax.set_title(
        f"Luma and cover events (offset {b0 * 1000:+.1f} ms); green = active window"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "sync_luma.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    if mo:
        c = mo["_curves"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        axes[0].plot(c["coarse_lag"], c["coarse_r"], lw=0.8)
        axes[0].set_xlabel("offset (s)")
        axes[0].set_ylabel("correlation r")
        axes[0].set_title("Motion correlation, coarse search")
        axes[1].plot(c["fine_lag"] * 1000, c["fine_r"], ".-")
        axes[1].axvline(mo["offset_s"] * 1000, color="r")
        axes[1].set_xlabel("offset (ms)")
        axes[1].set_title(
            f"Fine search: peak {mo['offset_s'] * 1000:+.1f} ms, r {mo['peak_r']:.3f}"
        )
        for a in axes:
            a.grid(alpha=0.3)
        fig.savefig(
            os.path.join(plot_dir, "sync_correlation.png"), dpi=110, bbox_inches="tight"
        )
        plt.close(fig)

    if windows:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        tc = np.array([w["t_center_s"] for w in windows])
        bo = np.array([w["offset_s"] for w in windows])
        pr = np.array([w["peak_r"] for w in windows])
        good = np.isfinite(pr) & (pr >= DRIFT_MIN_R)
        ax.plot(
            tc[good],
            (bo[good] - b0) * 1000,
            "o",
            label=f"window offset (r >= {DRIFT_MIN_R})",
        )
        ax.plot(
            tc[~good],
            (bo[~good] - b0) * 1000,
            "x",
            color="grey",
            label="weak window (ignored)",
        )
        if fit.get("windows_used", 0) >= 4:
            xs = np.array([tc.min(), tc.max()])
            ax.plot(
                xs,
                (np.polyval([fit["slope_s_per_s"], fit["intercept_s"]], xs) - b0)
                * 1000,
                "r-",
                label=f"linear fit {fit['slope_ppm']:+.1f} ppm ({'applied' if fit['apply'] else 'not applied'})",
            )
        half = 500.0 / sony["fps"]
        ax.axhspan(-half, half, color="g", alpha=0.08, label="+- half a Sony frame")
        ax.set_xlabel("Sony time (s)")
        ax.set_ylabel("offset - global offset (ms)")
        ax.set_title("Drift check: offset measured in sliding windows")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(
            os.path.join(plot_dir, "sync_drift.png"), dpi=110, bbox_inches="tight"
        )
        plt.close(fig)

    if residual is not None and len(residual):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(residual * 1000, bins=60)
        ax.set_xlabel("ZED frame time - target time (ms)")
        ax.set_ylabel("Sony frames")
        ax.set_title("Mapping residuals in the active window")
        ax.grid(alpha=0.3)
        fig.savefig(
            os.path.join(plot_dir, "sync_residuals.png"), dpi=110, bbox_inches="tight"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy method: ResNet18 features + FastDTW (unchanged algorithm)
# ---------------------------------------------------------------------------


def run_dtw_method(args):
    import glob
    import torch
    from fastdtw import fastdtw
    from PIL import Image
    from scipy.spatial.distance import cosine
    from torchvision.models import ResNet18_Weights, resnet18

    def extract_features(img_dir, model, preprocess, device, batch_size=64):
        files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not files:
            return np.array([]), []
        features, batch_imgs = [], []
        print(
            f"  -> Extracting features for {len(files)} frames in {os.path.basename(img_dir)}..."
        )
        model.eval()
        with torch.no_grad():
            for i, f in enumerate(files):
                batch_imgs.append(preprocess(Image.open(f).convert("RGB")))
                if len(batch_imgs) == batch_size or i == len(files) - 1:
                    features.extend(
                        model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                    )
                    batch_imgs = []
                    if (i + 1) % 500 == 0 or i == len(files) - 1:
                        print(f"     Processed {i + 1} / {len(files)} frames")
        return np.array(features), files

    ext_sony = os.path.join(BASE_DIR, "data", "extracted", args.dataset, "sony_rgb")
    ext_zed = os.path.join(BASE_DIR, "data", "extracted", args.dataset, "zed_rgb")
    out_dir = os.path.join(BASE_DIR, "data", "json_output", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(ext_sony) or not os.path.isdir(ext_zed):
        sys.exit(f"[ERROR] Extracted directories not found for {args.dataset}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Legacy DTW method, compute device: {device.type.upper()}")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    sony_feats, _ = extract_features(ext_sony, model, weights.transforms(), device)
    zed_feats, _ = extract_features(ext_zed, model, weights.transforms(), device)
    if len(sony_feats) == 0 or len(zed_feats) == 0:
        sys.exit("[ERROR] No images found to process.")
    distance, path = fastdtw(sony_feats, zed_feats, dist=cosine)
    print(f"  -> DTW Alignment Complete. Total Distance Cost: {distance:.4f}")
    mapping = {}
    for sony_idx, zed_idx in path:
        mapping[str(sony_idx + 1)] = int(
            zed_idx + 1
        )  # the last ZED frame is kept per Sony frame
    save_json(os.path.join(out_dir, "frame_mapping.json"), mapping)
    print(
        f"[SUCCESS] Frame mapping saved -> {os.path.join(out_dir, 'frame_mapping.json')}"
    )


def main():
    p = argparse.ArgumentParser(description="Sony-to-ZED frame mapping.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", choices=["time", "dtw"], default="time")
    p.add_argument(
        "--max-offset",
        type=float,
        default=30.0,
        help="largest start-time difference searched, seconds (default 30)",
    )
    p.add_argument(
        "--guard",
        type=float,
        default=0.5,
        help="margin around cover events excluded from the mapping, seconds (default 0.5)",
    )
    p.add_argument(
        "--min-dark",
        type=float,
        default=0.25,
        help="shortest dark segment accepted as a cover event, seconds (default 0.25)",
    )
    p.add_argument(
        "--dark-threshold-sony",
        type=float,
        default=None,
        help="luma below which a Sony frame is dark (default: 0.35 x median luma, max 40)",
    )
    p.add_argument(
        "--dark-threshold-zed",
        type=float,
        default=None,
        help="luma below which a ZED frame is dark (default: 0.35 x median luma, max 40)",
    )
    p.add_argument(
        "--event-tol",
        type=float,
        default=0.15,
        help="tolerance when pairing cover-event edges between streams, seconds",
    )
    p.add_argument(
        "--agree-tol",
        type=float,
        default=0.1,
        help="largest accepted difference between event and motion offsets, seconds",
    )
    p.add_argument(
        "--offset-source", choices=["auto", "events", "motion"], default="auto"
    )
    p.add_argument(
        "--offset",
        type=float,
        default=None,
        help="manual offset in seconds (t_zed = t_sony + offset)",
    )
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="manual active window in Sony seconds",
    )
    p.add_argument(
        "--allow-full-overlap",
        action="store_true",
        help="use the whole overlap when no cover event is found",
    )
    p.add_argument(
        "--max-dt-ms",
        type=float,
        default=None,
        help="largest accepted distance to the nearest ZED frame (default 0.6 frame)",
    )
    p.add_argument(
        "--drift-window",
        type=float,
        default=40.0,
        help="window length of the drift check, seconds",
    )
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if args.method == "dtw":
        run_dtw_method(args)
    else:
        run_time_method(args)


if __name__ == "__main__":
    main()
