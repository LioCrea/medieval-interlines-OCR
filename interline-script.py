#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw
import csv
from typing import List, Optional

# --------------------- existing utilities ---------------------

def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    # Between-class variance; add tiny epsilon to avoid division by zero
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.nanargmax(sigma_b2))

def to_binary(image: Image.Image) -> np.ndarray:
    g = image.convert("L")
    arr = np.asarray(g, dtype=np.uint8)
    t = otsu_threshold(arr)
    return (arr < t).astype(np.uint8)  # dark text = 1, background = 0

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    # Ensure odd window size for symmetric moving average
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp.astype(np.float64), kernel, mode="valid")

def find_local_minima(y: np.ndarray, min_dist: int, max_val: float) -> List[int]:
    """Return indices of local minima that are at least `min_dist` apart
    and whose values are <= `max_val`. A simple non-maximum suppression keeps
    the lowest valley within each neighborhood.
    """
    n = len(y)
    mins = []
    for i in range(1, n - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1] and y[i] <= max_val:
            mins.append(i)
    selected = []
    last = -10**9
    for idx in mins:
        if selected and (idx - last) < min_dist:
            if y[idx] < y[selected[-1]]:
                selected[-1] = idx
                last = idx
        else:
            selected.append(idx)
            last = idx
    return selected

def track_minima_across_segments(minima_by_seg: List[List[int]], max_jump: int) -> List[List[Optional[int]]]:
    """Greedy left→right assignment of minima across segments, with a maximum
    allowed vertical jump `max_jump`. Creates tracks, inserting None where a
    segment has no match. Tracks are sorted top-to-bottom by their mean y.
    """
    tracks: List[List[Optional[int]]] = []
    if not minima_by_seg:
        return tracks
    # Initialize tracks with minima found in the first segment
    for y in minima_by_seg[0]:
        tracks.append([y])
    # Propagate through remaining segments
    for s in range(1, len(minima_by_seg)):
        ys = minima_by_seg[s]
        # Extend all existing tracks with a placeholder for this segment
        for tr in tracks:
            tr.append(None)
        used = set()
        for y in ys:
            best_track, best_dist = None, 10**9
            for ti, tr in enumerate(tracks):
                prev_y = tr[s - 1]
                if prev_y is None:
                    continue
                d = abs(y - prev_y)
                if d < best_dist and d <= max_jump and ti not in used:
                    best_dist, best_track = d, ti
            if best_track is not None:
                tracks[best_track][s] = y
                used.add(best_track)
            else:
                # Start a new track that only appears from this segment onwards
                new_tr = [None] * s + [y]
                tracks.append(new_tr)
    # Sort tracks top-to-bottom
    def avg_y(tr):
        vals = [v for v in tr if v is not None]
        return np.mean(vals) if vals else 10**9
    tracks.sort(key=avg_y)
    return tracks

def interpolate_track(track: List[Optional[int]]) -> List[int]:
    """Fill missing y positions with linear interpolation and extend
    leading/trailing None with the first/last known value."""
    n = len(track)
    known_idx = [i for i, y in enumerate(track) if y is not None]
    if not known_idx:
        return [0] * n
    result = track[:]
    first_val = track[known_idx[0]]
    for i in range(0, known_idx[0]):
        result[i] = first_val
    last_val = track[known_idx[-1]]
    for i in range(known_idx[-1] + 1, n):
        result[i] = last_val
    for i1, i2 in zip(known_idx, known_idx[1:]):
        y1, y2 = track[i1], track[i2]
        for k in range(i1 + 1, i2):
            alpha = (k - i1) / (i2 - i1)
            result[k] = int(round((1 - alpha) * y1 + alpha * y2))
    return result

# --------------------- new: adaptive band grouping ---------------------

def keep_first_in_each_band(tracks: List[List[int]], factor: float = 0.6) -> List[List[int]]:
    """
    Group detected lines into true interlines.
    - Estimate the median spacing between tracks
    - Define tolerance = factor * median_spacing
    - Keep only the first track within each band (top-to-bottom)
    """
    means = sorted([np.mean(tr) for tr in tracks])
    spacings = [b - a for a, b in zip(means[:-1], means[1:])]
    if not spacings:
        return tracks
    median_space = np.median(spacings)
    tolerance = factor * median_space

    means_tr = [(np.mean(tr), tr) for tr in tracks]
    means_tr.sort(key=lambda x: x[0])

    filtered = []
    last_y = -1e9
    for m, tr in means_tr:
        if m - last_y > tolerance:
            filtered.append(tr)
            last_y = m
    return filtered

# --------------------- main processing pipeline ---------------------

def detect_interlines(input_path: str,
                      output_path: str,
                      segments: int = 12,
                      smooth: int = 9,
                      min_gap: int = 18,
                      valley_percentile: float = 35.0,
                      max_jump: int = 20,
                      csv_path: Optional[str] = None,
                      band_factor: float = 0.6):
    # Load and binarize
    im = Image.open(input_path).convert("RGB")
    bin_img = to_binary(im)
    h, w = bin_img.shape

    # Split width into `segments` columns (last one absorbs remainder)
    widths = [w // segments] * segments
    widths[-1] += w - sum(widths)
    x_starts = [0]
    for i in range(segments - 1):
        x_starts.append(x_starts[-1] + widths[i])

    # Per-segment vertical projection and minima detection
    minima_by_seg = []
    for s in range(segments):
        x0, x1 = x_starts[s], x_starts[s] + widths[s]
        strip = bin_img[:, x0:x1]
        proj = strip.sum(axis=1).astype(np.float64)
        proj_s = smooth_1d(proj, smooth)
        thr = np.percentile(proj_s, valley_percentile)
        mins = find_local_minima(proj_s, min_dist=min_gap, max_val=thr)
        minima_by_seg.append(mins)

    # Track and interpolate across segments
    tracks = track_minima_across_segments(minima_by_seg, max_jump=max_jump)
    interp_tracks = [interpolate_track(tr) for tr in tracks]

    # De-duplicate via adaptive band grouping
    filtered_tracks = keep_first_in_each_band(interp_tracks, factor=band_factor)

    # Visualization
    vis = im.copy()
    draw = ImageDraw.Draw(vis)
    x_centers = [xs + widths[i] // 2 for i, xs in enumerate(x_starts)]
    for tr in filtered_tracks:
        pts = [(x_centers[s], int(y)) for s, y in enumerate(tr)]
        draw.line(pts, fill=(255, 0, 0), width=2)
    vis.save(output_path)

    # Optional CSV export
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["track_id"] + [f"seg_{i}_y" for i in range(segments)]
            writer.writerow(header)
            for tid, tr in enumerate(filtered_tracks):
                writer.writerow([tid] + [int(y) for y in tr])

    # ---- spacing statistics ----
    means = [np.mean(tr) for tr in filtered_tracks]
    spacings = [b - a for a, b in zip(means[:-1], means[1:])]
    if spacings:
        print(f"Median spacing : {np.median(spacings):.1f} px")
        print(f"Mean spacing   : {np.mean(spacings):.1f} px")
        print(f"Min spacing    : {np.min(spacings):.1f} px")
        print(f"Max spacing    : {np.max(spacings):.1f} px")

    print(f"Annotated image saved to : {output_path}")
    print(f"# of detected interlines : {len(filtered_tracks)}")

    return filtered_tracks

# --------------------- entry point ---------------------

def main():
    input_path = "texttotest2.png"
    output_path = "texttotest_interlines_band.png"
    csv_path = "texttotest_interlines_band.csv"

    params = {
        "segments": 12,
        "smooth": 9,
        "min_gap": 50,
        "valley_percentile": 20.0,
        "max_jump": 20,
        "band_factor": 3.3,  # fraction of median spacing used as band tolerance
    }

    detect_interlines(input_path, output_path, csv_path=csv_path, **params)

if __name__ == "__main__":
    main()
