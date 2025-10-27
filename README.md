# medieval-interlines-OCR
Python code that helps you isolate lines of any type of documents. Was first created for old handwritten mediaval books. 

This project provides a Python script that automatically detects **interlines** (the spaces between lines of text) in a scanned or photographed page. It produces:

* an **annotated image** with red polylines following each detected interline;
* a **CSV file** listing, for each interline, the estimated vertical position in several horizontal segments;
* **statistics** about line spacing (min / max / mean / median) printed in the console.

---

## Purpose

* **Layout analysis**: estimate line spacing to normalize or assess document layout.
* **OCR preprocessing**: locate interlines to segment text into individual lines.
* **Scan quality control**: identify spacing issues such as overlapping or uneven line spacing.

---

## Algorithm Overview

1. **Binarization (Otsu)**

   * Convert the image to grayscale, then apply Otsu’s automatic thresholding to produce a binary mask (dark text = 1, background = 0).

2. **Vertical Segmentation**

   * The image is divided into `segments` columns of equal width. This helps handle local curvature or distortions.

3. **Vertical Projection + Smoothing**

   * For each segment, sum the binary pixels per row to obtain a vertical profile. Apply a moving average filter of size `smooth` to reduce noise.

4. **Adaptive Threshold by Percentile**

   * Set a threshold equal to the `valley_percentile` percentile of the smoothed projection. Valleys (local minima) below this threshold are interline candidates.

5. **Local Minima Detection with Spacing Constraint**

   * `find_local_minima` selects minima at least `min_gap` pixels apart and below the threshold.

6. **Tracking Minima Across Segments**

   * `track_minima_across_segments` associates minima between adjacent segments, allowing a vertical deviation of up to `max_jump` pixels. This creates continuous tracks.

7. **Track Interpolation**

   * `interpolate_track` fills missing values and yields a complete y-position for each track across all segments.

8. **Adaptive Band Grouping (De-duplication)**

   * `keep_first_in_each_band` computes the median interline spacing and keeps only the **first line** in each group separated by at least `band_factor × median_spacing`. This merges duplicate detections and keeps only real interlines.

9. **Visualization and Export**

   * Red polylines are drawn through the estimated interline positions. The annotated image and CSV are saved to disk.

---

## Code Structure

* `otsu_threshold`, `to_binary`: robust binarization (Otsu), text = 1.
* `smooth_1d`: moving average filter (odd window).
* `find_local_minima`: finds local minima below a threshold and spaced by `min_dist`.
* `track_minima_across_segments`: connects minima left→right with `max_jump` tolerance.
* `interpolate_track`: linear interpolation and edge extension.
* `keep_first_in_each_band`: filters duplicate lines using median spacing × `factor`.
* `detect_interlines(...)`: complete processing pipeline + image/CSV output + stats.
* `main()`: example entry point with sample parameters.

---

## Installation

Requires Python ≥ 3.8.

```bash
pip install pillow numpy
```

---

## Usage

1. Place your input image (e.g., `texttotest2.png`) in the project directory.
2. Run the script:

```bash
python detect_interlines.py
```

By default (as in `main()`), this generates:

* `texttotest_interlines_band.png` (annotated image)
* `texttotest_interlines_band.csv` (y-positions per segment)
* line spacing statistics printed in the console.

> 💡 You can also import and call `detect_interlines(...)` directly from another script.

---

## Main Parameters

Parameters are passed to `detect_interlines(...)` (default values in parentheses):

* `segments` (12): number of vertical columns. ↑ for more curvature robustness, ↓ for faster processing.
* `smooth` (9): smoothing kernel size. Too small = noise; too large = flattened minima.
* `min_gap` (18 default, 50 in example): minimum vertical distance between two candidate interlines (pixels). Tune based on **font size** and image resolution.
* `valley_percentile` (35.0 default, 20.0 in example): adaptive threshold percentile; lower → more selective.
* `max_jump` (20): maximum allowed vertical deviation between adjacent segments.
* `band_factor` (0.6 default, 3.3 in example): fraction of median spacing used to merge nearby tracks. Larger → stronger merging (fewer detected lines).
* `csv_path` (None): if provided, the CSV file is written.

---

## Tuning Tips

* **High-resolution images**: increase `min_gap` according to DPI.
* **Tightly spaced text**: lower `min_gap` and/or raise `valley_percentile`.
* **Noisy background**: increase `smooth`; consider pre-denoising before Otsu.
* **Curved or warped pages**: increase `segments` and possibly `max_jump`.
* **Duplicate detections**: increase `band_factor` for stronger grouping.

---

## Output Details

### Annotated Image

Red polylines connecting the detected interline positions (one per segment).

### CSV File

Columns:

* `track_id`: ID of the retained interline after grouping.
* `seg_0_y` … `seg_{N-1}_y`: vertical positions (in pixels) for each segment.

### Console Statistics

* Median, mean, min, and max spacing between detected interlines (computed on average y per track).

---

## Limitations & Future Improvements

* **Tilted pages (skew)**: strong rotation may affect vertical projections — a **deskew** step (Hough transform, etc.) is recommended.
* **Lighting variations / shadows**: Otsu’s global threshold may fail — local thresholding (Sauvola/Niblack) would help.
* **Highly irregular lines**: tracking assumes smooth vertical continuity.
* **Dense handwriting**: valleys less pronounced; tune `smooth`, `valley_percentile`, `band_factor`.
* **Non-Latin scripts / calligraphy**: projection-based approach may need adaptation.

Planned enhancements:

* CLI with `argparse` + logging;
* optional local thresholding (Sauvola);
* automatic deskewing;
* outlier removal heuristics;
* GeoJSON/JSON export for line geometry;
* unit tests with synthetic line patterns.

---

## Minimal Example

```python
from detect_interlines import detect_interlines

tracks = detect_interlines(
    input_path="text.png",
    output_path="text_interlines.png",
    csv_path="text_interlines.csv",
    segments=16,
    smooth=11,
    min_gap=40,
    valley_percentile=25.0,
    max_jump=25,
    band_factor=2.5,
)
print(f"Detected interlines: {len(tracks)}")
```

![Result](https://github.com/LioCrea/medieval-interlines-OCR/blob/main/texttotest_interlines.png)



---

## License

Feel free to use. Just DM me.
