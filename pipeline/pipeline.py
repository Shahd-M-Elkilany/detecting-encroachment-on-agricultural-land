"""
Food Security ML Pipeline — End-to-End Orchestrator

Runs all 8 steps sequentially:
  01. Data Acquisition (GEE or offline)
  02. Cloud Detection (U-Net + ResNet34)
  03. Cloud Removal (OpenCV Telea)  ← SKIPPED automatically if coverage < 15%
  04. Spectral Indices (NDVI, NDBI, MNDWI) → yellow-alert signal
  05. Change Detection (ChangeFormer) → confidence map
  06. Agriculture Segmentation (SegFormer-B4)
  07. Building Detection (SAM + YOLOv8-seg)
  08. Final Output (colored map, before/after chips, geocoding,
                    interactive Folium map, area report)

Multi-temporal mode (run_temporal)
-----------------------------------
Tracks run history in outputs/temporal_state.json.

  No prior change detected   → rolling window: previous vs new
  Prior change detected      → dual comparison:
      1. baseline vs new   (total cumulative encroachment)
      2. previous vs new   (new encroachment this period only)
  Regions are tagged as "new_encroachment" or "existing_encroachment".
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from config.settings import (
    CLOUD_DETECTION_CONFIG,
    LOG_CONFIG,
    OUTPUT_DIR,
)
from src.utils.geo_utils import get_rgb_from_multiband, read_geotiff
from src.utils.logger import get_logger

logger = get_logger("pipeline", log_file=LOG_CONFIG["log_file"])

CLOUD_SKIP_THRESHOLD = CLOUD_DETECTION_CONFIG["skip_removal_below_pct"]


class FoodSecurityPipeline:
    """
    End-to-end pipeline orchestrator.

    Manages data flow between all 8 steps, handles intermediate
    result caching, and supports resuming from any step.

    Cloud threshold logic
    ---------------------
    After Step 02, if max(T1_coverage, T2_coverage) < CLOUD_SKIP_THRESHOLD (15%),
    Step 03 is bypassed: the raw images are passed directly to Step 04 as if
    they were already cloud-free. This avoids artefacts introduced by unnecessary
    inpainting on clean imagery.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}

    def _log_step_start(self, step: int, name: str) -> float:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  STEP {step:02d} — {name}")
        logger.info("=" * 70)
        return time.time()

    def _log_step_end(self, step: int, name: str, start_time: float) -> None:
        elapsed = time.time() - start_time
        self.timings[f"step_{step:02d}_{name}"] = elapsed
        logger.info(f"  Step {step:02d} completed in {elapsed:.1f}s")
        logger.info("")

    # ----------------------------------------------------------
    # Step 01 — Data Acquisition
    # ----------------------------------------------------------
    def step_01_data_acquisition(
        self,
        t1_path: Optional[str | Path] = None,
        t2_path: Optional[str | Path] = None,
        use_gee: bool = False,
    ) -> Dict[str, Path]:
        start = self._log_step_start(1, "DATA ACQUISITION")
        from src.step_01_data_acquisition.acquire import run, run_offline

        if use_gee:
            result = run()
        else:
            if t1_path is None or t2_path is None:
                raise ValueError("Provide t1_path and t2_path for offline mode, or set use_gee=True")
            result = run_offline(t1_path, t2_path)

        self.results["step_01"] = result
        self._log_step_end(1, "data_acquisition", start)
        return result

    # ----------------------------------------------------------
    # Step 02 — Cloud Detection
    # ----------------------------------------------------------
    def step_02_cloud_detection(self) -> Dict[str, Any]:
        """Detect clouds in T1 and T2. Sets skip_removal flag if coverage < threshold."""
        start = self._log_step_start(2, "CLOUD DETECTION")
        from src.step_02_cloud_detection.detect_clouds import run

        paths = self.results["step_01"]
        result = run(paths["T1"], paths["T2"])
        # result["skip_removal"] is set by detect_clouds.run()

        self.results["step_02"] = result
        self._log_step_end(2, "cloud_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 03 — Cloud Removal  (conditional)
    # ----------------------------------------------------------
    def step_03_cloud_removal(self) -> Dict[str, Dict[str, Any]]:
        """
        Remove clouds — OR — pass raw images through if coverage is low.

        When skipped, populates self.results["step_03"] with the raw T1/T2
        images so that all downstream steps continue unchanged.
        """
        clouds = self.results["step_02"]

        if clouds.get("skip_removal", False):
            # ── SKIP: load raw images and pass them straight through ──────
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"  STEP 03 — CLOUD REMOVAL  [SKIPPED — coverage below {CLOUD_SKIP_THRESHOLD}%]")
            logger.info("=" * 70)
            paths = self.results["step_01"]
            t1_data, t1_meta = read_geotiff(paths["T1"])
            t2_data, t2_meta = read_geotiff(paths["T2"])
            result = {
                "T1": {"image": t1_data, "meta": t1_meta},
                "T2": {"image": t2_data, "meta": t2_meta},
                "skipped": True,
            }
            logger.info("  Raw images passed directly to Step 04")
            logger.info("")
        else:
            # ── RUN cloud removal as normal ───────────────────────────────
            start = self._log_step_start(3, "CLOUD REMOVAL")
            from src.step_03_cloud_removal.remove_clouds import run

            paths = self.results["step_01"]
            result = run(
                paths["T1"],
                paths["T2"],
                clouds["T1"]["mask"],
                clouds["T2"]["mask"],
            )
            result["skipped"] = False
            self._log_step_end(3, "cloud_removal", start)

        # ── Align T2 to T1 shape (real tiles may differ by 1-2 px) ──────────
        t1_img = result["T1"]["image"]
        t2_img = result["T2"]["image"]
        if t2_img.shape[1:] != t1_img.shape[1:]:
            import cv2

            h, w = t1_img.shape[1], t1_img.shape[2]
            logger.warning(f"Aligning T2 {t2_img.shape[1:]} → T1 {t1_img.shape[1:]} after Step 03")
            bands = [cv2.resize(t2_img[b], (w, h), interpolation=cv2.INTER_LINEAR) for b in range(t2_img.shape[0])]
            result["T2"]["image"] = np.stack(bands, axis=0)

        self.results["step_03"] = result
        return result

    # ----------------------------------------------------------
    # Step 04 — Spectral Indices
    # ----------------------------------------------------------
    def step_04_spectral_indices(self) -> Dict[str, Any]:
        """
        Compute NDVI, NDBI, MNDWI and derive yellow-alert degradation signal.
        """
        start = self._log_step_start(4, "SPECTRAL INDICES")
        from src.step_04_spectral_indices.compute_indices import run

        clean = self.results["step_03"]
        result = run(
            clean["T1"]["image"],
            clean["T2"]["image"],
            clean["T1"]["meta"],
            clean["T2"]["meta"],
        )

        self.results["step_04"] = result
        self._log_step_end(4, "spectral_indices", start)
        return result

    # ----------------------------------------------------------
    # Step 05 — Change Detection
    # ----------------------------------------------------------
    def step_05_change_detection(
        self,
        kemet1_mode: bool = False,
        kemet1_t1_path: Optional[str | Path] = None,
        kemet1_t2_path: Optional[str | Path] = None,
        kemet1_extra_pairs: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Detect land-use changes; returns binary map AND confidence scores.

        Standard mode (kemet1_mode=False):
            Uses ChangeFormer (Siamese Transformer) on raw image arrays.

        KEMET1 mode (kemet1_mode=True):
            Uses the trained Random Forest classifier on pre-computed 6-band
            spectral index GeoTIFFs. Applies temporal consistency filter when
            extra pairs are provided. Returns a tile-level score converted to a
            full-image change_map for compatibility with downstream steps.

        Args:
            kemet1_mode:        Switch to RF classifier path.
            kemet1_t1_path:     Path to T1 spectral-index GeoTIFF (KEMET1 mode).
            kemet1_t2_path:     Path to T2 spectral-index GeoTIFF (KEMET1 mode).
            kemet1_extra_pairs: List of (t1_path, t2_path) tuples for additional
                                consecutive pairs (enables temporal consistency).
                                Example: [(T2, T3), (T3, T4)]
        """
        start = self._log_step_start(5, "CHANGE DETECTION")

        if kemet1_mode:
            result = self._step_05_kemet1(
                kemet1_t1_path,
                kemet1_t2_path,
                kemet1_extra_pairs or [],
            )
        else:
            from src.step_05_change_detection.detect_changes import run

            clean = self.results["step_03"]
            result = run(
                clean["T1"]["image"],
                clean["T2"]["image"],
                clean["T1"]["meta"],
            )

        self.results["step_05"] = result
        self._log_step_end(5, "change_detection", start)
        return result

    def _step_05_kemet1(
        self,
        t1_path: Optional[str | Path],
        t2_path: Optional[str | Path],
        extra_pairs: list,
    ) -> Dict[str, Any]:
        """
        Classifier path for Step 05, using the maintained ``encroachment`` package.

        The tile classifier gives the probability that the site was encroached.
        The change map is the set of pixels that went from vegetated to built-up,
        kept only when the classifier puts the site at Medium risk or above; the
        probability is used as the change confidence for the Step 08 red score.
        Extra consecutive pairs (T2->T3, ...) are scored too, and a change that
        recurs in a majority of them is dampened as likely seasonal.
        """
        if t1_path is None or t2_path is None:
            raise ValueError("kemet1_mode requires kemet1_t1_path and kemet1_t2_path.")

        _add_repo_root_to_path()
        from encroachment import alarm
        from encroachment.analysis import Scorer
        from encroachment.features import conversion_mask, read_pair

        scorer = Scorer()
        pairs = [(Path(t1_path), Path(t2_path))] + [(Path(a), Path(b)) for a, b in extra_pairs]
        scores = []
        for before, after in pairs:
            result = scorer.score(read_pair(before, after), before.stem)
            scores.append(result.probability)
            logger.info(f"  Pair {before.name} -> {after.name}: probability={result.probability:.4f}")

        seasonal_dampen, majority = 0.6, 2
        if len(scores) > 1 and sum(s >= alarm.MEDIUM_THRESHOLD for s in scores) >= majority:
            scores = [s * seasonal_dampen for s in scores]
            logger.info(f"  Change recurs across pairs -- dampening scores x{seasonal_dampen}")

        primary = scores[0]
        risk = alarm.risk_level(primary)
        pair = read_pair(*pairs[0])
        converted = conversion_mask(np.nan_to_num(pair.before), np.nan_to_num(pair.after))
        flagged = primary >= alarm.MEDIUM_THRESHOLD
        change_map = (converted & flagged).astype(np.uint8)

        clean = self.results.get("step_03", {})
        t1_image = clean.get("T1", {}).get("image")
        if t1_image is not None and t1_image.shape[1:] != change_map.shape:
            import cv2

            h, w = t1_image.shape[1:]
            change_map = cv2.resize(change_map, (w, h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"  Primary probability: {primary:.4f} ({risk})")

        return {
            "change_map": change_map,
            "change_confidence": np.full(change_map.shape, primary, dtype=np.float32),
            "kemet1_score": primary,
            "kemet1_all_scores": scores,
            "kemet1_decision": risk,
            "kemet1_threshold": alarm.MEDIUM_THRESHOLD,
            "kemet1_model": "logistic regression (encroachment package)",
        }

    # ----------------------------------------------------------
    # Step 06 — Agriculture Segmentation
    # ----------------------------------------------------------
    def step_06_agriculture_segmentation(self) -> Dict[str, Any]:
        """Segment agricultural land in the T1 (before) image."""
        start = self._log_step_start(6, "AGRICULTURE SEGMENTATION")
        from src.step_06_agriculture_segmentation.segment_agriculture import run

        clean = self.results["step_03"]
        result = run(clean["T1"]["image"], clean["T1"]["meta"])

        self.results["step_06"] = result
        self._log_step_end(6, "agriculture_segmentation", start)
        return result

    # ----------------------------------------------------------
    # Step 07 — Building Detection
    # ----------------------------------------------------------
    def step_07_building_detection(self) -> Dict[str, Any]:
        """Detect buildings in changed agricultural areas."""
        start = self._log_step_start(7, "BUILDING DETECTION")
        from src.step_07_building_detection.detect_buildings import run

        clean = self.results["step_03"]
        change = self.results["step_05"]
        agri = self.results["step_06"]

        result = run(
            clean["T2"]["image"],
            change["change_map"],
            agri["agri_mask"],
            clean["T2"]["meta"],
        )

        self.results["step_07"] = result
        self._log_step_end(7, "building_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 08 — Final Output
    # ----------------------------------------------------------
    def step_08_final_output(self) -> Dict[str, Any]:
        """
        Generate all outputs:
          • Colored encroachment map (PNG + GeoTIFF)
          • Per-region before/after chips with lat/lon bbox
          • Weighted red-alert score (0.65×change + 0.35×spectral)
          • Reverse geocoded location names
          • Total area lost
          • Interactive Folium HTML map
          • JSON summary report
        """
        start = self._log_step_start(8, "FINAL OUTPUT")
        from src.step_08_final_output.generate_output import run

        clean = self.results["step_03"]
        spectral = self.results["step_04"]
        change = self.results["step_05"]
        agri = self.results["step_06"]
        buildings = self.results["step_07"]

        t1_rgb = get_rgb_from_multiband(clean["T1"]["image"])
        t2_rgb = get_rgb_from_multiband(clean["T2"]["image"])

        result = run(
            t2_rgb=t2_rgb,
            change_map=change["change_map"],
            agri_mask=agri["agri_mask"],
            building_mask=buildings["building_mask"],
            polygons=buildings.get("polygons", []),
            meta=clean["T2"]["meta"],
            # New inputs for enhanced output
            change_confidence=change.get("change_confidence"),
            spectral_signal=spectral.get("spectral_signal"),
            yellow_mask=spectral.get("yellow_mask"),
            t1_rgb=t1_rgb,
        )

        self.results["step_08"] = result
        self._log_step_end(8, "final_output", start)
        return result

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------
    def run_full(
        self,
        t1_path: Optional[str | Path] = None,
        t2_path: Optional[str | Path] = None,
        use_gee: bool = False,
        start_from: int = 1,
        kemet1_mode: bool = False,
        kemet1_t1_path: Optional[str | Path] = None,
        kemet1_t2_path: Optional[str | Path] = None,
        kemet1_extra_pairs: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.

        Args:
            t1_path:            Path to T1 GeoTIFF (offline mode).
            t2_path:            Path to T2 GeoTIFF (offline mode).
            use_gee:            Download from GEE if True.
            start_from:         Resume from this step number (1-8).
            kemet1_mode:        Use KEMET1 RF classifier for Step 05 instead of
                                ChangeFormer. Requires pre-computed 6-band spectral
                                index GeoTIFFs.
            kemet1_t1_path:     T1 spectral-index tile path (KEMET1 mode).
            kemet1_t2_path:     T2 spectral-index tile path (KEMET1 mode).
            kemet1_extra_pairs: Extra consecutive pairs for temporal consistency.
                                List of (t1_path, t2_path) tuples.
        """
        total_start = time.time()

        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║   FOOD SECURITY ML PIPELINE — STARTING                          ║")
        logger.info("║   Detect Buildings on Agricultural Land                          ║")
        logger.info("╚" + "═" * 68 + "╝")

        steps = [
            (1, lambda: self.step_01_data_acquisition(t1_path, t2_path, use_gee)),
            (2, lambda: self.step_02_cloud_detection()),
            (3, lambda: self.step_03_cloud_removal()),  # auto-skips if coverage < 15%
            (4, lambda: self.step_04_spectral_indices()),
            (
                5,
                lambda: self.step_05_change_detection(
                    kemet1_mode=kemet1_mode,
                    kemet1_t1_path=kemet1_t1_path,
                    kemet1_t2_path=kemet1_t2_path,
                    kemet1_extra_pairs=kemet1_extra_pairs,
                ),
            ),
            (6, lambda: self.step_06_agriculture_segmentation()),
            (7, lambda: self.step_07_building_detection()),
            (8, lambda: self.step_08_final_output()),
        ]

        total_results: Dict[str, Any] = {}
        for step_num, step_fn in steps:
            if step_num < start_from:
                logger.info(f"  Skipping step {step_num} (start_from={start_from})")
                continue
            result = step_fn()
            total_results[f"step_{step_num:02d}"] = result

        elapsed = time.time() - total_start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        return total_results

    def run_beforeafter(
        self,
        before_path: str | Path,
        after_path: str | Path,
        site_name: str = "site",
    ) -> Dict[str, Any]:
        """Skip the eight steps and score a before/after index-tile pair with the
        tile classifier, writing its HTML site report to ``outputs/``."""
        _add_repo_root_to_path()
        from encroachment.analysis import Scorer
        from encroachment.report import write_site_report

        result = Scorer().score_files(Path(before_path), Path(after_path), site_name)
        html = write_site_report(result, Path(OUTPUT_DIR))
        logger.info(f"[BA] {site_name}: probability={result.probability:.3f} ({result.risk})")
        return {**result.summary(), "html": str(html)}


def _add_repo_root_to_path() -> None:
    """Make the top-level ``encroachment`` package importable from pipeline/."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Food Security ML Pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    # full pipeline
    p_full = sub.add_parser("full", help="Run full 8-step pipeline")
    p_full.add_argument("--t1", type=str, default=None)
    p_full.add_argument("--t2", type=str, default=None)
    p_full.add_argument("--start-from", type=int, default=1)
    p_full.add_argument("--kemet1", action="store_true")
    p_full.add_argument("--kemet1-t1", type=str, default=None)
    p_full.add_argument("--kemet1-t2", type=str, default=None)

    # beforeafter mode
    p_ba = sub.add_parser("beforeafter", help="BA inference (before/after GeoTIFF pair)")
    p_ba.add_argument("--before", type=str, required=True)
    p_ba.add_argument("--after", type=str, required=True)
    p_ba.add_argument("--site-name", type=str, default="site")

    args = parser.parse_args()
    pipeline = FoodSecurityPipeline()

    if args.mode == "full":
        pipeline.run_full(
            t1_path=args.t1,
            t2_path=args.t2,
            start_from=args.start_from,
            kemet1_mode=args.kemet1,
            kemet1_t1_path=args.kemet1_t1,
            kemet1_t2_path=args.kemet1_t2,
        )
    elif args.mode == "beforeafter":
        result = pipeline.run_beforeafter(args.before, args.after, args.site_name)
        print(f"\nResult: {result}")
        print(f"Report: {result['html']}")
