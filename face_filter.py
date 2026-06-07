#!/usr/bin/env python3
"""Filter images by face matching against a reference photo."""

import argparse
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pillow_heif
from deepface import DeepFace
from tqdm import tqdm

pillow_heif.register_heif_opener()

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

MATCH_THRESHOLD_FACTOR = 1.0
UNCERTAIN_THRESHOLD_FACTOR = 1.4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter images by matching faces against a reference photo."
    )
    parser.add_argument(
        "--ref", required=True, help="Path to reference image of the target person"
    )
    parser.add_argument(
        "--src", required=True, help="Source folder to scan recursively"
    )
    parser.add_argument(
        "--out", required=True, help="Destination folder for matched images"
    )
    parser.add_argument(
        "--uncertain",
        required=True,
        help="Destination folder for uncertain matches",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    return parser.parse_args()


def _is_no_face_error(exc):
    messages = [str(exc)]
    cause = exc.__cause__
    if cause is not None:
        messages.append(str(cause))
    combined = " ".join(messages).lower()
    return "face could not be detected" in combined


def _image_has_face(image_path, deepface_lock):
    with deepface_lock:
        DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="retinaface",
            enforce_detection=True,
        )


def validate_reference_image(ref_path, deepface_lock):
    ref_path = Path(ref_path)
    if not ref_path.is_file():
        print(f"Error: Reference image not found: {ref_path}")
        raise SystemExit(1)

    try:
        _image_has_face(ref_path, deepface_lock)
    except ValueError as exc:
        if _is_no_face_error(exc):
            print(f"Error: No face found in reference image: {ref_path}")
            raise SystemExit(1) from exc
        raise


def collect_images(src_dir):
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        print(f"Error: Source folder not found: {src_dir}")
        raise SystemExit(1)

    images = []
    for path in src_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            images.append(path)

    return sorted(images)


def move_image(image_path, src_root, dest_root):
    relative_path = image_path.relative_to(src_root)
    dest_path = dest_root / relative_path
    os.makedirs(dest_path.parent, exist_ok=True)
    stat = os.stat(image_path)
    shutil.move(str(image_path), str(dest_path))
    os.utime(dest_path, (stat.st_atime, stat.st_mtime))


def process_image(
    image_path,
    src_root,
    ref_path,
    out_dir,
    uncertain_dir,
    lock,
    deepface_lock,
    counters,
    report,
):
    rel_path = str(image_path.relative_to(src_root))

    try:
        try:
            _image_has_face(image_path, deepface_lock)
        except ValueError as exc:
            if _is_no_face_error(exc):
                with lock:
                    counters["no_face"] += 1
                    report["no_face"].append(rel_path)
                    print(f"⚠️ NO_FACE   : {rel_path}")
                return
            raise

        with deepface_lock:
            result = DeepFace.verify(
                img1_path=str(ref_path),
                img2_path=str(image_path),
                model_name="Facenet512",
                detector_backend="retinaface",
                enforce_detection=False,
                silent=True,
            )

        distance = float(result["distance"])
        threshold = float(result["threshold"])
        match_limit = threshold * MATCH_THRESHOLD_FACTOR
        uncertain_limit = threshold * UNCERTAIN_THRESHOLD_FACTOR

        if distance <= match_limit:
            move_image(image_path, src_root, out_dir)
            with lock:
                counters["matched"] += 1
                report["matched"].append((rel_path, distance))
                print(f"✅ MATCHED   : {rel_path} (distance: {distance:.2f})")
        elif distance > match_limit and distance <= uncertain_limit:
            print(
                f"DEBUG UNCERTAIN: {image_path} dist={distance:.3f} thresh={threshold:.3f}"
            )
            relative_path = image_path.relative_to(src_root)
            dest_path = uncertain_dir / relative_path
            os.makedirs(dest_path.parent, exist_ok=True)
            stat = os.stat(image_path)
            shutil.move(str(image_path), str(dest_path))
            os.utime(dest_path, (stat.st_atime, stat.st_mtime))
            with lock:
                counters["uncertain"] += 1
                report["uncertain"].append((rel_path, distance))
                print(f"❓ UNCERTAIN : {rel_path} (distance: {distance:.2f})")
        else:
            with lock:
                counters["not_matched"] += 1
                report["not_matched"].append((rel_path, distance))

    except Exception as exc:
        message = str(exc)
        with lock:
            counters["errors"] += 1
            report["errors"].append((rel_path, message))
            print(f"🔴 ERROR     : {rel_path} - {message}")


def write_report(report_path, args, counters, report):
    lines = [
        "========== FACE FILTER REPORT ==========",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Reference image: {Path(args.ref).resolve()}",
        f"Source folder: {Path(args.src).resolve()}",
        "",
        "SUMMARY:",
        f"  Total images scanned : {counters['total']}",
        f"  Matched (moved)      : {counters['matched']}",
        f"  Uncertain (moved)    : {counters['uncertain']}",
        f"  No face detected     : {counters['no_face']}",
        f"  Errors               : {counters['errors']}",
        f"  Not matched          : {counters['not_matched']}",
        "",
        "----------------------------------------",
        "✅ MATCHED FILES:",
    ]

    for rel_path, distance in report["matched"]:
        lines.append(f"  {rel_path} (distance: {distance:.2f})")
    if not report["matched"]:
        lines.append("  (none)")

    lines.extend(["", "❓ UNCERTAIN FILES (manual review needed):"])
    for rel_path, distance in report["uncertain"]:
        lines.append(f"  {rel_path} (distance: {distance:.2f})")
    if not report["uncertain"]:
        lines.append("  (none)")

    lines.extend(["", "⚠️ NO FACE DETECTED:"])
    for rel_path in report["no_face"]:
        lines.append(f"  {rel_path}")
    if not report["no_face"]:
        lines.append("  (none)")

    lines.extend(["", "🔴 ERRORS:"])
    for rel_path, message in report["errors"]:
        lines.append(f"  {rel_path} - {message}")
    if not report["errors"]:
        lines.append("  (none)")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    out_dir = Path(args.out).resolve()
    uncertain_dir = Path(args.uncertain).resolve()
    src_root = Path(args.src).resolve()
    ref_path = Path(args.ref).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    uncertain_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    deepface_lock = threading.Lock()
    validate_reference_image(ref_path, deepface_lock)
    images = collect_images(src_root)

    counters = {
        "total": len(images),
        "matched": 0,
        "uncertain": 0,
        "no_face": 0,
        "errors": 0,
        "not_matched": 0,
    }
    report = {
        "matched": [],
        "uncertain": [],
        "no_face": [],
        "errors": [],
        "not_matched": [],
    }

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(
                process_image,
                image_path,
                src_root,
                ref_path,
                out_dir,
                uncertain_dir,
                lock,
                deepface_lock,
                counters,
                report,
            )
            for image_path in images
        ]

        with tqdm(total=len(futures), desc="Scanning images") as progress:
            for future in as_completed(futures):
                future.result()
                progress.update(1)

    report_path = Path("report.txt")
    write_report(report_path, args, counters, report)
    print(f"\nReport written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
