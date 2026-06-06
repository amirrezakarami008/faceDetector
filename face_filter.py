#!/usr/bin/env python3
"""Filter images by face matching against a reference photo."""

import argparse
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import face_recognition
import numpy as np
import pillow_heif
from PIL import Image
from tqdm import tqdm

pillow_heif.register_heif_opener()

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

MATCH_THRESHOLD = 0.45
UNCERTAIN_THRESHOLD = 0.60
COMPARE_TOLERANCE = 0.5


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


def load_image_array(image_path):
    with Image.open(image_path) as pil_image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)


def load_reference_encoding(ref_path):
    ref_path = Path(ref_path)
    if not ref_path.is_file():
        print(f"Error: Reference image not found: {ref_path}")
        raise SystemExit(1)

    image = load_image_array(ref_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        print(f"Error: No face found in reference image: {ref_path}")
        raise SystemExit(1)

    return encodings[0]


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
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    stat = os.stat(image_path)
    shutil.move(str(image_path), str(dest_path))
    os.utime(dest_path, (stat.st_atime, stat.st_mtime))


def process_image(image_path, src_root, ref_encoding, out_dir, uncertain_dir, lock, counters, report):
    rel_path = str(image_path.relative_to(src_root))

    try:
        image = load_image_array(image_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            with lock:
                counters["no_face"] += 1
                report["no_face"].append(rel_path)
                print(f"⚠️ NO_FACE   : {rel_path}")
            return

        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_recognition.compare_faces(
            [ref_encoding], face_encodings, tolerance=COMPARE_TOLERANCE
        )
        distances = face_recognition.face_distance(face_encodings, ref_encoding)
        min_distance = float(np.min(distances))

        if min_distance <= MATCH_THRESHOLD:
            move_image(image_path, src_root, out_dir)
            with lock:
                counters["matched"] += 1
                report["matched"].append((rel_path, min_distance))
                print(f"✅ MATCHED   : {rel_path} (distance: {min_distance:.2f})")
        elif min_distance <= UNCERTAIN_THRESHOLD:
            move_image(image_path, src_root, uncertain_dir)
            with lock:
                counters["uncertain"] += 1
                report["uncertain"].append((rel_path, min_distance))
                print(f"❓ UNCERTAIN : {rel_path} (distance: {min_distance:.2f})")
        else:
            with lock:
                counters["not_matched"] += 1
                report["not_matched"].append((rel_path, min_distance))

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

    out_dir = Path(args.out)
    uncertain_dir = Path(args.uncertain)
    src_root = Path(args.src)

    out_dir.mkdir(parents=True, exist_ok=True)
    uncertain_dir.mkdir(parents=True, exist_ok=True)

    ref_encoding = load_reference_encoding(args.ref)
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
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(
                process_image,
                image_path,
                src_root,
                ref_encoding,
                out_dir,
                uncertain_dir,
                lock,
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
