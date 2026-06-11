English | [فارسی](README.md)
# FaceOut

Recursively scan a folder of photos, compare detected faces against a reference image, and **move** matched or uncertain files into separate output folders while preserving the original subfolder structure.

## Requirements

- Python 3.8+
- System build tools for `dlib` (required by `face_recognition`)

On Ubuntu/Debian:

```bash
sudo apt install cmake build-essential
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python face_filter.py \
  --ref ./me.jpg \
  --src ./photos \
  --out ./matched \
  --uncertain ./check_these \
  --threads 4
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ref` | Yes | — | Path to a reference photo of the target person. Must contain exactly one detectable face. |
| `--src` | Yes | — | Source folder scanned recursively for images. |
| `--out` | Yes | — | Destination folder for confident matches. Created if missing. |
| `--uncertain` | Yes | — | Destination folder for uncertain matches (manual review). Created if missing. |
| `--threads` | No | `4` | Number of worker threads. |

## Supported formats

The script scans for these extensions (case-insensitive):

`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`

HEIC support is provided via `pillow-heif`.

## How matching works

1. The reference image is loaded and a single face encoding is extracted. If no face is found, the script exits with an error.
2. Each image under `--src` is processed in parallel.
3. All faces in the image are detected.
4. For every detected face, the script computes the face distance to the reference encoding (also using `compare_faces()` with `tolerance=0.5`).
5. The **closest** face distance in the image determines the result:

| Distance | Result | File action |
|----------|--------|-------------|
| ≤ 0.45 | **MATCHED** | Moved to `--out` |
| > 0.45 and ≤ 0.60 | **UNCERTAIN** | Moved to `--uncertain` |
| > 0.60 | **Not matched** | Left in `--src` |
| No face detected | **NO_FACE** | Left in `--src` |
| Read/processing failure | **ERROR** | Left in `--src` |

Moved files keep their relative path from `--src` and their original file timestamps.

## Output

### Live console

Each file is logged as it is processed:

```
✅ MATCHED   : photos/2023/birthday.jpg (distance: 0.38)
❓ UNCERTAIN : photos/trips/beach.jpg (distance: 0.54)
⚠️ NO_FACE   : photos/landscape.jpg
🔴 ERROR     : photos/corrupt.jpg - [error message]
```

A `tqdm` progress bar shows overall scan progress.

### Report file

After processing, `report.txt` is written to the **current working directory** (not necessarily next to the script). It includes a summary and lists of matched, uncertain, no-face, and error files.

## Important notes

- **Files are moved, not copied.** Matched and uncertain images are removed from `--src`. Back up your source folder before running on original photos.
- Images with no face, no match, or errors stay in `--src` and are only logged.
- Uncertain matches should be reviewed manually in `--uncertain`.
- Only the first detected face in the reference image is used if multiple faces are present.

## Project layout

```
FaceOut/
├── face_filter.py      # Main script
├── requirements.txt    # Python dependencies
├── report.txt          # Generated after each run (gitignored)
├── photos/             # Example source folder (gitignored)
├── matched/            # Example output for matches (gitignored)
└── check_these/        # Example output for uncertain matches (gitignored)
```

## Dependencies

- `face_recognition` — face detection and encoding
- `tqdm` — progress bar
- `Pillow` — image loading
- `pillow-heif` — HEIC/HEIF support
