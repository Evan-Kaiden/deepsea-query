import argparse
import csv
import io
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

PER_TAXON = 140          # ~50 taxa * 140 ~= 7000 images
SEARCH_LIMIT = 1000      # images to scan per taxon before capping
SEED = 0                 # dive shuffle is deterministic so reruns are stable
WORKERS = 8
OUT = Path("dataset")

DIVE_RE = re.compile(r"framegrabs/([^/]+)/images/(\d+)/")


def is_mbari(image_dto):
    email = image_dto.contributorsEmail or ""
    return email.endswith("mbari.org") or "/m3/" in image_dto.url


def dive_of(image_dto):
    """Vehicle + dive number, e.g. 'Doc Ricketts/0962'. Frames sharing this are
    seconds-to-minutes apart in the same water."""
    m = DIVE_RE.search(image_dto.url)
    return "/".join(m.groups()) if m else image_dto.url.rsplit("/", 1)[0]


def concepts_of(image_dto):
    return sorted({b.concept for b in (image_dto.boundingBoxes or [])})


def spread(seq, n):
    """Evenly-spaced subsample of a time-ordered list, keeping order."""
    if n >= len(seq):
        return list(seq)
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


def collect(concept, per_taxon=PER_TAXON, seen=None):
    """Pick up to `per_taxon` MBARI frames for a concept, spread across dives."""
    found = images.find(GeoImageConstraints(concept=concept, limit=SEARCH_LIMIT))

    by_dive = defaultdict(list)
    for img in found:
        if not is_mbari(img):
            continue
        if seen is not None and img.uuid in seen:
            continue  # already taken for another concept; don't store it twice
        by_dive[dive_of(img)].append(img)

    # Within a dive: mixed-concept frames first, each group spread over time.
    queues = []
    for frames in by_dive.values():
        frames.sort(key=lambda i: i.url)
        multi = [i for i in frames if len(concepts_of(i)) > 1]
        single = [i for i in frames if len(concepts_of(i)) == 1]
        cap = max(1, per_taxon // 4)  # no single dive supplies more than this
        queues.append(spread(multi, cap) + spread(single, cap))

    random.Random(SEED).shuffle(queues)

    # Across dives one frame from each before any dive's second.
    picked = []
    for rank in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if rank < len(q):
                picked.append(q[rank])
                if len(picked) >= per_taxon:
                    return picked
    return picked


def fetch(img_dto, concept, session):
    """Download (or reuse) one frame; returns its manifest row."""
    out_dir = OUT / concept.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{img_dto.uuid}.png"

    if path.exists():
        full = Image.open(path)
    else:
        resp = session.get(img_dto.url, timeout=30)
        resp.raise_for_status()
        full = Image.open(io.BytesIO(resp.content)).convert("RGB")
        full.save(path)

    concepts = concepts_of(img_dto)
    return {
        "path": path.relative_to(OUT).as_posix(),
        "concept": concept,
        "all_concepts": "|".join(concepts),
        "n_concepts": len(concepts),
        "n_boxes": len(img_dto.boundingBoxes or []),
        "mixed": int(len(concepts) > 1),
        "dive": dive_of(img_dto),
        "image_uuid": img_dto.uuid,
        "image_url": img_dto.url,
        "img_w": full.width,
        "img_h": full.height,
        "img_area": full.width * full.height,
        "imaging_type": img_dto.imagingType,
        "depth_m": img_dto.depthMeters,
    }


def load_taxa(path):
    """Read an {axis: [concept, ...]} mapping from a JSON config file."""
    with open(path) as fh:
        taxa = json.load(fh)
    if not isinstance(taxa, dict) or not all(
        isinstance(v, list) and all(isinstance(c, str) for c in v)
        for v in taxa.values()
    ):
        raise SystemExit(f"{path}: expected {{axis: [concept, ...]}}")
    return taxa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", required=True, help="JSON file of {axis: [concept, ...]}"
    )
    ap.add_argument("--per-taxon", type=int, default=PER_TAXON)
    ap.add_argument("--axis", action="append", help="only build these axes")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    session = requests.Session()

    taxa = load_taxa(args.config)
    axes = {k: v for k, v in taxa.items() if not args.axis or k in args.axis}
    for axis, concepts in axes.items():
        for concept in concepts:
            picked = collect(concept, args.per_taxon, seen)
            seen.update(i.uuid for i in picked)

            got = []
            with ThreadPoolExecutor(WORKERS) as ex:
                futures = {
                    ex.submit(fetch, i, concept, session): i for i in picked
                }
                for fut, img_dto in futures.items():
                    try:
                        got.append(fut.result())
                    except Exception as exc:  # a few framegrab URLs 404
                        print(f"  skip {img_dto.url}: {exc}")
            rows.extend(got)

            dives = len({r["dive"] for r in got})
            mixed = sum(r["mixed"] for r in got)
            print(
                f"{axis:20s} {concept:28s} {len(got):4d} images  "
                f"{dives:3d} dives  {mixed:4d} mixed"
            )

    if not rows:
        print("nothing collected")
        return

    manifest = OUT / "manifest.csv"
    if manifest.exists():  # a partial (--axis) run must not drop the rest
        fresh = {r["path"] for r in rows}
        with manifest.open(newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r["path"] not in fresh]

    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    taxa = {r["concept"] for r in rows}
    mixed = sum(int(r.get("mixed") or 0) for r in rows)
    print(
        f"\n{len(rows)} images across {len(taxa)} taxa, "
        f"{len({r.get('dive') for r in rows})} dives, "
        f"{mixed} multi-taxon frames -> {manifest}"
    )


if __name__ == "__main__":
    main()
