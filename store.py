"""DuckDB-backed store: one row per dataset image, with its metadata and embedding."""

from pathlib import Path

import duckdb
import numpy as np

DATASET_DIR = Path("dataset")
DB_PATH = DATASET_DIR / "deepsea.duckdb"
DIM = 512  # BioCLIP ViT-B/16 embedding width

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS images (
    path       VARCHAR PRIMARY KEY,  -- dataset/<concept>/<uuid>.png
    concept    VARCHAR,
    dive       VARCHAR,
    depth_m    DOUBLE,
    image_url  VARCHAR,
    embedding  FLOAT[{DIM}],         -- L2-normalized; NULL until embedded
    embed_mode VARCHAR               -- preprocess.MODE the embedding was made with
)
"""


class Store:
    def __init__(self, db_path=DB_PATH, read_only=False):
        self.con = duckdb.connect(str(db_path), read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA)

    def sync(self, dataset_dir=DATASET_DIR):
        """Add rows for new images, drop rows whose image is gone, refresh metadata."""
        pattern = f"{dataset_dir.as_posix()}/*/*.png"
        self.con.execute(f"""
            INSERT INTO images (path, concept)
            SELECT file, string_split(file, '/')[-2] FROM glob('{pattern}')
            ON CONFLICT DO NOTHING
        """)
        self.con.execute(
            f"DELETE FROM images WHERE path NOT IN (SELECT file FROM glob('{pattern}'))"
        )
        manifest = dataset_dir / "manifest.csv"
        if manifest.exists():
            self.con.execute(f"""
                UPDATE images
                SET dive = m.dive, depth_m = m.depth_m, image_url = m.image_url
                FROM read_csv('{manifest.as_posix()}') m
                WHERE images.path = '{dataset_dir.as_posix()}/' || m.path
            """)

    def pending(self, mode):
        """Paths with no embedding, or one made under a different preprocessing mode."""
        rows = self.con.execute(
            "SELECT path FROM images WHERE embedding IS NULL OR embed_mode <> ? ORDER BY path",
            [mode],
        ).fetchall()
        return [r[0] for r in rows]

    def put(self, items, mode):
        """Store [(path, normalized 1-D tensor), ...] embedded under `mode`."""
        if not items:
            return
        # binding Python lists costs ~40 ms per vector, so hand DuckDB flat numpy
        # columns it can scan directly and rebuild each row's array in SQL
        paths = np.array([path for path, _ in items])
        mat = np.stack([vec.cpu().numpy() for _, vec in items]).astype(np.float32)
        flat = {  # noqa: F841 -- read by name from the SQL below
            "path": np.repeat(paths, DIM),
            "i": np.tile(np.arange(DIM), len(paths)),
            "val": mat.ravel(),
        }
        self.con.execute(f"""
            UPDATE images
            SET embedding = new.vec, embed_mode = $mode
            FROM (
                SELECT path, array_agg(val ORDER BY i)::FLOAT[{DIM}] AS vec
                FROM flat GROUP BY path
            ) new
            WHERE images.path = new.path
        """, {"mode": mode})

    def counts(self, mode):
        """(embedded with `mode`, embedded with some other mode)."""
        return self.con.execute(
            "SELECT count(*) FILTER (embed_mode = $m), count(*) FILTER (embed_mode <> $m) FROM images",
            {"m": mode},
        ).fetchone()

    def search(self, vec, k, concept=None, min_depth=None, max_depth=None):
        """Top-k (path, concept, score) by cosine similarity to normalized `vec`."""
        return self.con.cursor().execute(f"""
            SELECT path, concept, array_inner_product(embedding, $q::FLOAT[{DIM}]) AS score
            FROM images
            WHERE embedding IS NOT NULL
              AND ($concept IS NULL OR concept = $concept)
              AND ($min_depth IS NULL OR depth_m >= $min_depth)
              AND ($max_depth IS NULL OR depth_m <= $max_depth)
            ORDER BY score DESC
            LIMIT $k
        """, {"q": vec.tolist(), "k": k, "concept": concept,
              "min_depth": min_depth, "max_depth": max_depth}).fetchall()
