"""Reading a directory of parquet files as one dataset and splitting it into train/val/test
by whole files, without touching the files on disk."""

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def is_valid_file(path):
    path = Path(path)
    # Check if file extension is in parquet format
    if path.suffix != ".parquet":
        raise ValueError(f"File must be in .parquet format, got: {path.suffix}")
    return path.is_file() and path.stat().st_size > 0


def natural_sort_key(path):
    """Sort key putting rep2 before rep10."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", Path(path).name)]


def list_parquet_files(source) -> list[Path]:
    """Resolve a parquet file, a directory of parquet files, or a list of files to a list of files.

    A directory gives its *.parquet files in natural order; the dataset-level `_metadata` and
    `_common_metadata` sidecars have no suffix and are skipped."""
    if isinstance(source, (list, tuple)):
        files = [Path(f) for f in source]
    elif Path(source).is_dir():
        files = sorted(Path(source).glob("*.parquet"), key=natural_sort_key)
        if not files:
            raise FileNotFoundError(f"No .parquet files in directory {source}")
    else:
        files = [Path(source)]
    for f in files:
        assert is_valid_file(f), f"Invalid file: {f}"
    return files


def read_parquet_files(files, columns=None) -> pa.Table:
    """Read `files` into one table, concatenated in the given order (event indices depend on it)."""
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


SPLITS = ("train", "val", "test")


def split_parquet_files(data_dir, split, seed=None) -> dict[str, list[Path]]:
    """Split the parquet files of `data_dir` into train/val/test lists of whole files.

    `split` has three entries, either all ints (number of files per split; leftover files are
    unused) or fractions summing to 1 (val and test get round(fraction * n_files) files, at least
    one when their fraction is non-zero, and train the rest). The file list is shuffled with
    `seed` before slicing; seed=None keeps the natural order. Nothing on disk is moved.
    """
    files = list_parquet_files(data_dir)
    n_files = len(files)
    if len(split) != len(SPLITS):
        raise ValueError(f"split needs {len(SPLITS)} entries {SPLITS}, got {split}")
    if any(v < 0 for v in split):
        raise ValueError(f"split entries must be non-negative, got {split}")

    if all(isinstance(v, int) for v in split):
        n_per_split = list(split)
        if sum(n_per_split) > n_files:
            raise ValueError(f"split {split} asks for {sum(n_per_split)} files but {data_dir} has {n_files}")
    else:
        if not np.isclose(sum(split), 1.0):
            raise ValueError(f"split fractions must sum to 1 (or be all ints = file counts), got {split}")
        n_val, n_test = (max(1, round(f * n_files)) if f > 0 else 0 for f in split[1:])
        n_per_split = [n_files - n_val - n_test, n_val, n_test]
        if split[0] > 0 and n_per_split[0] < 1:
            raise ValueError(f"split {split} leaves no training file out of {n_files}")

    if seed is not None:
        files = [files[i] for i in np.random.default_rng(seed).permutation(n_files)]
    bounds = np.cumsum([0, *n_per_split])
    return {name: files[lo:hi] for name, lo, hi in zip(SPLITS, bounds[:-1], bounds[1:])}
