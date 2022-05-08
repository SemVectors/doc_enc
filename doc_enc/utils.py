#!/usr/bin/env python3

from typing import Union
from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path


def _is_gzipped(fp):
    if isinstance(fp, Path):
        n = fp.name
    elif isinstance(fp, str):
        n = fp
    else:
        raise RuntimeError("logic error 82093")

    return n.endswith('.gz')


def open_bin_file(fp: Union[Path, str]):
    if _is_gzipped(fp):
        return GzipFile(fp, mode='rb')
    return open(fp, mode='rb')


def open_file(fp: Union[Path, str]):
    if _is_gzipped(fp):
        return TextIOWrapper(GzipFile(fp, 'rb'), encoding='utf8')
    return open(fp, 'rt', encoding='utf8')


def find_file(fp: Union[Path, str], throw_if_not_exist=True):
    if isinstance(fp, Path):
        sp = str(fp)
    elif isinstance(fp, str):
        sp = fp
        fp = Path(fp)
    else:
        raise RuntimeError("logic error 82094")

    np = Path(f"{sp}.gz")
    if np.exists():
        return np
    if fp.exists():
        return fp

    if throw_if_not_exist:
        raise RuntimeError(f"Failed to find {fp}[.gz]")
    return fp


def calc_line_cnt(fp):
    with open_bin_file(fp) as f:
        i = -1
        for i, _ in enumerate(f):
            pass
    return i + 1
