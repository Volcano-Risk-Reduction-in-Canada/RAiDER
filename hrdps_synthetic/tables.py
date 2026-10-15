"""CSV helpers for resumable manifests whose columns grow between versions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    path = Path(path)
    if not path.exists():
        return [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def rewrite(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    """Atomically rewrite a CSV with `fields` (missing values written as '')."""
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    tmp.replace(path)


def upgrade_header(path: Path, fields: Sequence[str]) -> list[str]:
    """Make an existing CSV's header contain `fields`, keeping any extra old columns. Returns the header."""
    old, rows = read_rows(path)
    if not old:
        return list(fields)
    merged = list(fields) + [c for c in old if c not in fields]
    if merged != old:
        rewrite(path, merged, rows)
    return merged


class CsvAppender:
    """Append rows to a CSV, upgrading an older header first so columns never misalign."""

    def __init__(self, path: Path, fields: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = upgrade_header(self.path, fields)
        new = not self.path.exists()
        self._f = open(self.path, 'a', newline='')
        self._w = csv.DictWriter(self._f, fieldnames=self.fields, extrasaction='ignore')
        if new:
            self._w.writeheader()

    def write(self, row: dict) -> None:
        self._w.writerow({k: row.get(k, '') for k in self.fields})
        self._f.flush()

    def close(self) -> None:
        self._f.close()
