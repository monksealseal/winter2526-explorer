"""D6: extract the monthly PNA file from the New Downloaded Files zip-003
into data/indices/pna_monthly.txt, with a normalized format the parser in
indices.py can read directly.

Source inside zip:
    Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/
        norm.pna.monthly.b5001.current.ascii

This file is NOAA CPC's normalized monthly PNA series, base 1950-current,
layout ``year month value`` (whitespace-delimited, no header). It extends
from January 1950 to the present month.

We copy it verbatim to data/indices/pna_monthly.txt so the raw file is
checked in alongside ao.csv/nao.csv/pna.txt/qbo.csv.

Usage (from repo root):
    python scripts/d6_extract_pna_monthly.py
"""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW  = ROOT / "New Downloaded Files"
DST  = ROOT / "data" / "indices" / "pna_monthly.txt"
ZIP_003 = NEW / "Gen Circ_ Group 2 Subgroup 2-20260418T201632Z-3-003.zip"
MEMBER  = "Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/norm.pna.monthly.b5001.current.ascii"


def main() -> None:
    if not ZIP_003.exists():
        raise FileNotFoundError(ZIP_003)
    DST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_003) as zf, zf.open(MEMBER) as src, open(DST, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    # Quick sanity check on the extracted file.
    with open(DST, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    first, last = lines[0].split(), lines[-1].split()
    print(f"wrote {DST}  ({DST.stat().st_size} bytes, {len(lines)} rows)")
    print(f"  first row: {first}")
    print(f"  last row : {last}")


if __name__ == "__main__":
    main()
