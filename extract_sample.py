#!/usr/bin/env python3
"""Order-preserving systematic subsample of a large CIC-IDS CSV.

Usage:  python3 extract_sample.py IN.csv OUT.csv [N_ROWS=300000]

Keeps every k-th data row (k = total_rows // N_ROWS), so row order, timestamps
and the benign/attack ratio are preserved. The header is kept, stray repeated
header rows (present in some CSE-CIC-IDS2018 files) are dropped, and all
columns are kept unchanged. Streams in chunks, so memory use stays small.
Needs only pandas.
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000

# pass 1: count data rows (fast, byte-level)
total = -1  # minus header
with open(src, 'rb') as f:
    for block in iter(lambda: f.read(1 << 24), b''):
        total += block.count(b'\n')
k = max(1, total // n_target)
print(f'{total:,} data rows; keeping every {k}th (about {total // k:,} rows)')

# pass 2: stream and keep rows whose global index is a multiple of k
seen, kept, first = 0, 0, True
with open(dst, 'w', newline='') as out:
    for chunk in pd.read_csv(src, chunksize=200_000, dtype=str, low_memory=False):
        idx = pd.RangeIndex(seen, seen + len(chunk))
        seen += len(chunk)
        sel = chunk[(idx % k) == 0]
        # drop repeated header rows embedded in the data
        sel = sel[sel.iloc[:, -1] != chunk.columns[-1]]
        sel.to_csv(out, header=first, index=False)
        first = False
        kept += len(sel)
print(f'wrote {kept:,} rows to {dst}')
if 'Label' in chunk.columns:
    lab = pd.read_csv(dst, usecols=['Label'], dtype=str)['Label'].value_counts()
    print(lab.to_string())
