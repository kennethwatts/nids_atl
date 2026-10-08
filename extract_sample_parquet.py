#!/usr/bin/env python3
"""Order-preserving systematic subsample of a large Parquet (or CSV) file, written as CSV.

Usage:  python3 extract_sample_parquet.py IN.parquet OUT.csv [N_ROWS=150000]

Keeps every k-th row (k = total_rows // N_ROWS), so row order, timestamps and the
benign/attack ratio are preserved; all columns are kept unchanged. Reads the Parquet
file in batches, so memory use stays small. Needs pyarrow (pip install pyarrow) and pandas.
"""
import sys
import pyarrow.parquet as pq

src, dst = sys.argv[1], sys.argv[2]
n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000

pf = pq.ParquetFile(src)
total = pf.metadata.num_rows
k = max(1, total // n_target)
print(f'{total:,} rows; keeping every {k}th (about {total // k:,} rows)')

seen, kept, first = 0, 0, True
with open(dst, 'w', newline='') as out:
    for batch in pf.iter_batches(batch_size=200_000):
        df = batch.to_pandas()
        sel = df[(df.index + seen) % k == 0]
        seen += len(df)
        sel.to_csv(out, header=first, index=False)
        first = False
        kept += len(sel)
print(f'wrote {kept:,} rows to {dst}')
for c in ('Label', 'Attack'):
    if c in df.columns:
        import pandas as pd
        print(pd.read_csv(dst, usecols=[c])[c].value_counts().head(12).to_string())
