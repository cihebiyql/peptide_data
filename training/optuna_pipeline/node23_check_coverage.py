#!/usr/bin/env python3
"""Check how many compounds in integrated data are missing from feature cache."""
import csv, sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)

INTEG = Path('/data/qlyu/s3_integrated')
CACHE = Path('/data/qlyu/s3/feat_cache')

with open(CACHE / 'cids.json') as f:
    cached_cids = set(json.load(f))

print(f"Cached compounds: {len(cached_cids):,}")

# Check each endpoint
total_missing = set()
for fp in sorted(INTEG.glob('*__clean.tsv')):
    ep = fp.stem.replace('__clean', '')
    cids = set()
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            if cid:
                cids.add(cid)
    missing = cids - cached_cids
    total_missing |= missing
    if missing or len(cids) > 100:
        print(f"  {ep:<22} total={len(cids):>6,}  missing={len(missing):>6,}  coverage={100*(len(cids)-len(missing))/max(len(cids),1):.0f}%")

print(f"\nTotal unique missing compounds: {len(total_missing):,}")
print(f"Total unique compounds in integrated data: {len(total_missing | (total_missing & cached_cids)):,}")

# Save missing compound IDs for feature computation
with open('/data/qlyu/s3/missing_cids.json', 'w') as f:
    json.dump(sorted(total_missing), f)
print(f"Missing compound IDs saved to /data/qlyu/s3/missing_cids.json")
