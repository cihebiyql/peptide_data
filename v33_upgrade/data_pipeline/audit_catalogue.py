import sqlite3
import json
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    print("=== TABLES & VIEWS ===")
    for t, name, sql in cur.execute("SELECT type, name, sql FROM sqlite_master WHERE type IN ('table','view')"):
        print(f"\n[{t.upper()}] {name}")
        print(sql)

    print("\n=== TABLE COUNTS ===")
    for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cnt = cur.execute(f"SELECT COUNT(*) FROM \"{name}\"").fetchone()[0]
        print(f"  {name}: {cnt:,}")

    print("\n=== TARGET ENDPOINT CANDIDATES ===")
    # Inspect all endpoints with >= 10 rows or matching keywords
    rows = cur.execute("""
        SELECT endpoint, lane, COUNT(*) as n
        FROM records
        GROUP BY endpoint, lane
    """).fetchall()
    from collections import defaultdict
    by_ep = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for ep, lane, n in rows:
        by_ep[ep][lane] += n
        totals[ep] += n

    kw = [
        'pampa', 'caco', 'mdck', 'rrck', 'bbb', 'blood', 'brain', 'permeab', 'papp',
        'logd', 'logp', 'solub',
        'ppb', 'protein_bind', 'plasma_protein', 'fu', 'unbound',
        'half', 't1/2', 't_half', 'stability',
        'cl', 'clear', 'vd', 'volume', 'bioavail', 'cmax', 'auc', 'tmax', 'mrt',
        'hemol', 'hc50', 'hc10', 'mhc', 'tox', 'cytotox', 'cc50', 'lc50', 'ld50',
        'mic', 'mbc', 'mfc'
    ]
    print("\nEndpoints matching keywords or total >= 200 (sorted by total desc):")
    selected = []
    for ep, tot in sorted(totals.items(), key=lambda x: -x[1]):
        epl = (ep or '').lower()
        if tot >= 200 or any(k in epl for k in kw):
            if '%' in epl and tot < 50:
                continue
            selected.append((ep, tot, dict(by_ep[ep])))
    for ep, tot, d in selected[:180]:
        s = ", ".join(f"{k}:{v}" for k, v in sorted(d.items()))
        print(f"  {ep!r:40s} total={tot:7d} | {s}")
    print(f"Total matching endpoints shown: {min(len(selected), 180)} / {len(selected)}")

if __name__ == "__main__":
    main()
