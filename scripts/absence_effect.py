"""Effect of the pre-registered absence rules on runs scored by absence_calib.py.

Flag: candidate has same-adduct reference spectra and the best entropy similarity to them is < THR
(THR fixed at 0.2 from the np-examples false-fire rate, 0.4%; not tuned here).
(a) leader: a flagged rank-1 candidate swaps with rank 2.
(b) top10: within the first 10 places flagged candidates go after the unflagged ones (stable).

Usage: python scripts/absence_effect.py RUN [RUN ...]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402

THR = 0.2


def rr(order, truth):
    return next((1 / (i + 1) for i, t in enumerate(order) if t), 0.0)


def rules(g):
    flag = ((g.n_ref_same_adduct > 0) & (g.ref_max < THR)).tolist()
    idx = list(range(len(g)))
    a = idx[:]
    if len(a) > 1 and flag[0]:
        a[0], a[1] = a[1], a[0]
    head = idx[:10]
    b = [i for i in head if not flag[i]] + [i for i in head if flag[i]] + idx[10:]
    return a, b


rows = []
for run in sys.argv[1:]:
    c = pd.read_parquet(EXTERNAL / "absence" / f"{run}.parquet").sort_values(["molecule_id", "rank"])
    base = da = db = 0.0
    ca = cb = hurt_a = hurt_b = 0
    n = 0
    for _, g in c.groupby("molecule_id"):
        t = g.is_true.tolist()
        a, b = rules(g)
        r0, ra, rb = rr(t, t), rr([t[i] for i in a], t), rr([t[i] for i in b], t)
        base += r0; da += ra - r0; db += rb - r0
        ca += a != list(range(len(g))); cb += b != list(range(len(g)))
        hurt_a += ra < r0; hurt_b += rb < r0
        n += 1
    rows.append(dict(run=run, n=n, mrr=base / n, changed_a=ca, delta_a=da / n, hurt_a=hurt_a,
                     changed_b=cb, delta_b=db / n, hurt_b=hurt_b))
print(pd.DataFrame(rows).round(4).to_string(index=False))
