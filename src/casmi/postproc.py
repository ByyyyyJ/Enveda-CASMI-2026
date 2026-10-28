"""Post-processing of the public pipeline's final answer file, using its per-candidate evidence.

main_first: within each molecule's final 25, candidates that are in the Engine 1 main list are put
first, in main-list order; the rest keep their relative order after them. On f0np (fold-0 natural
products whose answers are pool structures) the main list alone beats the fused order.
"""

import pandas as pd


def main_first(sub: pd.DataFrame, evidence: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Reorder `sub` (molecule_id, smiles 'a;b;...') by `evidence` (molecule_id, smiles, main_rank).

    `top`: only candidates with main_rank <= top are promoted (None = all in the main list).
    """
    mr = {(m, s): int(r) for m, s, r in zip(evidence["molecule_id"], evidence["smiles"],
                                           evidence["main_rank"]) if r and r > 0}
    out = []
    for mid, smis in zip(sub["molecule_id"], sub["smiles"]):
        guesses = [g for g in str(smis).split(";") if g]

        def key(ig):
            i, g = ig
            r = mr.get((mid, g))
            if r is not None and (top is None or r <= top):
                return (0, r, i)
            return (1, 0, i)

        out.append(";".join(g for _, g in sorted(enumerate(guesses), key=key)))
    return pd.DataFrame({"molecule_id": sub["molecule_id"], "smiles": out})
