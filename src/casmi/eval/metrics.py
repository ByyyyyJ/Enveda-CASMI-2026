"""Local MRR@25 matching the official CASMI metric, plus per-molecule ranks and bootstrap CIs.

`reciprocal_ranks` reproduces the official score molecule by molecule (same identity function,
same first-hit rule, unparseable guesses count as misses), so `mrr(...)` equals the leaderboard
number on the same answers. The rest is for comparing pipelines on small validation sets.
"""

import numpy as np
import pandas as pd

from casmi.chem.canon import heavy_atom_composition, ik14

MAX_RANK = 25


def answer_ranks(
    solution: dict[str, str | list[str]], predictions: dict[str, list[str]], max_rank: int = MAX_RANK
) -> pd.Series:
    """1-based rank of the first correct guess per molecule (0 = not in the top `max_rank`).

    `solution` maps molecule_id to an answer SMILES or a list of accepted SMILES.
    `predictions` maps molecule_id to ranked guesses; missing molecules count as misses.
    """
    ranks = {}
    for mol_id, answers in solution.items():
        answers = [answers] if isinstance(answers, str) else list(answers)
        accept = {k for k in map(ik14, answers) if k}
        if not accept:
            raise ValueError(f"answer SMILES for {mol_id} cannot be parsed: {answers!r}")
        accept_keys = {heavy_atom_composition(a) for a in answers}
        rank = 0
        for r, guess in enumerate(predictions.get(mol_id, [])[:max_rank], 1):
            if heavy_atom_composition(guess) in accept_keys and ik14(guess) in accept:
                rank = r
                break
        ranks[mol_id] = rank
    return pd.Series(ranks, name="rank", dtype=int)


def reciprocal_ranks(ranks: pd.Series) -> pd.Series:
    rr = 1.0 / ranks.where(ranks > 0)
    return rr.fillna(0.0).rename("rr")


def mrr(ranks: pd.Series) -> float:
    return float(reciprocal_ranks(ranks).mean())


def summary(ranks: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """MRR with a 95% bootstrap CI over molecules, and Hit@1/5/25."""
    rr = reciprocal_ranks(ranks).to_numpy()
    rng = np.random.default_rng(seed)
    boot = rr[rng.integers(0, len(rr), size=(n_boot, len(rr)))].mean(axis=1)
    hit = lambda k: float(((ranks > 0) & (ranks <= k)).mean())  # noqa: E731
    return {
        "n": len(rr),
        "mrr": float(rr.mean()),
        "mrr_lo": float(np.quantile(boot, 0.025)),
        "mrr_hi": float(np.quantile(boot, 0.975)),
        "hit@1": hit(1),
        "hit@5": hit(5),
        "hit@25": hit(MAX_RANK),
    }


def paired_delta(ranks_a: pd.Series, ranks_b: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """MRR(b) - MRR(a) on shared molecules, with a paired bootstrap 95% CI."""
    common = ranks_a.index.intersection(ranks_b.index)
    d = (reciprocal_ranks(ranks_b[common]) - reciprocal_ranks(ranks_a[common])).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {
        "n": len(d),
        "delta": float(d.mean()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "n_changed": int((d != 0).sum()),
    }


def read_submission(path) -> dict[str, list[str]]:
    sub = pd.read_csv(path)
    return {
        m: [g.strip() for g in str(s).split(";") if g.strip()]
        for m, s in zip(sub["molecule_id"], sub["smiles"])
    }
