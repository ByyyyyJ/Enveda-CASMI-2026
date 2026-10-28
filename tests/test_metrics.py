"""Cases from the official metric's doctests: local MRR must agree with the scorer."""

import pandas as pd
import pytest

from casmi.chem.canon import dedupe_by_identity, ik14
from casmi.eval.metrics import answer_ranks, mrr, paired_delta, summary

SOLUTION = {"m_1": "CCO", "m_2": "c1ccccc1"}


def test_reciprocal_rank():
    preds = {"m_1": ["CCO", "CC(=O)O"], "m_2": ["CC(=O)O", "c1ccccc1"]}
    assert mrr(answer_ranks(SOLUTION, preds)) == 0.75


def test_first_hit_only_and_misses():
    preds = {"m_1": ["CC(=O)O"], "m_2": ["c1ccccc1"] * 3}
    assert mrr(answer_ranks(SOLUTION, preds)) == 0.5


def test_missing_molecule_scores_zero():
    assert mrr(answer_ranks(SOLUTION, {"m_1": ["CCO"]})) == 0.5


def test_stereo_ignored():
    ranks = answer_ranks({"m_1": "C[C@H](N)C(=O)O"}, {"m_1": ["CC(N)C(=O)O"]})
    assert ranks["m_1"] == 1


def test_stereo_stripped_before_tautomer_canonicalization():
    # The official regression case: this stereocentre pins the enone double bond.
    assert ik14("C=C(C)C1CCC2CCC(=O)C=C2C1") == ik14("C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1")


def test_invalid_guess_is_a_miss():
    ranks = answer_ranks({"m_1": "CCO"}, {"m_1": ["not a molecule", "CCO"]})
    assert ranks["m_1"] == 2


def test_rank_cap():
    preds = {"m_1": ["C"] * 25 + ["CCO"]}
    assert answer_ranks({"m_1": "CCO"}, preds)["m_1"] == 0


def test_unparseable_answer_raises():
    with pytest.raises(ValueError):
        answer_ranks({"m_1": "not a molecule"}, {"m_1": ["CCO"]})


def test_dedupe_by_identity():
    assert dedupe_by_identity(["OCC", "CCO", "xx", "C[C@H](N)C(=O)O", "CC(N)C(=O)O"]) == [
        "OCC",
        "C[C@H](N)C(=O)O",
    ]


def test_summary_and_paired_delta():
    a = pd.Series({"x": 1, "y": 0, "z": 2})
    b = pd.Series({"x": 1, "y": 1, "z": 2})
    s = summary(a)
    assert s["n"] == 3 and s["mrr"] == pytest.approx(0.5)
    d = paired_delta(a, b)
    assert d["delta"] == pytest.approx(1 / 3) and d["n_changed"] == 1
