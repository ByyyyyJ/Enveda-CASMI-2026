"""Structure identity exactly as the official CASMI metric computes it.

Mirrors `canonicalize_smiles_to_ik14` and `heavy_atom_composition` from the official metric
notebook (https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank, pulled 2026-10-26):
strip stereochemistry, then RDKit TautomerEnumerator canonicalization, then InChIKey[:14].
The stereo strip must come first: a stereocentre can pin a double bond and change which
tautomer the enumerator picks. Requires RDKit 2026.03.3, the version the answer key used.
"""

import warnings
from functools import lru_cache

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

EXPECTED_RDKIT_VERSION = "2026.03.3"

RDLogger.DisableLog("rdApp.*")
if rdkit.__version__ != EXPECTED_RDKIT_VERSION:
    warnings.warn(
        f"RDKit {rdkit.__version__} differs from the metric's {EXPECTED_RDKIT_VERSION}; "
        "tautomer canonicalization may disagree with the scorer.",
        UserWarning,
    )

_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=1_000_000)
def ik14(smiles: str) -> str | None:
    """Metric identity of a SMILES: tautomer-canonical, stereo-free InChIKey14, or None."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(_tautomer_canonicalizer.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def heavy_atom_composition(smiles: str) -> tuple | None:
    """Heavy-atom element counts; equal for any two SMILES sharing an ik14 (cheap pre-gate)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "H":
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


def dedupe_by_identity(smiles_list: list[str]) -> list[str]:
    """Keep the first SMILES of each metric identity, dropping unparseable ones; order preserved."""
    seen: set[str] = set()
    out = []
    for s in smiles_list:
        key = ik14(s)
        if key is not None and key not in seen:
            seen.add(key)
            out.append(s)
    return out
