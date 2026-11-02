"""Same-formula isomer scorer: spectrum encoder, structure encoder and their shared preprocessing.

The state-dict layout matches scripts/isomer_train.py (Model with .s, .m, .logit_scale).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
NPEAK, NBIT, D = 64, 2048, 256
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBIT)


def fourier(x, n=32, lo=0.01, hi=2000.0):
    freqs = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n, device=x.device))
    a = x.unsqueeze(-1) * freqs
    return torch.cat([a.sin(), a.cos()], -1)


class SpecEnc(nn.Module):
    def __init__(self, n_add):
        super().__init__()
        self.inp = nn.Linear(64 * 2 + 1, D)
        self.add = nn.Embedding(n_add, D)
        self.tims = nn.Embedding(2, D)
        self.prec = nn.Linear(64, D)
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D, D)

    def forward(self, mz, it, prec, add, tims):
        pad = it <= 0
        x = self.inp(torch.cat([fourier(mz), fourier(prec[:, None] - mz), it[..., None]], -1))
        cls = (self.add(add) + self.tims(tims) + self.prec(fourier(prec)))[:, None]
        h = self.tf(torch.cat([cls, x], 1), src_key_padding_mask=torch.cat([torch.zeros_like(pad[:, :1]), pad], 1))
        return F.normalize(self.out(h[:, 0]), dim=-1)


class StructEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NBIT, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, D))

    def forward(self, f):
        return F.normalize(self.net(torch.log1p(f.float())), dim=-1)


class Model(nn.Module):
    def __init__(self, n_add):
        super().__init__()
        self.s, self.m = SpecEnc(n_add), StructEnc()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))


def prep_peaks(mzs, its, prec):
    """Top NPEAK peaks below the precursor, sqrt-scaled intensities; zero-padded."""
    m = np.asarray(mzs, np.float32); a = np.asarray(its, np.float32)
    keep = m < prec - 1.6
    m, a = m[keep], a[keep]
    if len(m) > NPEAK:
        top = np.argpartition(-a, NPEAK)[:NPEAK]
        m, a = m[top], a[top]
    out_m = np.zeros(NPEAK, np.float32); out_a = np.zeros(NPEAK, np.float32)
    if len(a):
        out_m[:len(m)] = m; out_a[:len(a)] = np.sqrt(a / a.max())
    return out_m, out_a


def morgan_counts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return np.minimum(_gen.GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
