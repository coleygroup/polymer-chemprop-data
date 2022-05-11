#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from glob import glob
import sys

import importlib
sys.path.append('calculations')
builder = importlib.import_module("make-polymers")

# -----------------------
# "Polymer" Fingerprints
# -----------------------

radius = 2
nBits = 2048

df = pd.read_csv('dataset.csv')

monoA = [Chem.MolFromSmiles(smi) for smi in df.loc[:, 'monoA']]
monoB = [Chem.MolFromSmiles(smi) for smi in df.loc[:, 'monoB']]
ptype = df.loc[:, 'poly_type']
comps = df.loc[:, 'comp']

fps_binary = []
fps_counts = []

c = 0
for mA, mB, pt, comp in zip(monoA, monoB, ptype, comps):
    
    polys = None
    if c % 100 == 0:
        print(c)
        
    nA = int(comp.split('_')[0][0])
    nB = int(comp.split('_')[1][0])
        
    if pt == 'alternating':
        polys = builder.create_all_alternating_copolymers(mA, mB, nA, nB)
    elif pt == 'block':
        polys = builder.create_all_block_copolymers(mA, mB, nA, nB, maxn=32)
    elif pt == 'random':
        polys = builder.create_n_random_copolymers(mA, mB, nA, nB, n=32)
        
        
    # Count fingerprints
    poly_fp_objs = np.array([AllChem.GetHashedMorganFingerprint(m, radius=radius, nBits=nBits) for m in polys])
    poly_fp_arrs = np.array([np.zeros((nBits,), dtype=np.float64)] * len(poly_fp_objs))
    _ = [DataStructs.ConvertToNumpyArray(fpo, fpa) for fpo, fpa in zip(poly_fp_objs, poly_fp_arrs)]
    polys_fps_cnt = poly_fp_arrs
    fp = np.mean(polys_fps_cnt, axis=0)
    fps_counts.append(fp)
    
    # Binary fingerprints
    polys_fps_bin = np.array([AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in polys]).astype(np.float64)
    fp = np.mean(polys_fps_bin, axis=0)
    fps_binary.append(fp)
    
    c += 1
    
    
X = pd.DataFrame(fps_binary, columns=[f'bit-{x}' for x in range(nBits)])
Y = df.loc[:, ['EA vs SHE (eV)', 'IP vs SHE (eV)']]
data = {'X':X, 'Y':Y}
with open('rf_input/dataset-poly_fps_binary.pkl', 'wb') as f:
    pickle.dump(data, f)

X = pd.DataFrame(fps_counts, columns=[f'bit-{x}' for x in range(nBits)])
Y = df.loc[:, ['EA vs SHE (eV)', 'IP vs SHE (eV)']]
data = {'X':X, 'Y':Y}
with open('rf_input/dataset-poly_fps_counts.pkl', 'wb') as f:
    pickle.dump(data, f)
    