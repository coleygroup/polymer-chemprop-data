#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
import numpy as np
import argparse
from copy import deepcopy
import os
from itertools import permutations
from datetime import datetime

import signal
from contextlib import contextmanager


def parse_inputs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--acids", dest='acids', help='CSV file with list of boronic acids')
    parser.add_argument("--bromides", dest='bromides', help='CSV file with list of organo bromides')
    parser.add_argument("--iAs", dest='iAs', help='Index of acids to consider; list all wanted from 0 to 8', nargs='+', type=int, default=0)
    parser.add_argument("--iBs", dest='iBs', help='Index of acids to consider. List all wanted from 0 to 681. -1 equals all', nargs='+', type=int, default=-1)
    parser.add_argument("--nA", dest='nA', help='Number of A monomers', type=int, default=4)
    parser.add_argument("--nB", dest='nB', help='Number of B monomers', type=int, default=4)
    parser.add_argument("--poly_types", dest='poly_types', 
                        help='Type of polymer sequences to generate: block, alternating, random. Default is all.', 
                        nargs='+', type=str, default=["block", "alternating", "random"], choices=["block", "alternating", "random"])
    parser.add_argument("--nconfs", dest='nconfs', help='Number of conformers per polymer sequence', default=8, type=int)
    parser.add_argument("--maxn", dest='maxn', help='Max number of polymers sequences to generate', default=32, type=int)
    parser.add_argument("--ncpu", dest='ncpu', help='Number of CPU cores to use', default=8, type=int)
    args = parser.parse_args()
    
    if args.nA != args.nB and "alternating" in args.poly_types:
        raise ValueError("cannot select 'alternating' co-polymer if nA != nB")
    
    return args


# ================
# Helper Functions
# ================
def make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError
    

# ===============
# RDKit Functions
# ===============
def rm_duplicate_mols(mols):
    smiles = list(set([Chem.MolToSmiles(m, canonical=True) for m in mols]))
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return mols


def rm_mol_without_CBr(mols):
    for i, m in enumerate(mols):
        patt = Chem.MolFromSmarts('cCBr')
        if m.HasSubstructMatch(patt) is False:
            _ = mols.pop(i)
    return mols


def protect_CBr(m):
    while m.HasSubstructMatch(Chem.MolFromSmarts('cCBr')):
        smarts = "[*:1]CBr>>[*:1]C[At]"
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((m,))
        products = rm_duplicate_mols([m[0] for m in ps])
        m = products[0]
    return m


def deprotect_CBr(m):
    while m.HasSubstructMatch(Chem.MolFromSmarts('C[At]')):
        smarts = "[*:1]C[At]>>[*:1]CBr"
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((m,))
        products = rm_duplicate_mols([m[0] for m in ps])
        m = products[0]
    return m


def BOO_Br_bond(m1, m2, bothways=False):
    m1 = protect_CBr(m1)
    m2 = protect_CBr(m2)
    
    smarts = "[*:1]([B](-O)(-O)).[*:2]Br>>[*:1]-[*:2]"
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants((m1,m2))
    products = rm_duplicate_mols([m[0] for m in ps])
    products = [deprotect_CBr(p) for p in products]
    
    if len(products) == 0 and bothways is True:    
        ps = rxn.RunReactants((m2,m1))
        products = rm_duplicate_mols([m[0] for m in ps])
        products = [deprotect_CBr(p) for p in products]
    
    return products


def Br_Br_bond(m1, m2):
    
    m1 = protect_CBr(m1)
    m2 = protect_CBr(m2)
    
    smarts = "[*:1]Br.[*:2]Br>>[*:1]-[*:2]"
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants((m1,m2))
    products = rm_duplicate_mols([m[0] for m in ps])
    
    products = [deprotect_CBr(p) for p in products]
    return products


def BOO_BOO_bond(m1, m2):
    smarts = "[*:1]([B](-O)(-O)).[*:2]([B](-O)(-O))>>[*:1]-[*:2]"
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants((m1,m2))
    products = rm_duplicate_mols([m[0] for m in ps])
    return products


def rm_termini(m):
    
    # rm all Br
    m = protect_CBr(m)
    while m.HasSubstructMatch(Chem.MolFromSmarts('cBr')):
        smarts = "[*:1]Br>>[*:1]"
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((m,))
        products = rm_duplicate_mols([m[0] for m in ps])
        m = products[0]
    m = deprotect_CBr(m)
    
    # rm all BOO
    while m.HasSubstructMatch(Chem.MolFromSmarts('[B](-O)(-O)')):
        smarts = "[*:1]([B](-O)(-O))>>[*:1]"
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((m,))
        products = rm_duplicate_mols([m[0] for m in ps])
        m = products[0]
        
    return m


def rm_one_Br(m):
    m = protect_CBr(m)
    smarts = "[*:1]Br>>[*:1]"
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants((m,))
    products = rm_duplicate_mols([m[0] for m in ps])
    m = products[0]
    m = deprotect_CBr(m)
    return m


def rm_one_BOO(m):
    smarts = "[*:1]([B](-O)(-O))>>[*:1]"
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants((m,))
    products = rm_duplicate_mols([m[0] for m in ps])
    m = products[0]
    return m


def create_block_copolymer(m1, m2):
    """returns one block co-polymer sampled at random
    """
    blockA = deepcopy(m1)
    for _ in range(3):
        ps = BOO_BOO_bond(blockA, m1)
        # we may have multiple products ==> pick one at random
        blockA = np.random.choice(ps)
        
    blockB = deepcopy(m2)
    for _ in range(3):
        ps = Br_Br_bond(blockB, m2)
        # we may have multiple products ==> pick one at random
        blockB = np.random.choice(ps)
    
    # link the 2 blocks
    ps = BOO_Br_bond(blockA, blockB)
    poly = np.random.choice(ps)
    
    # rm termini
    poly = rm_termini(poly)
    
    return poly


def create_all_block_copolymers(m1, m2, n1, n2, maxn=32):
    """returns a list of all possible block co-polymers
    """
    # check n >= 2, otherwise there is no "block"
    assert n1 >= 2
    assert n2 >= 2
    
    def extend_A(oligomers, monomer):
        products = []
        for m in oligomers:
            ps = BOO_BOO_bond(m, monomer)
            products.extend(ps)
        return products
    
    def extend_B(oligomers, monomer):
        products = []
        for m in oligomers:
            ps = Br_Br_bond(m, monomer)
            products.extend(ps)
        return products
            
    blockA = deepcopy(m1) # len 1
    blockA = BOO_BOO_bond(blockA, m1) # len 2
    for i in range(n1 - 2):
        blockA = extend_A(blockA, m1)
    
    blockB = deepcopy(m2) # len 1
    blockB = Br_Br_bond(blockB, m2) # len 2
    for i in range(n2 - 2):
        blockB = extend_B(blockB, m2)
    
    # link the 2 blocks
    polys = []
    for bA in blockA:
        for bB in blockB:
            ps = BOO_Br_bond(bA, bB)
            polys.extend(ps)
    
    # rm termini
    polys = [rm_termini(p) for p in polys]
    # rm duplicates
    polys = rm_duplicate_mols(polys)
    
    # allow maxN polymers
    if len(polys) > maxn:
        polys = np.random.choice(polys, size=maxn, replace=False)
    
    return polys


def create_alternating_copolymer(m1, m2):
    """returns one alternating co-polymer sampled at random
    """
    poly = deepcopy(m1)
    for m in [m2, m1, m2, m1, m2, m1, m2]:
        ps = BOO_Br_bond(poly, m, bothways=True)
        # we may have multiple products ==> pick one at random
        poly = np.random.choice(ps)
    return rm_termini(poly)


def create_all_alternating_copolymers(m1, m2, n1, n2):
    """returns all possible alternating co-polymers
    """
    assert n1 == n2
    # create list of [m1, m2, m1, m2 ...]
    l1 = [m1] * n1
    l2 = [m2] * n2
    ms = [val for pair in zip(l1, l2) for val in pair]
    # start poly with ms[0]
    polys = [ms.pop(0)]
    # then continue to append...
    for m in ms:
        new_polys = []
        for poly in polys:
            ps = BOO_Br_bond(poly, m, bothways=True)
            new_polys.extend(ps)
        polys = new_polys
        
    polys = [rm_termini(p) for p in polys]
    polys = rm_duplicate_mols(polys)
    return polys


def create_random_copolymer(m1, m2, balance_composition=True):
    """returns one random co-polymer sampled at random
    """
    
    if balance_composition is True:
        monomers = [m1, m2, m1, m2, m1, m2, m1, m2]
        choices = list(permutations(list(range(len(monomers)))))
        i = np.random.randint(len(choices))
        idx_list = list(choices[i])
        
        idx = idx_list.pop(0)
        mod = idx % 2
        poly = deepcopy(monomers[idx])
        
        for idx in idx_list:
            next_mod = idx % 2
            m = monomers[idx]
            
            if mod == next_mod:
                if mod == 0:
                    ps = BOO_BOO_bond(poly, m)
                elif mod == 1:
                    ps = Br_Br_bond(poly, m)
            else:
                ps = BOO_Br_bond(poly, m, bothways=True)
                
            # we may have multiple products ==> pick one at random
            poly = np.random.choice(ps)
            # update idx
            mod = next_mod

    else:
        monomers = [m1, m2]
        idx = np.random.choice([0,1])
        poly = deepcopy(monomers[idx])
        
        for _ in range(7):
            next_idx = np.random.choice([0,1])
            m = monomers[next_idx]
        
            if idx == next_idx:
                if idx == 0:
                    ps = BOO_BOO_bond(poly, m)
                elif idx == 1:
                    ps = Br_Br_bond(poly, m)
            else:
                ps = BOO_Br_bond(poly, m, bothways=True)
                
            # we may have multiple products ==> pick one at random
            poly = np.random.choice(ps)
            # update idx
            idx = next_idx
    
    return rm_termini(poly)


def create_n_random_copolymers(m1, m2, n1=4, n2=4, n=32):
    """returns N random co-polymers sampled at random
    """
    
    monomers = [m1] * n1 + [m2] * n2
    smi1 = Chem.MolToSmiles(m1, canonical=True)
    smi2 = Chem.MolToSmiles(m2, canonical=True)
    
    # if the 2 monomers are the same, chances are we won't be able to create many random co-polymers
    check_npoly = True
    if Chem.MolToSmiles(rm_termini(m1), canonical=True) == Chem.MolToSmiles(rm_termini(m2), canonical=True):
        print(f"Monomer A and B are the same: we won't be able to make {n} random co-polymers")
        check_npoly = False
        
    # create all possible sequences
    all_monomer_sequences = list(set(permutations(monomers)))
    
    # rm reverse sequences, which without termini will be equivalent
    monomer_sequences = []
    for i, seq in enumerate(all_monomer_sequences):
        seq_rev = seq[::-1]
        if seq_rev not in all_monomer_sequences[i+1:]:
            monomer_sequences.append(seq)
    
    if len(monomer_sequences) > n:
        # pick n sequences
        keep = np.random.choice(len(monomer_sequences), size=n, replace=False)
        monomer_sequences = np.array(monomer_sequences)[keep]
    else:
        print(f'WARNING: {n} random sequences requested, but only {len(monomer_sequences)} found')
        n = len(monomer_sequences)
        monomer_sequences = np.array(monomer_sequences)
    
    polys = []
    for sequence in monomer_sequences:
        poly = sequence[0]
        last_smi = Chem.MolToSmiles(poly, canonical=True)
        
        # rm one reactive group to avoid messing up the sequence
        if last_smi == smi1:
            poly = rm_one_BOO(poly)
        elif last_smi == smi2:
            poly = rm_one_Br(poly)
        
        for next_m in sequence[1:]:
            next_smi = Chem.MolToSmiles(next_m, canonical=True)
            
            if last_smi == next_smi and next_smi == smi1:
                ps = BOO_BOO_bond(poly, next_m)
            elif last_smi == next_smi and next_smi == smi2:
                ps = Br_Br_bond(poly, next_m)
            else:
                ps = BOO_Br_bond(poly, next_m, bothways=True)
        
            # NB we can have >1 ps becuase m2 can be asymmetric
            poly = ps[0]
            last_smi = next_smi
        polys.append(poly)
            
    # rm termini
    polys = [rm_termini(p) for p in polys]
    # rm duplicates
    polys = rm_duplicate_mols(polys)

    if check_npoly is True:
        if len(polys) != n:
            print(f"WARNING: could only generate {len(polys)} random sequences")
    return polys


def generate_conformers(m, nconfs, ncpu):
    # Add a timeout of 20 seconds
    # in some cases, ETKDG hangs, so we revert to ETDG + FF cleanup
    with timeout(20):
        cids = AllChem.EmbedMultipleConfs(m, numConfs=nconfs, numThreads=ncpu, useRandomCoords=False)
        # fail-safe option given we're looking at large molecules that do not always
        # generate enough conformers if useRandomCoords=False
        if len(cids) < nconfs:
            cids = AllChem.EmbedMultipleConfs(m, numConfs=nconfs, numThreads=ncpu, useRandomCoords=True)
        if len(cids) < nconfs:
            raise ValueError('cannot generate enough conformers with ETKDG')
        return m, cids
    
    print('Timeout reached --> using ETDG rather than ETKDG')
    param = Chem.rdDistGeom.ETDG()
    param.numThreads = ncpu
    param.useRandomCoords = False
    cids = Chem.rdDistGeom.EmbedMultipleConfs(m, nconfs, param)
    if len(cids) < nconfs:
        param.useRandomCoords = True
        cids = Chem.rdDistGeom.EmbedMultipleConfs(m, nconfs, param)
    if len(cids) < nconfs:
        raise ValueError('cannot generate enough conformers with ETDG')
    
    return m, cids


def build_copolymers_given_mA_mB(mA, mB, nA, nB, poly_types, maxn, nconfs, ncpu, folder):
    """
    mA: mol obj A
    mB: mol obj B
    mA: num monomers A
    nB: num monomers B
    """
    
    make_dir(f"{folder}")
    
    for poly_type in poly_types:
        
        start_time = datetime.now()
        print(f'    {poly_type}...', end=' ')
        
        # create folders
        make_dir(f"{folder}/{poly_type}")
        
        if poly_type == 'block':
            polys = create_all_block_copolymers(mA, mB, nA, nB, maxn=maxn)  # up to 16 when nA=4 and nB=4
        elif poly_type == 'alternating':
            polys = create_all_alternating_copolymers(mA, mB, nA, nB)  # up to 16 when nA=4 and nB=4
        elif poly_type == 'random':
            polys = create_n_random_copolymers(mA, mB, nA, nB, n=maxn)  # maxn at max
        else:
            raise ValueError()
    
        # save 3D coords for all sequences
        print(f'generating {nconfs} conformers for {len(polys)} co-polymers...', end='')
        for n, poly in enumerate(polys):
        
            # get charge of molecule and make sure it's 0 as expected
            chg = Chem.GetFormalCharge(poly)
            assert chg == 0
            
            m = Chem.AddHs(poly)
            # generate conformers
            m, cids = generate_conformers(m=m, nconfs=nconfs, ncpu=ncpu)
            # FF minimize
            res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=ncpu)
            # save confs to file
            for cid in cids:
                Chem.MolToMolFile(m, f"{folder}/{poly_type}/poly_{n:02d}_{cid:02d}.mol", confId=cid)
        
        end_time = datetime.now()
        time_taken = end_time - start_time
        print(f'[{time_taken.total_seconds()} s]')
        


def main(args):
    
    # load monomers
    monoA = pd.read_csv(args.acids)
    monoB = pd.read_csv(args.bromides)
    molsA = [Chem.MolFromSmiles(m) for m in monoA.loc[:, '0']] 
    molsB = [Chem.MolFromSmiles(m) for m in monoB.loc[:, '0']]
    
    iAs = args.iAs
    iBs = args.iBs
    
    if max(iAs) > 8:
        raise ValueError(f'index iA {max(iAs)} is largest than max allowed (8)')
    if max(iBs) > 681:
        raise ValueError(f'index iB {max(iBs)} is largest than max allowed (681)')
    
    if -1 in iBs:
        iBs = list(range(len(molsB)))

    for iA in iAs:
        for iB in iBs:
            start_time = datetime.now()
            print(f"generating co-polymer {iA} - {iB}")
            # get monomers
            mA = molsA[iA]
            mB = molsB[iB]
            # build co-polymers
            build_copolymers_given_mA_mB(mA=mA, mB=mB, nA=args.nA, nB=args.nB, poly_types=args.poly_types, 
                                         maxn=args.maxn, nconfs=args.nconfs, ncpu=args.ncpu, folder=f"{iA}_{iB}")
            
            end_time = datetime.now()
            time_taken = end_time - start_time
            print(f'    Exec time: {time_taken.total_seconds()} s')

if __name__ == '__main__':
    args = parse_inputs()
    main(args)