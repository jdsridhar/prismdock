"""
enumerate_scaffold.py — Nucleus (core scaffold) -> analog library.

Dockathon "nucleus" workflow: the committee hands you a fixed core scaffold
on event day and every candidate must keep that core. This module decorates
the nucleus at its open positions with a curated set of medchem R-groups and
writes a docking-ready library CSV (columns: id, smiles, core_kept) that
feeds straight into prepare_ligands / run_dockathon.

Two enumeration modes, chosen automatically from the nucleus SMILES:
  * Explicit attachment points — if the nucleus contains dummy atoms [*]
    (e.g. "O=C(N[*])c1ccc([*])cc1"), each [*] is combinatorially replaced by
    each chosen R-group. This is the precise, chemist-controlled mode.
  * Open C-H scan — if there are no dummy atoms, every aromatic C-H (or all
    C-H, if aromatic_only=False) is a candidate site and each R-group is
    substituted there, one at a time (mono-substitution), with an optional
    di-substitution pass.

Every analog is guaranteed to contain the original nucleus as a substructure
(verified with a substructure match) so the "keep the core" rule is enforced.

Open-source: RDKit only.
"""
from __future__ import annotations

import argparse
import itertools
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, Crippen


# ---------------------------------------------------------------------------
# Curated R-group library: single dummy atom [*] marks the attachment bond.
# Ordered roughly small -> large / lipophilic -> polar so early truncation
# (max_analogs) still yields a chemically diverse set.
# ---------------------------------------------------------------------------
RGROUP_LIBRARY = {
    "H (none)":        "[*][H]",
    "Fluoro":          "[*]F",
    "Chloro":          "[*]Cl",
    "Bromo":           "[*]Br",
    "Methyl":          "[*]C",
    "Ethyl":           "[*]CC",
    "Isopropyl":       "[*]C(C)C",
    "tert-Butyl":      "[*]C(C)(C)C",
    "Cyclopropyl":     "[*]C1CC1",
    "Trifluoromethyl": "[*]C(F)(F)F",
    "Hydroxy":         "[*]O",
    "Methoxy":         "[*]OC",
    "Amino":           "[*]N",
    "Methylamino":     "[*]NC",
    "Dimethylamino":   "[*]N(C)C",
    "Nitrile":         "[*]C#N",
    "Carboxyl":        "[*]C(=O)O",
    "Carboxamide":     "[*]C(=O)N",
    "Acetyl":          "[*]C(C)=O",
    "Methylsulfonyl":  "[*]S(C)(=O)=O",
    "Sulfonamide":     "[*]S(N)(=O)=O",
    "Nitro":           "[*][N+](=O)[O-]",
    "Phenyl":          "[*]c1ccccc1",
    "Benzyl":          "[*]Cc1ccccc1",
    "4-Pyridyl":       "[*]c1ccncc1",
    "3-Pyridyl":       "[*]c1cccnc1",
    "Morpholino":      "[*]N1CCOCC1",
    "Piperazinyl":     "[*]N1CCNCC1",
    "N-Me-piperazinyl":"[*]N1CCN(C)CC1",
    "Piperidinyl":     "[*]N1CCCCC1",
    "Pyrrolidinyl":    "[*]N1CCCC1",
    "Imidazolyl":      "[*]c1cnc[nH]1",
}

# A sensible default subset for "just enumerate something reasonable".
DEFAULT_RGROUPS = [
    "Fluoro", "Chloro", "Methyl", "Trifluoromethyl", "Methoxy", "Hydroxy",
    "Amino", "Dimethylamino", "Nitrile", "Carboxamide", "Methylsulfonyl",
    "Phenyl", "4-Pyridyl", "Morpholino", "N-Me-piperazinyl",
]


def rgroup_smiles(names):
    """Map friendly R-group names -> SMILES fragments (unknown names ignored)."""
    return [RGROUP_LIBRARY[n] for n in names if n in RGROUP_LIBRARY]


def _short(frag_smiles):
    inv = {v: k for k, v in RGROUP_LIBRARY.items()}
    nm = inv.get(frag_smiles)
    if nm:
        return nm.split()[0].replace("(none)", "H")
    return frag_smiles.replace("[*]", "R")


def _sanitize_or_none(m):
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def substitutable_positions(core, aromatic_only=True):
    """Indices of carbon atoms bearing >=1 H (candidate decoration sites)."""
    idxs = []
    for a in core.GetAtoms():
        if a.GetAtomicNum() != 6:
            continue
        if a.GetTotalNumHs() < 1:
            continue
        if aromatic_only and not a.GetIsAromatic():
            continue
        idxs.append(a.GetIdx())
    return idxs


def _attach_fragment(core, atom_idx, frag_smiles):
    """Replace one H on core atom_idx with fragment (dummy = attach point)."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    dummy = next((a for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None or dummy.GetDegree() != 1:
        # "H (none)" -> [*][H]: returns the bare core unchanged
        if frag_smiles == "[*][H]":
            return Chem.Mol(core)
        return None
    frag_attach = dummy.GetNeighbors()[0].GetIdx()

    combo = Chem.CombineMols(core, frag)
    rw = Chem.RWMol(combo)
    n_core = core.GetNumAtoms()
    dummy_idx = n_core + dummy.GetIdx()
    frag_attach_idx = n_core + frag_attach

    ca = rw.GetAtomWithIdx(atom_idx)
    orig_hs = ca.GetTotalNumHs()
    if orig_hs < 1:
        return None
    rw.AddBond(atom_idx, frag_attach_idx, Chem.BondType.SINGLE)
    ca.SetNumExplicitHs(max(0, orig_hs - 1))
    ca.SetNoImplicit(False)
    rw.RemoveAtom(dummy_idx)
    try:
        rw.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return _sanitize_or_none(rw.GetMol())


def _replace_dummy(mol, dummy_idx, frag_smiles):
    """Replace one dummy atom in mol with an R-group fragment."""
    if frag_smiles == "[*][H]":
        # cap the dummy's neighbor with an implicit H: just delete the dummy
        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(dummy_idx)
        nbr = a.GetNeighbors()[0]
        nbr.SetNoImplicit(False)
        rw.RemoveAtom(dummy_idx)
        return _sanitize_or_none(rw.GetMol())
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    fdummy = next((x for x in frag.GetAtoms() if x.GetAtomicNum() == 0), None)
    if fdummy is None:
        return None
    fattach = fdummy.GetNeighbors()[0].GetIdx()
    nbr = mol.GetAtomWithIdx(dummy_idx).GetNeighbors()[0].GetIdx()
    combo = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combo)
    base_n = mol.GetNumAtoms()
    rw.AddBond(nbr, base_n + fattach, Chem.BondType.SINGLE)
    for ri in sorted([dummy_idx, base_n + fdummy.GetIdx()], reverse=True):
        rw.RemoveAtom(ri)
    return _sanitize_or_none(rw.GetMol())


def _core_query(core_smiles):
    """Substructure query for the nucleus (dummies stripped -> match core skeleton)."""
    q = Chem.MolFromSmiles(core_smiles)
    if q is None:
        return None
    if any(a.GetAtomicNum() == 0 for a in q.GetAtoms()):
        # drop dummies for the substructure test
        rw = Chem.RWMol(q)
        for idx in sorted([a.GetIdx() for a in q.GetAtoms() if a.GetAtomicNum() == 0],
                          reverse=True):
            rw.RemoveAtom(idx)
        q = _sanitize_or_none(rw.GetMol()) or q
    return q


def enumerate_analogs(core_smiles, rgroup_names=None, max_analogs=200,
                      aromatic_only=True, di_substitution=False,
                      include_core=True):
    """Decorate the nucleus with chosen R-groups. Returns a pandas DataFrame
    [id, smiles, r_group, sites, core_kept]. Every row is guaranteed to contain
    the nucleus skeleton as a substructure."""
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        raise ValueError(f"Could not parse nucleus SMILES: {core_smiles!r}")

    names = rgroup_names or DEFAULT_RGROUPS
    frags = rgroup_smiles(names)
    if not frags:
        raise ValueError("No valid R-groups selected.")
    core_q = _core_query(core_smiles)

    has_dummy = any(a.GetAtomicNum() == 0 for a in core.GetAtoms())
    rows, seen = [], set()

    def _add(m, rg, sites):
        if m is None:
            return
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            return
        if smi in seen:
            return
        mm = Chem.MolFromSmiles(smi)
        if mm is None:
            return
        if core_q is not None and not mm.HasSubstructMatch(core_q):
            return  # enforce "keep the core"
        seen.add(smi)
        rows.append({"smiles": smi, "r_group": rg, "sites": sites})

    if include_core:
        _add(Chem.MolFromSmiles(core_smiles.replace("[*]", "[H]")) or core,
             "core", "-")

    if has_dummy:
        dummy_positions = [a.GetIdx() for a in core.GetAtoms() if a.GetAtomicNum() == 0]
        # combinatorial: each dummy independently gets one of the R-groups
        for combo in itertools.product(names, repeat=len(dummy_positions)):
            if len(rows) >= max_analogs:
                break
            m = Chem.MolFromSmiles(core_smiles)
            ok = True
            # replace dummies left-to-right; re-find dummy indices each step
            for frag_name in combo:
                d = next((a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0), None)
                if d is None:
                    ok = False
                    break
                m = _replace_dummy(m, d, RGROUP_LIBRARY[frag_name])
                if m is None:
                    ok = False
                    break
            if ok:
                _add(m, "+".join(_short(RGROUP_LIBRARY[c]) for c in combo),
                     f"{len(dummy_positions)} pts")
    else:
        positions = substitutable_positions(core, aromatic_only=aromatic_only)
        for frag_name in names:
            fs = RGROUP_LIBRARY[frag_name]
            if fs == "[*][H]":
                continue
            for pos in positions:
                if len(rows) >= max_analogs:
                    break
                m = _attach_fragment(Chem.MolFromSmiles(core_smiles), pos, fs)
                _add(m, _short(fs), f"@{pos}")
            if len(rows) >= max_analogs:
                break
        if di_substitution and len(rows) < max_analogs:
            for frag_name in names:
                fs = RGROUP_LIBRARY[frag_name]
                if fs == "[*][H]":
                    continue
                for p1, p2 in itertools.combinations(positions, 2):
                    if len(rows) >= max_analogs:
                        break
                    m1 = _attach_fragment(Chem.MolFromSmiles(core_smiles), p1, fs)
                    if m1 is None:
                        continue
                    m2 = _attach_fragment(m1, p2, fs)
                    _add(m2, _short(fs) + "x2", f"@{p1},{p2}")
                if len(rows) >= max_analogs:
                    break

    df = pd.DataFrame(rows[:max_analogs])
    if df.empty:
        raise ValueError("Enumeration produced no valid analogs — check the nucleus SMILES.")
    df.insert(0, "id", [f"analog_{i:03d}" if r["r_group"] != "core" else "CORE"
                        for i, r in df.iterrows()])
    df["core_kept"] = True
    return df


def add_properties(df):
    """Append MW / logP / HBD / HBA / TPSA / RotB drug-likeness columns."""
    props = {"MW": [], "logP": [], "HBD": [], "HBA": [], "TPSA": [], "RotB": [],
             "Lipinski_pass": []}
    for smi in df["smiles"]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            for k in props:
                props[k].append(None)
            continue
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        hbd = rdMolDescriptors.CalcNumHBD(m)
        hba = rdMolDescriptors.CalcNumHBA(m)
        viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        props["MW"].append(round(mw, 1))
        props["logP"].append(round(logp, 2))
        props["HBD"].append(hbd)
        props["HBA"].append(hba)
        props["TPSA"].append(round(rdMolDescriptors.CalcTPSA(m), 1))
        props["RotB"].append(rdMolDescriptors.CalcNumRotatableBonds(m))
        props["Lipinski_pass"].append(viol <= 1)
    for k, v in props.items():
        df[k] = v
    return df


def write_library(df, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def grid_image(df, out_png, mols_per_row=5, max_mols=20, legend_col="id"):
    """Save a 2D grid depiction of the enumerated analogs."""
    sub = df.head(max_mols)
    mols, legends = [], []
    for _, r in sub.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        mols.append(m)
        legends.append(str(r.get(legend_col, "")))
    if not mols:
        return None
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,
                               subImgSize=(240, 200), legends=legends)
    img.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Enumerate analogs around a nucleus scaffold.")
    ap.add_argument("--nucleus", required=True, help="Core scaffold SMILES (may contain [*]).")
    ap.add_argument("--out", required=True, help="Output library CSV.")
    ap.add_argument("--rgroups", nargs="*", default=None,
                    help="R-group names (default: a curated 15-member set).")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--all-ch", action="store_true",
                    help="Scan all C-H, not just aromatic C-H.")
    ap.add_argument("--di", action="store_true", help="Add di-substitution pass.")
    ap.add_argument("--grid", default=None, help="Optional 2D grid PNG path.")
    args = ap.parse_args()

    df = enumerate_analogs(args.nucleus, rgroup_names=args.rgroups,
                           max_analogs=args.max, aromatic_only=not args.all_ch,
                           di_substitution=args.di)
    df = add_properties(df)
    write_library(df, args.out)
    print(f"[enumerate] {len(df)} analogs -> {args.out}")
    if args.grid:
        grid_image(df, args.grid)
        print(f"[enumerate] grid -> {args.grid}")


if __name__ == "__main__":
    main()
