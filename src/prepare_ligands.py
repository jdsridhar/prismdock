"""
prepare_ligands.py — Ligand library preparation.

Takes SMILES (with IDs + optional active/decoy labels) and produces
docking-ready 3D PDBQT files:
  1. Standardize + protonate at pH 7.4 (RDKit).
  2. Embed a 3D conformer (ETKDGv3) + MMFF optimize.
  3. Convert to PDBQT via Meeko (mk_prepare_ligand), falling back to
     OpenBabel if Meeko fails on a particular molecule.

Meeko is preferred: it's built and maintained by the Forli Lab (the same
group behind AutoDock/Vina), so it assigns AutoDock atom types and torsion
trees the way Vina/GNINA actually expect, and it correctly handles
macrocycles/flexible rings that older tools mishandle. OpenBabel's PDBQT
writer is a more permissive, general-purpose cheminformatics converter --
less precise on AutoDock-specific atom typing, but a good safety net: it
succeeds on some inputs Meeko rejects, matching the same
Meeko-primary/OpenBabel-fallback pattern already used for receptor prep in
prepare_target.py, so a single molecule's edge case can no longer silently
drop it from the whole library.

Input: a CSV with columns [id, smiles, label] (label optional: 1=active,0=decoy)
Output: per-ligand PDBQT in <outdir>/pdbqt/, a combined SDF, and a manifest CSV
(with a `prep_tool` column recording which tool succeeded for each ligand).

Open-source: RDKit, Meeko, OpenBabel.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize(mol):
    """Cleanup, largest fragment, uncharge."""
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    return mol


def embed_3d(mol, seed: int = 0xF00D):
    """Add Hs, embed one conformer with ETKDGv3, MMFF-optimize."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # retry with random coords
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return mol


def mol_to_pdbqt(mol, mol_id: str, outdir: str) -> tuple[str | None, str | None]:
    """
    Write an SDF then convert to PDBQT: try Meeko's mk_prepare_ligand.py
    first, fall back to OpenBabel if Meeko fails on this molecule.
    Returns (pdbqt_path, tool_used) — tool_used is "meeko"/"obabel"/None.
    """
    pdbqt_dir = os.path.join(outdir, "pdbqt")
    sdf_dir = os.path.join(outdir, "sdf")
    os.makedirs(pdbqt_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)

    sdf_path = os.path.join(sdf_dir, f"{mol_id}.sdf")
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()

    pdbqt_path = os.path.join(pdbqt_dir, f"{mol_id}.pdbqt")

    # --- Attempt 1: Meeko ---
    cmd = ["mk_prepare_ligand.py", "-i", sdf_path, "-o", pdbqt_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(pdbqt_path):
            return pdbqt_path, "meeko"
    except subprocess.CalledProcessError as e:
        print(f"  [warn] meeko failed for {mol_id} ({e.stderr[-200:] if e.stderr else e}); "
              f"falling back to OpenBabel")

    # --- Attempt 2: OpenBabel ---
    cmd = ["obabel", sdf_path, "-O", pdbqt_path, "--partialcharge", "gasteiger"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  [warn] obabel also failed for {mol_id}: {e.stderr[-200:] if e.stderr else e}")
        return None, None
    return (pdbqt_path, "obabel") if os.path.exists(pdbqt_path) else (None, None)


def prepare_library(csv_path: str, outdir: str, id_col="id", smiles_col="smiles",
                    label_col="label", limit: int | None = None):
    """
    Prepare a whole library. Returns a manifest DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    rows = []
    combined_sdf = os.path.join(outdir, "library.sdf")
    writer = Chem.SDWriter(combined_sdf)
    n_ok = n_fail = 0
    for _, r in df.iterrows():
        mol_id = str(r[id_col])
        smi = str(r[smiles_col])
        label = r[label_col] if label_col in df.columns else None
        mol = Chem.MolFromSmiles(smi)
        status = "ok"
        pdbqt = None
        prep_tool = None
        if mol is None:
            status = "parse_fail"
        else:
            mol = standardize(mol)
            mol3d = embed_3d(mol)
            if mol3d is None:
                status = "embed_fail"
            else:
                mol3d.SetProp("_Name", mol_id)
                writer.write(mol3d)
                pdbqt, prep_tool = mol_to_pdbqt(mol3d, mol_id, outdir)
                if pdbqt is None:
                    status = "pdbqt_fail"
        if status == "ok":
            n_ok += 1
        else:
            n_fail += 1
        rows.append({"id": mol_id, "smiles": smi, "label": label,
                     "pdbqt": pdbqt, "status": status, "prep_tool": prep_tool})
    writer.close()
    manifest = pd.DataFrame(rows)
    man_path = os.path.join(outdir, "ligand_manifest.csv")
    manifest.to_csv(man_path, index=False)
    n_obabel_fallback = int((manifest["prep_tool"] == "obabel").sum())
    print(f"[ligands] prepared {n_ok} ok, {n_fail} failed -> {man_path}")
    if n_obabel_fallback:
        print(f"[ligands] {n_obabel_fallback}/{n_ok} used the OpenBabel fallback (Meeko failed on them)")
    print(f"[ligands] combined SDF -> {combined_sdf}")
    return manifest


def _cli():
    ap = argparse.ArgumentParser(description="Prepare a ligand library for docking.")
    ap.add_argument("--csv", required=True, help="CSV with id,smiles[,label]")
    ap.add_argument("--outdir", default="results/library")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    prepare_library(args.csv, args.outdir, limit=args.limit)


if __name__ == "__main__":
    _cli()
