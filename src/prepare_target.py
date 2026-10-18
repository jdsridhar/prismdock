"""
prepare_target.py — Target ingestion + receptor preparation.

Given a PDB ID (or a local structure file / AlphaFold model), this module:
  1. Fetches the structure from RCSB (or reads a local file).
  2. Separates the protein receptor from HETATM records.
  3. Auto-extracts the co-crystal ligand (largest non-water/non-ion HETATM group).
  4. Cleans + protonates the receptor with PDBFixer (adds missing heavy atoms,
     adds hydrogens at a target pH).
  5. Converts the receptor to PDBQT for AutoDock Vina / Smina.

Everything here is 100% open-source: requests, PDBFixer/OpenMM, Meeko, OpenBabel.

CLI:
    python prepare_target.py --pdb 1H00 --outdir results/1H00
    python prepare_target.py --infile my_model.pdb --outdir results/mymodel
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict

# Residue names we never treat as "the ligand"
_WATER = {"HOH", "WAT", "DOD", "H2O"}
_IONS = {
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO", "CD",
    "HG", "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "DMS", "FMT", "MPD",
    "TRS", "BME", "IOD", "BR", "NO3", "NH4", "CO3",
}
_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def fetch_pdb(pdb_id: str, outdir: str) -> str:
    """Download a PDB file from RCSB. Returns the local path."""
    pdb_id = pdb_id.strip().lower()
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"{pdb_id}.pdb")
    if os.path.exists(dest):
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[fetch] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def split_receptor_ligand(pdb_path: str, outdir: str):
    """
    Parse a PDB into protein (ATOM + std-residue) and candidate ligands
    (HETATM groups that are not water/ions/standard residues).

    Returns (receptor_raw_pdb, ligand_records) where ligand_records maps
    (resname, chain, resseq) -> list of raw PDB lines.
    """
    os.makedirs(outdir, exist_ok=True)
    protein_lines = []
    het_groups: dict = defaultdict(list)

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                if rec == "ATOM" or resname in _STD_AA:
                    # MSE (selenomethionine) etc. kept as protein
                    protein_lines.append(line)
                elif resname in _WATER or resname in _IONS:
                    continue  # drop
                else:
                    het_groups[(resname, chain, resseq)].append(line)
            elif rec in ("TER",):
                protein_lines.append(line)

    receptor_raw = os.path.join(outdir, "receptor_raw.pdb")
    with open(receptor_raw, "w") as fh:
        fh.writelines(protein_lines)
        fh.write("END\n")

    return receptor_raw, het_groups


def pick_main_ligand(het_groups: dict, outdir: str):
    """
    Choose the largest HETATM group (by atom count) as the co-crystal ligand.
    Writes it to ligand_native.pdb. Returns (path, resname) or (None, None).
    """
    if not het_groups:
        print("[ligand] no co-crystal ligand found (apo structure)")
        return None, None
    # largest by atom count
    key = max(het_groups, key=lambda k: len(het_groups[k]))
    resname, chain, resseq = key
    lines = het_groups[key]
    path = os.path.join(outdir, "ligand_native.pdb")
    with open(path, "w") as fh:
        fh.writelines(lines)
        fh.write("END\n")
    print(f"[ligand] native ligand = {resname} (chain {chain}, res {resseq}), "
          f"{len(lines)} atoms -> {path}")
    return path, resname


def clean_receptor(receptor_raw: str, outdir: str, ph: float = 7.4) -> str:
    """
    Clean + protonate the receptor with PDBFixer.
    Adds missing residues/atoms and hydrogens at the given pH.
    Returns path to receptor_clean.pdb.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    print(f"[clean] running PDBFixer (pH {ph}) ...")
    fixer = PDBFixer(filename=receptor_raw)
    fixer.findMissingResidues()
    # Do not model missing terminal loops (avoid inventing large regions);
    # keep only internal missing residues.
    keys = list(fixer.missingResidues.keys())
    chains = list(fixer.topology.chains())
    for key in keys:
        chain = chains[key[0]]
        n_res = len(list(chain.residues()))
        if key[1] == 0 or key[1] == n_res:
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    clean = os.path.join(outdir, "receptor_clean.pdb")
    with open(clean, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    print(f"[clean] wrote {clean}")
    return clean


def receptor_to_pdbqt(receptor_clean: str, outdir: str) -> str:
    """
    Convert the cleaned receptor to PDBQT.
    Try Meeko's mk_prepare_receptor first; fall back to OpenBabel.
    """
    pdbqt = os.path.join(outdir, "receptor.pdbqt")

    # --- Attempt 1: Meeko ---
    mk = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if mk:
        try:
            cmd = [mk, "--read_pdb", receptor_clean, "-o",
                   os.path.join(outdir, "receptor"), "-p"]
            print(f"[pdbqt] meeko: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            cand = os.path.join(outdir, "receptor.pdbqt")
            if os.path.exists(cand):
                print(f"[pdbqt] wrote {cand} (meeko)")
                return cand
        except subprocess.CalledProcessError as e:
            print(f"[pdbqt] meeko failed ({e.returncode}); falling back to OpenBabel")
            print(e.stderr[-500:] if e.stderr else "")

    # --- Attempt 2: OpenBabel ---
    cmd = ["obabel", receptor_clean, "-xr", "-O", pdbqt, "--partialcharge", "gasteiger"]
    print(f"[pdbqt] openbabel: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    # OpenBabel writes ROOT/branch tags for -xr rigid receptor
    print(f"[pdbqt] wrote {pdbqt} (openbabel)")
    return pdbqt


def prepare_target(pdb_id=None, infile=None, outdir="results/target", ph=7.4):
    """
    Full target prep. Returns a dict of output paths.
    """
    os.makedirs(outdir, exist_ok=True)
    if infile:
        src = os.path.join(outdir, os.path.basename(infile))
        if os.path.abspath(infile) != os.path.abspath(src):
            shutil.copy(infile, src)
        pdb_path = src
    elif pdb_id:
        pdb_path = fetch_pdb(pdb_id, outdir)
    else:
        raise ValueError("provide either pdb_id or infile")

    receptor_raw, het_groups = split_receptor_ligand(pdb_path, outdir)
    ligand_native, lig_resname = pick_main_ligand(het_groups, outdir)
    receptor_clean = clean_receptor(receptor_raw, outdir, ph=ph)
    receptor_pdbqt = receptor_to_pdbqt(receptor_clean, outdir)

    result = {
        "source_pdb": pdb_path,
        "receptor_raw": receptor_raw,
        "receptor_clean": receptor_clean,
        "receptor_pdbqt": receptor_pdbqt,
        "ligand_native": ligand_native,
        "ligand_resname": lig_resname,
        "n_het_groups": len(het_groups),
    }
    print("[done] target prepared:")
    for k, v in result.items():
        print(f"    {k}: {v}")
    return result


def _cli():
    ap = argparse.ArgumentParser(description="Prepare a docking target (receptor + native ligand).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", help="RCSB PDB ID to fetch, e.g. 1H00")
    g.add_argument("--infile", help="Local PDB/mmCIF structure file")
    ap.add_argument("--outdir", default="results/target")
    ap.add_argument("--ph", type=float, default=7.4)
    args = ap.parse_args()
    prepare_target(pdb_id=args.pdb, infile=args.infile, outdir=args.outdir, ph=args.ph)


if __name__ == "__main__":
    _cli()
