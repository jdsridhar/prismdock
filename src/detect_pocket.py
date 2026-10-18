"""
detect_pocket.py — Automatic binding-site / docking-box detection.

Strategies, chosen automatically in this priority order:
  A. LIGAND-GUIDED (preferred): if a co-crystal ligand exists, the docking box
     is centered on the ligand centroid with padding. This is the most
     reliable box definition and is what we use whenever a bound ligand is
     present.
  B. P2RANK (apo primary): a machine-learning pocket predictor (random forest
     over solvent-accessible-surface points, trained on known binding sites).
     Published benchmarks (Krivak & Hoksza, J Cheminform 2018; subsequent
     P2Rank papers) show it beats purely-geometric detectors like fpocket on
     top-1/top-3 pocket identification accuracy (DCA/DCC), especially on apo
     structures where there's no bound ligand to hint at the site. Used when
     a `prank` binary is available (see find_p2rank()).
  C. FPOCKET (apo fallback): pure-geometry alpha-sphere clustering. Used
     automatically if P2Rank (or its Java runtime) isn't installed, so the
     pipeline never hard-fails on a missing optional dependency.

Both apo methods return not just the winning box but ALL ranked candidate
pockets (`all_pockets`) so a human (or the UI) can sanity-check the auto
choice or override it — useful when the true site is cryptic or the
top-ranked cavity is a crystallographic artifact.

Output: a dict {center:(x,y,z), size:(sx,sy,sz), method, all_pockets:[...]}
plus a JSON file for the driver.

Open-source: numpy, P2Rank (Apache-2.0), fpocket (MIT/GPL).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess

import numpy as np


def _read_atom_xyz(pdb_path: str) -> np.ndarray:
    """Return Nx3 array of atom coordinates from ATOM/HETATM records."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6].strip() in ("ATOM", "HETATM"):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
    return np.array(coords, dtype=float)


def box_from_ligand(ligand_pdb: str, padding: float = 5.0, min_size: float = 18.0):
    """
    Define a docking box from a bound ligand: center = centroid,
    size = ligand extent + 2*padding (clamped to a minimum cube).
    """
    xyz = _read_atom_xyz(ligand_pdb)
    if xyz.size == 0:
        raise ValueError(f"no atoms parsed from {ligand_pdb}")
    center = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    size = np.maximum(extent + 2 * padding, min_size)
    return {
        "center": [round(float(c), 3) for c in center],
        "size": [round(float(s), 3) for s in size],
        "method": "ligand-guided",
        "n_ligand_atoms": int(xyz.shape[0]),
    }


# --------------------------------------------------------------- P2Rank (ML)
def find_p2rank(extra_dirs=None):
    """
    Locate the P2Rank `prank` launcher script. Checked in order: PATH,
    `P2RANK_HOME` env var, `extra_dirs`, and this repo's own `vendor/p2rank/`
    (the location setup_env.sh installs it to). Returns an absolute path to
    the `prank` script, or None if not found.
    """
    on_path = shutil.which("prank")
    if on_path:
        return on_path
    candidates = list(extra_dirs or [])
    home = os.environ.get("P2RANK_HOME")
    if home:
        candidates.append(home)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "vendor", "p2rank"))
    launcher_names = ["prank.bat", "prank.cmd", "prank"] if os.name == "nt" else ["prank"]
    for d in candidates:
        for name in launcher_names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return os.path.abspath(p)
    return None


def _parse_p2rank_predictions(csv_path: str):
    """Parse a P2Rank `*_predictions.csv` into a list of pocket dicts, best first."""
    pockets = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        for row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                pockets.append({
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "probability": float(row["probability"]),
                    "center": [round(float(row["center_x"]), 3),
                               round(float(row["center_y"]), 3),
                               round(float(row["center_z"]), 3)],
                    "n_sas_points": int(float(row.get("sas_points", 0) or 0)),
                    "residue_ids": row.get("residue_ids", ""),
                })
            except (KeyError, ValueError):
                continue
    pockets.sort(key=lambda p: p["rank"])
    return pockets


def box_from_p2rank(receptor_pdb: str, outdir: str, size: float = 22.0, prank_bin: str | None = None):
    """
    Run P2Rank on the receptor and build a box around the top-ranked
    (highest-probability) predicted pocket. Returns all ranked pockets too.
    """
    prank = prank_bin or find_p2rank()
    if not prank:
        raise FileNotFoundError("p2rank 'prank' launcher not found (checked PATH, "
                                 "$P2RANK_HOME, vendor/p2rank/)")
    p2rank_out = os.path.join(outdir, "p2rank")
    if os.path.isdir(p2rank_out):
        shutil.rmtree(p2rank_out)  # idempotent: never reuse a stale prior run
    os.makedirs(p2rank_out, exist_ok=True)

    use_shell = os.name == "nt" and prank.lower().endswith((".bat", ".cmd"))
    subprocess.run([prank, "predict", "-f", receptor_pdb, "-o", p2rank_out],
                   check=True, capture_output=True, text=True, shell=use_shell)

    cands = glob.glob(os.path.join(p2rank_out, "**", "*_predictions.csv"), recursive=True)
    if not cands:
        raise RuntimeError("p2rank produced no *_predictions.csv output")
    pockets = _parse_p2rank_predictions(cands[0])
    if not pockets:
        raise RuntimeError(f"p2rank predictions file had no parseable pockets: {cands[0]}")

    top = pockets[0]
    all_pockets = [{
        "rank": p["rank"], "center": p["center"], "size": [float(size)] * 3,
        "probability": p["probability"], "score": p["score"],
        "residue_ids": p["residue_ids"],
    } for p in pockets]
    return {
        "center": top["center"],
        "size": [float(size)] * 3,
        "method": "p2rank",
        "probability": top["probability"],
        "n_pockets_found": len(pockets),
        "all_pockets": all_pockets,
        "predictions_csv": cands[0],
    }


# --------------------------------------------------------- fpocket (geometry)
def box_from_fpocket(receptor_pdb: str, outdir: str, size: float = 22.0):
    """
    Run fpocket on the receptor and build a box around the top-ranked
    (highest-druggability-score) pocket. Returns all ranked pockets too.
    """
    base = os.path.splitext(receptor_pdb)[0]
    out_dir = base + "_out"
    if os.path.isdir(out_dir):
        # fpocket exits 0 but silently fails to (re)write output if this
        # directory already exists (mkdir error swallowed internally),
        # leaving STALE pocket files from a prior run in place. Must clear
        # it first so a re-run of the pipeline can't silently reuse old data.
        shutil.rmtree(out_dir)

    print(f"[fpocket] analysing {receptor_pdb} ...")
    subprocess.run(["fpocket", "-f", receptor_pdb], check=True,
                   capture_output=True, text=True)

    pocket_files = sorted(
        glob.glob(os.path.join(out_dir, "pockets", "pocket*_atm.pdb")),
        key=lambda p: int("".join(ch for ch in os.path.basename(p).split("_")[0] if ch.isdigit()) or 0),
    )
    if not pocket_files:
        raise RuntimeError("fpocket produced no pocket files")

    all_pockets = []
    for pf in pocket_files:
        xyz = _read_atom_xyz(pf)
        if xyz.size == 0:
            continue
        rank = int("".join(ch for ch in os.path.basename(pf).split("_")[0] if ch.isdigit()))
        all_pockets.append({
            "rank": rank,
            "center": [round(float(c), 3) for c in xyz.mean(axis=0)],
            "size": [float(size)] * 3,
            "pocket_file": pf,
            "n_pocket_atoms": int(xyz.shape[0]),
        })
    all_pockets.sort(key=lambda p: p["rank"])
    top = all_pockets[0]
    return {
        "center": top["center"],
        "size": [float(size)] * 3,
        "method": "fpocket",
        "pocket_file": top["pocket_file"],
        "n_pocket_atoms": top["n_pocket_atoms"],
        "n_pockets_found": len(all_pockets),
        "all_pockets": all_pockets,
    }


def detect_pocket(receptor_pdb: str, ligand_pdb: str | None, outdir: str,
                  padding: float = 5.0, apo_size: float = 22.0):
    """
    Auto-select box strategy. Writes box.json to outdir. Returns the box dict.

    Priority: ligand-guided > P2Rank > fpocket. P2Rank is preferred for apo
    targets because it consistently out-identifies fpocket's true top-1
    pocket on published benchmarks; fpocket is used automatically if P2Rank
    (or Java) isn't installed on this machine, so a missing optional
    dependency never blocks a run.
    """
    os.makedirs(outdir, exist_ok=True)
    if ligand_pdb and os.path.exists(ligand_pdb):
        box = box_from_ligand(ligand_pdb, padding=padding)
    else:
        try:
            box = box_from_p2rank(receptor_pdb, outdir, size=apo_size)
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
            print(f"[pocket] p2rank unavailable/failed ({e}); falling back to fpocket")
            box = box_from_fpocket(receptor_pdb, outdir, size=apo_size)
    box_path = os.path.join(outdir, "box.json")
    with open(box_path, "w") as fh:
        json.dump(box, fh, indent=2)
    n_alt = len(box.get("all_pockets", [])) - 1
    alt_note = f", {n_alt} alternate candidate(s) recorded" if n_alt > 0 else ""
    print(f"[pocket] method={box['method']} center={box['center']} "
          f"size={box['size']}{alt_note} -> {box_path}")
    return box


def _cli():
    ap = argparse.ArgumentParser(description="Detect the docking box.")
    ap.add_argument("--receptor", required=True, help="Cleaned receptor PDB")
    ap.add_argument("--ligand", default=None, help="Native ligand PDB (optional)")
    ap.add_argument("--outdir", default="results/target")
    ap.add_argument("--padding", type=float, default=5.0)
    ap.add_argument("--apo-size", type=float, default=22.0)
    args = ap.parse_args()
    detect_pocket(args.receptor, args.ligand, args.outdir,
                  padding=args.padding, apo_size=args.apo_size)


if __name__ == "__main__":
    _cli()
