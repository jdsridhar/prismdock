"""
prep_check.py — Docking-readiness QC for prepared receptor and ligand PDBQT files.

The point of this module: BEFORE spending compute on docking, confirm that the
target and every ligand were actually prepared correctly. A silently-broken
PDBQT (missing charges, no hydrogens, empty torsion tree, zero atoms) will
dock and produce a number — a meaningless one. This catches those cases and
reports a traffic-light verdict (pass / warn / fail) with human-readable
reasons for each item.

No external dependencies beyond the standard library — pure text parsing of
the AutoDock PDBQT format, so it is fast and cannot itself fail to import.

Public API
----------
check_receptor(pdbqt_path)          -> dict record
check_ligand(pdbqt_path)            -> dict record
check_run(run_dir)                  -> dict {"target": rec, "ligands": [recs], "summary": {...}}

Each record is:
    {"name", "status" in {"pass","warn","fail"}, "reasons": [str], "metrics": {...}}
"""
import os
import glob


# AutoDock atom types that carry polar hydrogens / heteroatom flavours — their
# presence is a proxy for "this was run through a real preparation tool".
_AD_POLAR_TYPES = {"HD", "HS", "NA", "NS", "OA", "OS", "SA"}
_HYDROGEN_TYPES = {"HD", "HS", "H"}


def _read(path):
    with open(path, "r", errors="replace") as fh:
        return fh.readlines()


def _parse_atom_line(line):
    """Return (atom_type, partial_charge) from a PDBQT ATOM/HETATM line, or (None, None)."""
    # PDBQT: last whitespace token is the AutoDock atom type; the token before
    # it is the partial charge. Columns are not perfectly fixed across writers,
    # so we parse from the right by whitespace.
    parts = line.split()
    if len(parts) < 2:
        return None, None
    atom_type = parts[-1]
    charge = None
    try:
        charge = float(parts[-2])
    except ValueError:
        charge = None
    return atom_type, charge


def _atom_records(lines):
    return [ln for ln in lines if ln.startswith("ATOM") or ln.startswith("HETATM")]


# ---------------------------------------------------------------------------
# Receptor
# ---------------------------------------------------------------------------
def check_receptor(pdbqt_path):
    name = os.path.basename(pdbqt_path)
    rec = {"name": name, "path": pdbqt_path, "status": "pass",
           "reasons": [], "metrics": {}}

    if not pdbqt_path or not os.path.exists(pdbqt_path):
        rec["status"] = "fail"
        rec["reasons"].append("Receptor PDBQT not found.")
        return rec

    lines = _read(pdbqt_path)
    atoms = _atom_records(lines)
    n_atoms = len(atoms)
    rec["metrics"]["n_atoms"] = n_atoms

    if n_atoms == 0:
        rec["status"] = "fail"
        rec["reasons"].append("No ATOM/HETATM records — receptor is empty.")
        return rec

    # Charge coverage
    charges = []
    types = []
    n_charge_parse_fail = 0
    for ln in atoms:
        atype, q = _parse_atom_line(ln)
        types.append(atype)
        if q is None:
            n_charge_parse_fail += 1
        else:
            charges.append(q)

    n_hydrogens = sum(1 for t in types if t in _HYDROGEN_TYPES)
    n_polar = sum(1 for t in types if t in _AD_POLAR_TYPES)
    total_abs_charge = sum(abs(q) for q in charges)
    rec["metrics"]["n_hydrogens"] = n_hydrogens
    rec["metrics"]["n_polar_atom_types"] = n_polar
    rec["metrics"]["sum_abs_charge"] = round(total_abs_charge, 3)
    rec["metrics"]["net_charge"] = round(sum(charges), 3)

    # Rule: charges assigned?
    if n_charge_parse_fail > 0.1 * n_atoms:
        rec["status"] = "fail"
        rec["reasons"].append(
            f"Partial charges missing/unparseable on {n_charge_parse_fail}/{n_atoms} atoms.")
    elif total_abs_charge < 1e-6:
        rec["status"] = "fail"
        rec["reasons"].append("All partial charges are zero — charge assignment did not run.")

    # Rule: hydrogens present? (a receptor with zero H was never protonated)
    if n_hydrogens == 0:
        rec["status"] = "fail"
        rec["reasons"].append("No hydrogen atoms — receptor was not protonated.")
    elif n_polar == 0:
        # No polar AutoDock atom types is suspicious for a protein
        if rec["status"] != "fail":
            rec["status"] = "warn"
        rec["reasons"].append("No polar AutoDock atom types (NA/OA/HD…) — check preparation.")

    # Rule: plausible size for a protein receptor
    if 0 < n_atoms < 200:
        if rec["status"] == "pass":
            rec["status"] = "warn"
        rec["reasons"].append(f"Only {n_atoms} atoms — unusually small for a protein receptor.")

    if not rec["reasons"]:
        rec["reasons"].append(
            f"{n_atoms} atoms, {n_hydrogens} H, charges assigned (net {rec['metrics']['net_charge']}).")
    return rec


# ---------------------------------------------------------------------------
# Ligand
# ---------------------------------------------------------------------------
def check_ligand(pdbqt_path, name=None):
    name = name or os.path.basename(pdbqt_path)
    rec = {"name": name, "path": pdbqt_path, "status": "pass",
           "reasons": [], "metrics": {}}

    if not pdbqt_path or not os.path.exists(pdbqt_path):
        rec["status"] = "fail"
        rec["reasons"].append("Ligand PDBQT not found.")
        return rec

    lines = _read(pdbqt_path)
    atoms = _atom_records(lines)
    n_atoms = len(atoms)
    rec["metrics"]["n_atoms"] = n_atoms

    if n_atoms == 0:
        rec["status"] = "fail"
        rec["reasons"].append("No ATOM records — ligand is empty.")
        return rec

    # Torsion tree
    has_root = any(ln.startswith("ROOT") for ln in lines)
    has_endroot = any(ln.startswith("ENDROOT") for ln in lines)
    n_branch = sum(1 for ln in lines if ln.startswith("BRANCH"))
    n_endbranch = sum(1 for ln in lines if ln.startswith("ENDBRANCH"))
    torsdof = None
    for ln in lines:
        if ln.startswith("TORSDOF"):
            try:
                torsdof = int(ln.split()[1])
            except (IndexError, ValueError):
                pass
    rec["metrics"]["n_branches"] = n_branch
    rec["metrics"]["torsdof"] = torsdof

    # Charges & coordinates
    charges = []
    coords_ok = True
    for ln in atoms:
        _, q = _parse_atom_line(ln)
        if q is not None:
            charges.append(q)
        try:
            float(ln[30:38]); float(ln[38:46]); float(ln[46:54])
        except (ValueError, IndexError):
            coords_ok = False
    total_abs_charge = sum(abs(q) for q in charges)
    rec["metrics"]["sum_abs_charge"] = round(total_abs_charge, 3)
    rec["metrics"]["net_charge"] = round(sum(charges), 3) if charges else None

    # Rule: torsion tree present (a ligand PDBQT MUST have ROOT/ENDROOT)
    if not (has_root and has_endroot):
        rec["status"] = "fail"
        rec["reasons"].append("Missing ROOT/ENDROOT — torsion tree not defined; Vina will reject it.")
    elif n_branch != n_endbranch:
        rec["status"] = "fail"
        rec["reasons"].append(
            f"Unbalanced torsion tree ({n_branch} BRANCH vs {n_endbranch} ENDBRANCH).")

    # Rule: valid 3D coordinates
    if not coords_ok:
        rec["status"] = "fail"
        rec["reasons"].append("Unparseable / missing 3D coordinates.")

    # Rule: charges assigned
    if not charges or total_abs_charge < 1e-6:
        if rec["status"] != "fail":
            rec["status"] = "warn"
        rec["reasons"].append("Partial charges look unset (all zero) — Gasteiger step may have failed.")

    # Rule: rigid ligand (no rotatable bonds) — legal but worth flagging
    if torsdof == 0 or n_branch == 0:
        if rec["status"] == "pass":
            rec["reasons"].append("Rigid ligand (0 rotatable bonds) — valid, just noted.")

    if not rec["reasons"]:
        rec["reasons"].append(
            f"{n_atoms} atoms, {n_branch} rotatable bonds, charges assigned.")
    return rec


# ---------------------------------------------------------------------------
# Whole-run driver
# ---------------------------------------------------------------------------
def check_run(run_dir):
    """QC a full run directory: the prepared receptor + every prepared ligand PDBQT."""
    receptor = os.path.join(run_dir, "target", "receptor.pdbqt")
    target_rec = check_receptor(receptor)

    lig_dir = os.path.join(run_dir, "library", "pdbqt")
    ligand_recs = []
    if os.path.isdir(lig_dir):
        for p in sorted(glob.glob(os.path.join(lig_dir, "*.pdbqt"))):
            ligand_recs.append(check_ligand(p, name=os.path.splitext(os.path.basename(p))[0]))

    n = len(ligand_recs)
    n_pass = sum(1 for r in ligand_recs if r["status"] == "pass")
    n_warn = sum(1 for r in ligand_recs if r["status"] == "warn")
    n_fail = sum(1 for r in ligand_recs if r["status"] == "fail")

    # Overall verdict: target must pass/warn; ligands need a usable majority
    if target_rec["status"] == "fail":
        overall = "fail"
        verdict = "NOT READY — receptor failed preparation QC."
    elif n == 0:
        overall = "warn"
        verdict = "Receptor OK, but no prepared ligands found."
    elif n_fail == n:
        overall = "fail"
        verdict = "NOT READY — every ligand failed preparation QC."
    elif n_fail > 0 or target_rec["status"] == "warn":
        overall = "warn"
        verdict = (f"MOSTLY READY — receptor {target_rec['status']}, "
                   f"{n_pass}/{n} ligands pass, {n_fail} failed (will be skipped).")
    else:
        overall = "pass"
        verdict = f"READY — receptor OK and all {n} ligands dock-ready."

    return {
        "target": target_rec,
        "ligands": ligand_recs,
        "summary": {
            "overall": overall,
            "verdict": verdict,
            "n_ligands": n, "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
        },
    }


if __name__ == "__main__":
    import sys, json
    print(json.dumps(check_run(sys.argv[1]), indent=2))
