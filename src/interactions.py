"""
interactions.py — 2D protein–ligand interaction analysis (open-source LigPlot+ analogue).

Uses ProLIF (RDKit + MDAnalysis based, MIT licensed) to detect and render the
non-covalent contacts between a docked ligand pose and the receptor:
H-bonds (donor/acceptor), hydrophobic, pi-stacking, cation-pi, salt bridges
(anionic/cationic), and van-der-Waals contacts.

Outputs:
  * a 2D interaction network diagram (ProLIF lignetwork) as embeddable HTML
  * a tidy contact table (residue, interaction type) as a DataFrame

This is the open-source stand-in for LigPlot+ (which is free-for-academic GUI
only and not redistributable). ProLIF's lignetwork is the closest 2D analogue:
ligand at the centre, interacting residues around it, edges coloured by
interaction type.

Public API
----------
analyze_pose(receptor_pdb, pose_sdf, pose_index=0)  -> dict
    {"ok": bool, "n_interactions": int, "table": DataFrame,
     "fingerprint": Fingerprint, "ligand": Mol, "protein": Molecule,
     "message": str}

lignetwork_html(result)                              -> str (HTML) or None
contact_table(result)                                -> DataFrame

Multi-engine comparison
------------------------
Vina and Smina search independently and can land on genuinely different
poses for the same ligand -- which means genuinely different contacts.
Historically only Smina's pose was ever analyzed here (its SDF output keeps
bond orders; Vina's PDBQT output doesn't, and ProLIF needs bond orders to
type interactions correctly). vina_pose_to_sdf() converts Vina's pose via
OpenBabel so it can be analyzed the same way, and analyze_pose_multi() /
contact_comparison_table() build a side-by-side view of what each engine's
pose actually contacts, with shared (cross-engine-agreed) contacts flagged
as the more geometrically robust ones.

analyze_pose_multi(receptor_pdb, run_dir, ligand_id, pose_index=0) -> dict
    {engine_name: analyze_pose() result, ...}
contact_comparison_table(per_engine_results)         -> DataFrame
"""
import os
import sys
import shutil
import subprocess
import warnings

warnings.filterwarnings("ignore")


def _load_protein(receptor_pdb):
    from rdkit import Chem
    import prolif as plf
    if not receptor_pdb or not os.path.exists(receptor_pdb):
        raise FileNotFoundError(f"Receptor file not found: {receptor_pdb}")
    prot_rd = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
    if prot_rd is None:
        raise ValueError(f"RDKit could not parse receptor: {receptor_pdb}")
    try:
        prot_rd.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return plf.Molecule.from_rdkit(prot_rd)


def analyze_pose(receptor_pdb, pose_sdf, pose_index=0):
    """Compute the interaction fingerprint for one docked pose against the receptor.

    Returns a result dict. `ok` is False (with an explanatory `message`) when
    the receptor/pose are missing or when no contacts are found — the latter
    usually means the pose sits outside the receptor pocket (e.g. a stale pose
    from a different coordinate frame), which is worth surfacing rather than
    silently drawing an empty diagram.
    """
    import prolif as plf

    result = {"ok": False, "n_interactions": 0, "table": None,
              "fingerprint": None, "ligand": None, "protein": None,
              "message": ""}

    if not receptor_pdb or not os.path.exists(receptor_pdb):
        result["message"] = "Receptor PDB not found for this run."
        return result
    if not pose_sdf or not os.path.exists(pose_sdf):
        result["message"] = f"Pose SDF not found: {pose_sdf}"
        return result

    poses = list(plf.sdf_supplier(pose_sdf))
    if not poses:
        result["message"] = "No poses in SDF file."
        return result
    pose_index = max(0, min(pose_index, len(poses) - 1))
    lig = poses[pose_index]

    prot = _load_protein(receptor_pdb)

    fp = plf.Fingerprint()
    fp.run_from_iterable([lig], prot)
    df = fp.to_dataframe()

    result["fingerprint"] = fp
    result["ligand"] = lig
    result["protein"] = prot
    result["n_interactions"] = int(df.shape[1])

    if df.shape[1] == 0:
        result["message"] = (
            "No interactions detected — the pose may sit outside the receptor "
            "pocket (check that pose and receptor share the same coordinate frame).")
        result["table"] = _empty_table()
        return result

    result["ok"] = True
    result["table"] = _tidy_table(fp, lig)
    result["message"] = f"{df.shape[1]} contacts across {result['table']['residue'].nunique()} residues."
    return result


def _empty_table():
    import pandas as pd
    return pd.DataFrame(columns=["residue", "interaction", "ligand_atom",
                                 "ligand_atom_idx", "protein_atom_idx"])


def _tidy_table(fp, lig):
    """
    Build the tidy contact table straight from ProLIF's raw fingerprint
    (fp.ifp), not the wide to_dataframe() view -- the raw IFP carries the
    specific ATOM index behind each contact. Docked poses come from a plain
    SDF with no PDB metadata, so every ligand is generically named "UNL1" by
    ProLIF/MDAnalysis (nothing to tell atoms apart by name); reporting which
    atom (e.g. "O3", the carbonyl oxygen) actually makes each contact is far
    more useful than repeating that placeholder on every row.

    Uses `parent_indices`, NOT `indices` -- ProLIF's `indices` are LOCAL to
    the small per-residue fragment it extracts internally for the protein
    side (and can collide across different residues), while
    `parent_indices` are the real atom index in the full ligand/protein
    molecule. Confirmed by direct measurement: looking up 3D coordinates
    via `indices` produced ~25 Å "contacts" and duplicate positions across
    different residues; `parent_indices` produced the expected 2-4 Å.
    `ligand_atom_idx`/`protein_atom_idx` are exposed (not just the display
    label) so callers can look up exact 3D coordinates -- e.g. to draw a
    real interaction line between the two atoms, not just highlight a
    residue.
    """
    import pandas as pd
    order = {"HBDonor": 0, "HBAcceptor": 1, "Cationic": 2, "Anionic": 3,
             "CationPi": 4, "PiCation": 5, "PiStacking": 6,
             "Hydrophobic": 7, "VdWContact": 8}
    rows = []
    ifp = fp.ifp[0]  # single-pose fingerprint: {(lig_resid, prot_resid): {itype: (meta, ...)}}
    for (_lig_resid, prot_resid), itypes in ifp.items():
        for itype, metadatas in itypes.items():
            for meta in metadatas:
                parent = meta.get("parent_indices", {})
                lig_idxs = parent.get("ligand", ())
                prot_idxs = parent.get("protein", ())
                atom_label = "+".join(
                    f"{lig.GetAtomWithIdx(i).GetSymbol()}{i}" for i in lig_idxs) or "?"
                rows.append({"residue": str(prot_resid), "interaction": itype,
                            "ligand_atom": atom_label,
                            "ligand_atom_idx": lig_idxs[0] if lig_idxs else None,
                            "protein_atom_idx": prot_idxs[0] if prot_idxs else None})
    tbl = pd.DataFrame(rows, columns=["residue", "interaction", "ligand_atom",
                                      "ligand_atom_idx", "protein_atom_idx"])
    if not tbl.empty:
        # order interactions with the "specific" ones first for readability
        tbl["_o"] = tbl["interaction"].map(lambda x: order.get(x, 99))
        tbl = tbl.sort_values(["_o", "residue"]).drop(columns="_o").reset_index(drop=True)
    return tbl


def lignetwork_html(result):
    """Return an embeddable HTML string for the 2D interaction network, or None."""
    if not result.get("ok"):
        return None
    fp = result["fingerprint"]
    lig = result["ligand"]
    try:
        net = fp.plot_lignetwork(lig, kind="frame", frame=0)
        # ProLIF 2.x returns a LigNetwork object whose .save(path) writes a
        # self-contained vis-network HTML page. Round-trip through a temp file.
        import tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".html", delete=False) as tf:
            tmp = tf.name
        try:
            net.save(tmp)
            with open(tmp) as fh:
                return fh.read()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as e:  # noqa: BLE001 — surface as message, never crash the tab
        result["message"] = f"lignetwork render failed: {e}"
        return None


def contact_table(result):
    if result.get("table") is not None:
        return result["table"]
    return _empty_table()


def list_pose_sdfs(run_dir):
    """Return {ligand_id: sdf_path} for smina SDF poses in a run (SDF keeps bond orders)."""
    pose_dir = os.path.join(run_dir, "docking", "poses")
    out = {}
    if os.path.isdir(pose_dir):
        for f in sorted(os.listdir(pose_dir)):
            if f.endswith("_smina.sdf"):
                out[f[:-len("_smina.sdf")]] = os.path.join(pose_dir, f)
    return out


def _extract_top_model_pdbqt(pose_pdbqt, out_path):
    """Write just the first (top-ranked) MODEL block of a multi-pose Vina
    PDBQT, stripped of the MODEL/ENDMDL wrapper, so OpenBabel converts only
    the pose actually reported as the answer -- not an arbitrary mix."""
    lines, in_model, seen_model = [], False, False
    with open(pose_pdbqt) as f:
        for line in f:
            if line.startswith("MODEL"):
                if seen_model:
                    break
                in_model = True
                seen_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if in_model:
                lines.append(line)
            elif not seen_model and not line.startswith("REMARK") and line.strip():
                lines.append(line)  # no MODEL/ENDMDL wrapper -- already single-pose
    if not lines:
        return False
    with open(out_path, "w") as f:
        f.writelines(lines)
    return True


def vina_pose_to_sdf(pose_pdbqt, cache=True):
    """
    Convert Vina's top-ranked PDBQT pose to SDF via OpenBabel so ProLIF can
    analyze it (PDBQT has no bond orders; ProLIF needs them to type
    interactions like aromaticity-dependent pi-stacking correctly).
    Cached next to the source file -- returns the cached path if present.
    Returns the SDF path, or None if extraction/conversion failed.
    """
    if not pose_pdbqt or not os.path.exists(pose_pdbqt):
        return None
    out_sdf = pose_pdbqt.replace("_vina.pdbqt", "_vina_converted.sdf")
    if cache and os.path.exists(out_sdf):
        return out_sdf
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix="_top.pdbqt", delete=False) as tf:
        top_path = tf.name
    try:
        if not _extract_top_model_pdbqt(pose_pdbqt, top_path):
            return None
        obabel_bin = shutil.which("obabel") or shutil.which("obabel.exe")
        if not obabel_bin:
            cand = os.path.join(sys.prefix, "Scripts", "obabel.exe" if os.name == "nt" else "obabel")
            if os.path.isfile(cand):
                obabel_bin = cand
            else:
                obabel_bin = "obabel"
        proc = subprocess.run([obabel_bin, top_path, "-O", out_sdf],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_sdf):
            return None
        return out_sdf
    except (FileNotFoundError, OSError):
        return None
    finally:
        if os.path.exists(top_path):
            os.remove(top_path)


def list_engine_poses(run_dir, ligand_id):
    """
    Resolve every available engine's pose for one ligand to an SDF path
    ProLIF can read. Returns {engine_name: sdf_path}. GNINA is deliberately
    excluded: it re-scores Vina's exact pose (`--score_only`) rather than
    searching for a new one, so its "interactions" would just duplicate
    Vina's -- comparing them would manufacture a difference that isn't
    there. DiffDock is included when present: its poses come from an
    independent, box-free blind search, so its contacts are genuinely a
    third opinion, not a duplicate of the box-based engines.
    """
    pose_dir = os.path.join(run_dir, "docking", "poses")
    out = {}
    smina_sdf = os.path.join(pose_dir, f"{ligand_id}_smina.sdf")
    if os.path.exists(smina_sdf):
        out["smina"] = smina_sdf
    vina_pdbqt = os.path.join(pose_dir, f"{ligand_id}_vina.pdbqt")
    if os.path.exists(vina_pdbqt):
        converted = vina_pose_to_sdf(vina_pdbqt)
        if converted:
            out["vina"] = converted
    diffdock_sdf = os.path.join(run_dir, "docking", "diffdock", str(ligand_id), "rank1.sdf")
    if os.path.exists(diffdock_sdf):
        out["diffdock"] = diffdock_sdf
    return out


def analyze_pose_multi(receptor_pdb, run_dir, ligand_id, pose_index=0):
    """Run analyze_pose() for every engine's pose of one ligand.
    Returns {engine_name: analyze_pose() result}."""
    engine_poses = list_engine_poses(run_dir, ligand_id)
    return {engine: analyze_pose(receptor_pdb, sdf, pose_index=pose_index if engine != "vina" else 0)
           for engine, sdf in engine_poses.items()}


_ENGINE_ORDER = ["vina", "smina", "diffdock"]


def contact_comparison_table(per_engine_results):
    """
    Combine per-engine contact tables into one matrix: one row per unique
    (residue, interaction) pair, one column per engine holding the specific
    atom that engine's pose uses for that contact (blank if that engine's
    pose doesn't make it). `n_engines` counts how many engines agree on
    that exact contact -- rows are sorted so fully-agreed (most
    geometrically robust) contacts come first.
    """
    import pandas as pd
    engines = [e for e in _ENGINE_ORDER if e in per_engine_results] + \
             [e for e in per_engine_results if e not in _ENGINE_ORDER]
    order = {"HBDonor": 0, "HBAcceptor": 1, "Cationic": 2, "Anionic": 3,
             "CationPi": 4, "PiCation": 5, "PiStacking": 6,
             "Hydrophobic": 7, "VdWContact": 8}

    cells = {}  # (residue, interaction) -> {engine: atom_label}
    for engine in engines:
        res = per_engine_results.get(engine)
        tbl = res["table"] if res and res.get("ok") else None
        if tbl is None or tbl.empty:
            continue
        for _, row in tbl.iterrows():
            key = (row["residue"], row["interaction"])
            cells.setdefault(key, {})[engine] = row["ligand_atom"]

    rows = []
    for (residue, interaction), by_engine in cells.items():
        row = {"residue": residue, "interaction": interaction}
        for engine in engines:
            row[engine] = by_engine.get(engine, "")
        row["n_engines"] = len(by_engine)
        rows.append(row)

    cols = ["residue", "interaction"] + engines + ["n_engines"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["_o"] = out["interaction"].map(lambda x: order.get(x, 99))
        out = out.sort_values(["n_engines", "_o", "residue"], ascending=[False, True, True])
        out = out.drop(columns="_o").reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys, json
    r = analyze_pose(sys.argv[1], sys.argv[2])
    print(json.dumps({"ok": r["ok"], "n": r["n_interactions"], "msg": r["message"]}, indent=2))
    if r["table"] is not None:
        print(r["table"].to_string())
