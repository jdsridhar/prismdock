"""
rescore_gpu.py — OPTIONAL GPU rescoring hooks (event-day machine).

The CPU consensus (Vina+Smina) is the validated core and runs everywhere.
When a GPU is available, this module can add:

  * GNINA CNN rescoring -- a genuine THIRD CONSENSUS MEMBER. GNINA re-scores
    the Vina poses with a convolutional-neural-network pose/affinity model.
    This is not cosmetic: published benchmarks show GNINA's CNN scoring
    out-ranks Vina's empirical scoring on virtual-screening enrichment on
    the large majority of DUD-E/LIT-PCBA targets tested (see README for
    citations), on the same AUC/EF/BEDROC axis this pipeline already
    validates against. `gnina_rescore` therefore returns a `gnina_score`
    column in the SAME lower-is-better convention as vina_score/smina_score
    (GNINA's raw CNNaffinity is higher-is-better, so it is negated), and
    `run()` merges it back into consensus_results.csv and RE-RANKS via
    `dock_consensus.consensus_rank` with all engines present -- it is not a
    side file. Open-source (Apache-2.0).
    Install: `conda install -c gnina gnina` (needs a CUDA GPU + driver).

  * DiffDock-L -- NOT a ranking input. Benchmarks show DiffDock's blind pose
    accuracy trails Vina/GNINA when the binding site is already known (as it
    always is in this pipeline, via ligand-guided or fpocket box detection):
    a recent fair comparison found Vina and GNINA's top-1 success rates beat
    DiffDock's by roughly 20-25 points at the 2 Angstrom RMSD threshold (see
    README). DiffDock's real value here is a box-free SANITY CHECK: does its
    blind pose land in the same pocket the box assumes, or does it flag a
    different (possibly cryptic) site? Its poses are written for inspection,
    never merged into consensus_z. Requires a separate clone of
    github.com/gcorso/DiffDock (`DIFFDOCK_REPO` env var pointing at it) plus
    model weights and a GPU.

Both require their binaries/models + a GPU. If unavailable, run_dockathon.py
catches the ImportError/FileNotFoundError/CalledProcessError and proceeds
with CPU consensus only -- these are pure add-ons, never a hard dependency.

This file is intentionally a thin, well-documented HOOK: on the event machine
you install gnina / DiffDock, flip the flags in config.yaml (or the checkbox
in the Streamlit Run tab), and the extra output folds in automatically.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pandas as pd


def find_gnina(extra_dirs=None):
    """
    Locate a `gnina` executable. `shutil.which` only searches PATH, which
    misses the common case of a user downloading the ~1.2 GB standalone
    binary straight into the project/repo folder and never adding it to
    PATH. We additionally check (in order): `extra_dirs` (if given), the
    current working directory, this file's own repo root, and the
    `GNINA_PATH` env var (a direct path to the binary, not a directory).
    Returns an absolute path, or None if not found anywhere.
    """
    on_path = shutil.which("gnina")
    if on_path:
        return on_path

    env_path = os.environ.get("GNINA_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return os.path.abspath(env_path)

    candidates = list(extra_dirs or [])
    candidates.append(os.getcwd())
    candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in candidates:
        p = os.path.join(d, "gnina")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None


def find_diffdock_python():
    """
    Locate the Python interpreter to run DiffDock's inference.py with.
    DiffDock pins old, specific versions (Python 3.9, torch==1.13.1+cu117,
    torch-geometric==2.2.0, ...) that conflict with this project's own
    docking environment (rdkit/numpy/scipy versions Vina/Smina/ProLIF rely
    on) -- so it must live in its own conda env, never the docking env.
    Checked in order: `DIFFDOCK_PYTHON` env var (direct path to the
    interpreter), then the conda env named `diffdock` at its default
    location. Falls back to plain `python3` on PATH (works if the caller
    has already activated an env with DiffDock's deps -- e.g. CI/Docker).
    """
    env_path = os.environ.get("DIFFDOCK_PYTHON")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    default_conda_env = os.path.expanduser("~/miniconda3/envs/diffdock/bin/python3")
    if os.path.isfile(default_conda_env) and os.access(default_conda_env, os.X_OK):
        return default_conda_env
    return "python3"


def find_diffdock_repo(extra_dirs=None):
    """
    Locate the DiffDock repo clone. Checked in order: `DIFFDOCK_REPO` env
    var, `extra_dirs`, and this repo's own `vendor/DiffDock/` (the location
    documented in setup_env.sh). Mirrors detect_pocket.find_p2rank()'s
    pattern -- without this auto-fallback, a run launched from the
    Streamlit Run tab's subprocess never sees DIFFDOCK_REPO even when it's
    exported in whatever shell you happen to be reading this from, because
    Popen() inherits Streamlit's own environment, not yours. Returns an
    absolute path, or None if not found.
    """
    env_path = os.environ.get("DIFFDOCK_REPO")
    if env_path and os.path.isdir(env_path):
        return os.path.abspath(env_path)
    candidates = list(extra_dirs or [])
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "vendor", "DiffDock"))
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "inference.py")):
            return os.path.abspath(c)
    return None


def _extract_best_model(pdbqt_path, tmp_path):
    """
    GNINA's `--score_only` mode rejects multi-MODEL PDBQT input (the format
    Vina/Smina write for multi-pose output) with 'Unexpected multi-MODEL
    input'. Extract just the first (best-ranked, lowest-energy) MODEL block
    and strip the MODEL/ENDMDL wrapper lines, writing a clean single-pose
    PDBQT gnina can parse. Returns True if a model was found and written.
    """
    lines = []
    in_model = False
    seen_model = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if seen_model:
                    break  # only take the first model
                in_model = True
                seen_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if in_model:
                lines.append(line)
            elif not seen_model and not line.startswith("REMARK") and line.strip():
                # file has no MODEL/ENDMDL wrapper at all -- already single-pose
                lines.append(line)
    if not lines:
        return False
    with open(tmp_path, "w") as f:
        f.writelines(lines)
    return True


def gnina_rescore(results_csv, receptor_pdbqt, pose_dir, out_csv, gnina_path=None):
    """
    Re-score docked poses with GNINA's CNN. Requires a `gnina` binary.
    Runs fine on CPU (auto-fallback with a WARNING when no GPU is detected)
    -- just slower than with a GPU; no special flag is needed for that.

    Adds three columns and returns the updated DataFrame:
      * gnina_cnnaffinity -- GNINA's raw predicted affinity (higher = more
        favorable binding, its native convention -- kept for interpretability).
      * gnina_score -- the SAME value negated, so it sorts lower-is-better
        like vina_score/smina_score. This is the column dock_consensus.
        consensus_rank looks for so GNINA becomes a real third engine in
        the consensus, not just a reference number.
      * gnina_cnnscore -- GNINA's separate pose-quality confidence (0-1,
        higher = more pose-like), kept alongside affinity for the report.
    """
    gnina = gnina_path or find_gnina()
    if not gnina:
        raise FileNotFoundError(
            "gnina binary not found on PATH, in the working/repo directory, "
            "or at $GNINA_PATH")
    df = pd.read_csv(results_csv)
    cnnaff, cnnscore = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for lid in df["id"]:
            pose = os.path.join(pose_dir, f"{lid}_vina.pdbqt")
            if not os.path.exists(pose):
                continue
            clean_pose = os.path.join(tmp, f"{lid}_best.pdbqt")
            if not _extract_best_model(pose, clean_pose):
                continue
            cmd = [gnina, "-r", receptor_pdbqt, "-l", clean_pose, "--score_only",
                   "--cnn_scoring", "rescore"]
            out = subprocess.run(cmd, capture_output=True, text=True)
            aff, score = None, None
            for line in out.stdout.splitlines():
                if line.startswith("CNNaffinity"):
                    try:
                        aff = float(line.split()[-1])
                    except Exception:
                        pass
                elif line.startswith("CNNscore"):
                    try:
                        score = float(line.split()[-1])
                    except Exception:
                        pass
            if aff is None:
                # scoring failed for this ligand (bad pose, parse error, etc.)
                # -- skip it, don't poison the column; n_engines_scored in
                # dock_consensus.consensus_rank handles partial coverage.
                continue
            cnnaff[lid] = aff
            cnnscore[lid] = score
    df["gnina_cnnaffinity"] = df["id"].map(cnnaff)
    df["gnina_cnnscore"] = df["id"].map(cnnscore)
    df["gnina_score"] = -df["gnina_cnnaffinity"]
    df.to_csv(out_csv, index=False)
    return df


def diffdock_poses(manifest_csv, receptor_pdb, outdir, diffdock_repo=None, box=None):
    """
    Run DiffDock-L blind pose prediction for every ligand in the manifest,
    against `receptor_pdb`. Requires a clone of github.com/gcorso/DiffDock
    (path from `diffdock_repo` or the `DIFFDOCK_REPO` env var) with its
    conda env / weights already set up, plus a CUDA GPU.

    DiffDock is run BOX-FREE (it searches the whole protein surface), which
    is exactly what makes it useful as a POCKET SANITY CHECK rather than a
    ranking input: if `box` (the {center:[x,y,z], size:[x,y,z]} dict the
    rest of the pipeline docked into) is given, this also reports whether
    DiffDock's own top blind pose landed inside that box -- agreement is
    reassuring, a mismatch is worth a human look (possible cryptic/alternate
    site), neither changes consensus_z.

    Writes `{outdir}/diffdock/<complex_name>/rank1.sdf` per ligand and
    returns a DataFrame with columns
    [id, diffdock_rank1_sdf, diffdock_ok, diffdock_in_box (if box given)].
    DiffDock's confidence score is a pose-plausibility logit, NOT a binding
    affinity -- it is reported for reference, not merged into consensus_z.
    """
    repo = diffdock_repo or find_diffdock_repo()
    if not repo or not os.path.isdir(repo):
        raise FileNotFoundError(
            "DiffDock repo not found. Clone github.com/gcorso/DiffDock to "
            "vendor/DiffDock (auto-detected) or set DIFFDOCK_REPO (or pass "
            "diffdock_repo=) to point at it."
        )
    df = pd.read_csv(manifest_csv)
    dd_outdir = os.path.abspath(os.path.join(outdir, "diffdock"))
    os.makedirs(dd_outdir, exist_ok=True)
    # The inference subprocess runs with cwd=repo (DiffDock's own directory),
    # so any relative path here (receptor_pdb is typically built from a
    # relative `output.outdir` in config.yaml) would silently resolve
    # against the wrong directory and fail prody's parsePDB. Absolutize.
    receptor_pdb = os.path.abspath(receptor_pdb)

    batch_csv = os.path.join(dd_outdir, "diffdock_batch.csv")
    batch = pd.DataFrame({
        "complex_name": df["id"],
        "protein_path": receptor_pdb,
        "ligand_description": df["smiles"],
        "protein_sequence": "",
    })
    batch.to_csv(batch_csv, index=False)

    cmd = [find_diffdock_python(), "-m", "inference",
           "--config", "default_inference_args.yaml",
           "--protein_ligand_csv", batch_csv,
           "--out_dir", dd_outdir]
    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"DiffDock inference failed (exit {out.returncode}): "
                            f"{out.stderr[-2000:]}")

    rows = []
    for lid in df["id"]:
        rank1 = os.path.join(dd_outdir, str(lid), "rank1.sdf")
        ok = os.path.exists(rank1)
        row = {"id": lid, "diffdock_rank1_sdf": rank1 if ok else None, "diffdock_ok": ok}
        if box is not None and ok:
            row["diffdock_in_box"] = _pose_center_in_box(rank1, box)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(dd_outdir, "diffdock_summary.csv"), index=False)
    return result


def _pose_center_in_box(sdf_path, box, pad=2.0):
    """True if an SDF pose's centroid falls inside `box` (+pad Angstrom slack)."""
    try:
        from rdkit import Chem
        mol = next(Chem.SDMolSupplier(sdf_path, sanitize=False))
        if mol is None:
            return None
        pos = mol.GetConformer().GetPositions()
        cx, cy, cz = pos.mean(axis=0)
        bc, bs = box["center"], box["size"]
        return (abs(cx - bc[0]) <= bs[0] / 2 + pad and
                abs(cy - bc[1]) <= bs[1] / 2 + pad and
                abs(cz - bc[2]) <= bs[2] / 2 + pad)
    except Exception:
        return None


def run(results_csv, tprep, box, outdir, rescoring_cfg):
    """
    Dispatch enabled GPU rescorers. Called by run_dockathon.py.

    When GNINA is enabled, this MERGES gnina_score into the main
    consensus_results.csv and re-ranks with dock_consensus.consensus_rank
    across all three engines (Vina + Smina + GNINA) -- GNINA becomes a real
    consensus member, not a disconnected side file. A copy of the original
    2-engine ranking is kept as consensus_results_2engine.csv for comparison
    (e.g. to show judges how the ranking shifted once GNINA was added).
    """
    pose_dir = os.path.join(outdir, "poses")
    if rescoring_cfg.get("gnina_cnn"):
        gnina_csv = os.path.join(outdir, "gnina_scores.csv")
        print("  [gnina] CNN rescoring poses ...")
        try:
            gnina_df = gnina_rescore(results_csv, tprep["receptor_pdbqt"], pose_dir, gnina_csv)
            base = pd.read_csv(results_csv)
            base.to_csv(os.path.join(outdir, "consensus_results_2engine.csv"), index=False)
            keep_cols = [c for c in base.columns
                         if c not in ("vina_rank", "smina_rank", "vina_norm", "smina_norm",
                                      "consensus_rank", "consensus_z", "agreement",
                                      "final_rank", "n_engines_scored")]
            merged = base[keep_cols].merge(
                gnina_df[["id", "gnina_cnnaffinity", "gnina_score"]], on="id", how="left")
            import dock_consensus as dc
            reranked = dc.consensus_rank(merged, engines=["vina", "smina", "gnina"])
            reranked.to_csv(results_csv, index=False)
            n_scored = int(merged["gnina_score"].notna().sum())
            print(f"  [gnina] merged into consensus: {n_scored}/{len(merged)} ligands "
                  f"got a GNINA score -> re-ranked {results_csv}")
        except FileNotFoundError as e:
            print(f"  [gnina] skipped: {e}")
    if rescoring_cfg.get("diffdock"):
        manifest_csv = os.path.join(os.path.dirname(outdir), "library", "ligand_manifest.csv")
        print("  [diffdock] running blind pose prediction (this can take a while per ligand) ...")
        try:
            dd = diffdock_poses(manifest_csv, tprep["receptor_clean"], outdir, box=box)
            if "diffdock_in_box" in dd.columns:
                n_ok = int(dd["diffdock_ok"].sum())
                n_agree = int(dd["diffdock_in_box"].fillna(False).sum())
                print(f"  [diffdock] {n_ok}/{len(dd)} poses predicted; "
                      f"{n_agree}/{n_ok} land inside the docking box (pocket sanity check).")
            print(f"  [diffdock] -> {os.path.join(outdir, 'diffdock', 'diffdock_summary.csv')}")
        except FileNotFoundError as e:
            print(f"  [diffdock] skipped: {e}")
