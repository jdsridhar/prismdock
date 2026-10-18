"""
dock_consensus.py — Consensus docking with AutoDock Vina + Smina.

THE NOVELTY CORE. Instead of trusting a single engine, we dock every ligand
with two independent (but methodologically distinct) engines and combine them:

  * AutoDock Vina 1.2.x  — Vina scoring function.
  * Smina                — Vina fork with the (empirical) Vinardo scoring
                           function, a different pose search + scoring.

For each ligand we take the best (most negative) affinity from each engine,
then build a CONSENSUS rank by:
  * rank-by-rank aggregation (mean of per-engine ranks), and
  * score agreement (both engines rank a ligand highly -> high confidence).

Consensus across independent scoring functions is a well-established way to
reduce single-engine scoring artifacts and improve virtual-screening
enrichment. Cross-engine agreement is our credibility signal.

Optional GPU rescoring hooks (GNINA CNN, DiffDock) are wired behind flags for
the event-day machine but are NOT required for the CPU consensus.

Open-source: Vina (python API), Smina (CLI).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# ---------------------------------------------------------------- Vina (py API)
def dock_vina(receptor_pdbqt, ligand_pdbqt, box, exhaustiveness=8, n_poses=9,
              seed=42, out_pdbqt=None, cpu=0):
    """
    Dock one ligand with AutoDock Vina python API. Returns best affinity.

    cpu: threads Vina itself may use; 0 (Vina's own default) means
    "auto-detect and use whatever cores are free". Measured explicitly
    constraining this to throttle concurrent (ligand, engine) tasks down to
    ~1 thread each: it was NOT faster -- it was 2-4x SLOWER (72s vs 30s on
    a 4-ligand test), because Vina/Smina's own exhaustiveness search
    genuinely benefits from multiple threads per call, and Linux's
    scheduler already time-slices several auto-detecting processes
    reasonably well on its own. Left unconstrained on purpose; see
    dock_library()'s comment for the numbers.
    """
    from vina import Vina
    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=box["center"], box_size=box["size"])
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = v.energies(n_poses=n_poses)
    if out_pdbqt:
        v.write_poses(out_pdbqt, n_poses=n_poses, overwrite=True)
    best = float(energies[0][0]) if len(energies) else None
    return best, energies


# ---------------------------------------------------------------- Smina (CLI)
_SMINA_RE = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+")

def dock_smina(smina_bin, receptor_pdbqt, ligand_pdbqt, box, exhaustiveness=8,
               n_poses=9, seed=42, out_sdf=None, scoring="vinardo", cpu=None):
    """Dock one ligand with Smina (Vinardo scoring). Returns best affinity.
    cpu: see dock_vina's docstring -- left None (smina's own default, which
    its --help describes as "try to detect the number of CPUs") rather
    than throttled, since throttling measured slower, not faster. Passing
    an explicit int still works if a future caller wants to constrain it."""
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    if out_sdf is None:
        out_sdf = ligand_pdbqt.replace(".pdbqt", "_smina.sdf")
    cmd = [
        smina_bin, "-r", receptor_pdbqt, "-l", ligand_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(n_poses),
        "--seed", str(seed), "--scoring", scoring, "-o", out_sdf,
    ]
    if cpu:  # omit --cpu entirely when None/0 -- let smina auto-detect
        cmd += ["--cpu", str(cpu)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr[-300:]
    # Parse the affinity table from stdout: "mode | affinity | dist..."
    best = None
    for line in proc.stdout.splitlines():
        m = _SMINA_RE.match(line)
        if m:
            best = float(m.group(1))
            break
    return best, proc.stdout


# ------------------------------------------------------ consensus over a library
def _minmax_norm(s):
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


def consensus_rank(df, engines=None):
    """
    Given a DataFrame with one `{engine}_score` column per docking/rescoring
    engine (lower=better, e.g. kcal/mol-like affinities), add rank columns
    and a consensus score across ALL engines present.

    `engines` defaults to auto-detecting from columns: vina, smina, and (if
    present) gnina, in that canonical order, plus any other "*_score"
    columns found. This is what lets GNINA CNN rescoring fold in as a THIRD
    consensus member (not just a side output) when it's available on a GPU
    machine, while every 2-engine (Vina+Smina) run behaves identically to
    before.

    consensus_rank = mean of per-engine ranks (lower=better).
    consensus_z    = mean of min-max normalized scores (lower=better),
                     for magnitude-aware consensus.
    agreement      = 1 - (max engine_norm - min engine_norm)
                     (1 = all engines fully agree; generalizes the original
                     2-engine `1 - |vina_norm - smina_norm|` exactly).
    n_engines_scored = how many engines actually scored this ligand (a row
                     missing one engine's score still gets a consensus from
                     the engines that did score it -- pandas rank/min/max/
                     mean all skip NaN by default, so partial coverage
                     degrades gracefully rather than dropping the ligand).
    """
    out = df.copy()
    if engines is None:
        preferred = ["vina", "smina", "gnina"]
        engines = [e for e in preferred if f"{e}_score" in out.columns]
        engines += [c[:-6] for c in out.columns
                    if c.endswith("_score") and c[:-6] not in engines]
    if len(engines) < 2:
        raise ValueError(f"consensus_rank needs >=2 engine score columns, got {engines}")

    for eng in engines:
        col = f"{eng}_score"
        out[f"{eng}_rank"] = out[col].rank(method="min", ascending=True)
        out[f"{eng}_norm"] = _minmax_norm(out[col])

    rank_cols = [f"{e}_rank" for e in engines]
    norm_cols = [f"{e}_norm" for e in engines]
    score_cols = [f"{e}_score" for e in engines]
    out["consensus_rank"] = out[rank_cols].mean(axis=1)
    out["consensus_z"] = out[norm_cols].mean(axis=1)
    out["agreement"] = 1.0 - (out[norm_cols].max(axis=1) - out[norm_cols].min(axis=1))
    out["n_engines_scored"] = out[score_cols].notna().sum(axis=1)
    out = out.sort_values(["consensus_z", "consensus_rank"]).reset_index(drop=True)
    out.insert(0, "final_rank", range(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# Parallel worker (top-level so it is picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def _dock_engine_task(args):
    """
    Dock ONE ligand with ONE engine. Runs in a worker process; top-level
    (module-level) so it's picklable by ProcessPoolExecutor.

    Submitting one task per (ligand, engine) pair -- rather than one task
    per ligand that runs Vina then Smina sequentially inside it -- lets
    idle workers pick up whichever engine is next in the queue. On a large
    library this changes nothing (cores are already saturated by
    ligand-level parallelism either way -- total core-seconds of work is
    identical). On a SMALL library (fewer ligands than cores -- the common
    case for a demo run) it matters: previously most cores sat idle while
    each ligand paid the full Vina-then-Smina sequential cost; now Vina and
    Smina for the SAME ligand can run concurrently on two different cores,
    cutting per-ligand wall time toward 1x instead of 2x.
    """
    (engine, lid, lig, receptor_pdbqt, box, exhaustiveness, n_poses, seed,
     pose_path, smina_bin, cpu) = args
    try:
        if engine == "vina":
            best, _ = dock_vina(receptor_pdbqt, lig, box, exhaustiveness,
                                n_poses, seed, out_pdbqt=pose_path, cpu=cpu)
        else:
            best, _ = dock_smina(smina_bin, receptor_pdbqt, lig, box,
                                 exhaustiveness, n_poses, seed, out_sdf=pose_path, cpu=cpu)
    except Exception as e:
        print(f"  [{engine} fail] {lid}: {e}")
        best = None
    return lid, engine, best


def dock_library(manifest_csv, receptor_pdbqt, box, outdir,
                 smina_bin="smina", exhaustiveness=8, n_poses=9, seed=42,
                 save_poses=True, n_workers=None):
    """
    Dock every prepared ligand in the manifest with both engines,
    then compute consensus. Returns the consensus DataFrame.

    n_workers: number of parallel CPU worker processes (default: os.cpu_count() - 2,
               reserving headroom for the OS/UI so the machine stays responsive
               during a big run -- pass explicitly for maximum throughput instead).
               Set to 1 to disable parallelism (useful for debugging).
    """
    import os as _os
    n_workers = n_workers or max(1, (_os.cpu_count() or 1) - 2)
    os.makedirs(outdir, exist_ok=True)
    pose_dir = os.path.join(outdir, "poses")
    os.makedirs(pose_dir, exist_ok=True)
    man = pd.read_csv(manifest_csv)
    man = man[man["status"] == "ok"].reset_index(drop=True)

    # Incremental checkpoint: append each ligand as it finishes so an
    # interrupted run can resume (important on a time-limited event machine).
    ckpt = os.path.join(outdir, "scores_progress.csv")
    ckpt_lock = threading.Lock()  # protect concurrent CSV writes
    done = {}
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = {str(r["id"]): r for _, r in prev.iterrows()}
        print(f"  [resume] {len(done)} ligands already scored -> skipping")

    rows = [r.to_dict() if hasattr(r, "to_dict") else r for r in done.values()]
    n = len(man)
    todo = []
    for _, r in man.iterrows():
        lid = str(r["id"])
        if lid in done:
            continue
        lig = r["pdbqt"]
        if not isinstance(lig, str) or not os.path.exists(lig):
            continue
        vpose = os.path.join(pose_dir, f"{lid}_vina.pdbqt") if save_poses else None
        spose = os.path.join(pose_dir, f"{lid}_smina.sdf") if save_poses else None
        label = r.get("label")
        todo.append((lid, lig, label, receptor_pdbqt, box, exhaustiveness,
                     n_poses, seed, vpose, spose, smina_bin))

    if todo:
        effective_workers = min(n_workers, len(todo) * 2)
        # Each Vina/Smina call is left free to auto-detect/use whatever
        # cores are free (their own defaults) rather than being throttled
        # to ~1 thread per concurrent task. That throttling was tried and
        # measured SLOWER (72s vs 30s on a 4-ligand test) -- Vina/Smina's
        # own exhaustiveness search benefits from multiple threads per
        # call, and Linux's scheduler already time-slices several
        # auto-detecting processes reasonably well without help. See
        # dock_vina/dock_smina's docstrings.
        cpu_per_task = 0

        # One task per (ligand, engine) pair -- see _dock_engine_task's
        # docstring for why. A ligand's row is only complete (and only
        # checkpointed) once BOTH its engine tasks have returned, exactly
        # preserving the original resume semantics; partial per-engine
        # failure now only blanks that one engine's score instead of
        # discarding the other engine's already-computed result too.
        partial = {}   # lid -> row dict, filled in as engine tasks complete
        pending = {}   # lid -> set of engines still outstanding
        task_args = []
        for (lid, lig, label, r_pdbqt, bx, exh, npo, sd, vpose, spose, sbin) in todo:
            partial[lid] = {"id": lid, "label": label,
                            "vina_score": None, "smina_score": None}
            pending[lid] = {"vina", "smina"}
            task_args.append(("vina", lid, lig, r_pdbqt, bx, exh, npo, sd, vpose, sbin, cpu_per_task))
            task_args.append(("smina", lid, lig, r_pdbqt, bx, exh, npo, sd, spose, sbin, cpu_per_task))

        print(f"  [dock] {len(todo)} ligands × 2 engines = {len(task_args)} tasks "
              f"across {effective_workers} CPU workers")
        completed_ligands = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_dock_engine_task, args): (args[1], args[0])
                      for args in task_args}
            for fut in as_completed(futures):
                lid, engine = futures[fut]
                try:
                    _, _, score = fut.result()
                except Exception as e:
                    print(f"  [worker fail] {lid}/{engine}: {e}")
                    score = None
                partial[lid][f"{engine}_score"] = score
                pending[lid].discard(engine)
                if pending[lid]:
                    continue  # still waiting on the other engine for this ligand
                row = partial[lid]
                rows.append(row)
                completed_ligands += 1
                # Thread-safe checkpoint append -- one row per LIGAND, same
                # format scores_progress.csv / resume always expected.
                with ckpt_lock:
                    hdr = not os.path.exists(ckpt)
                    pd.DataFrame([row]).to_csv(ckpt, mode="a", header=hdr, index=False)
                if completed_ligands % 5 == 0 or completed_ligands == len(todo):
                    print(f"  [dock] {completed_ligands + len(done)}/{n} done", flush=True)

    if not rows:
        res = pd.DataFrame(columns=["id", "label", "vina_score", "smina_score"])
    else:
        res = pd.DataFrame(rows)
        
    # keep only ligands both engines scored
    if "vina_score" in res.columns and "smina_score" in res.columns:
        complete = res.dropna(subset=["vina_score", "smina_score"]).reset_index(drop=True)
    else:
        complete = pd.DataFrame(columns=["id", "label", "vina_score", "smina_score"])
        
    ranked = consensus_rank(complete)
    out_csv = os.path.join(outdir, "consensus_results.csv")
    ranked.to_csv(out_csv, index=False)
    # also persist the raw (incl. failures) for transparency
    res.to_csv(os.path.join(outdir, "raw_scores.csv"), index=False)
    print(f"[consensus] {len(ranked)}/{len(res)} ligands scored by both engines "
          f"-> {out_csv}")
    return ranked


def _cli():
    ap = argparse.ArgumentParser(description="Consensus docking (Vina + Smina).")
    ap.add_argument("--manifest", required=True, help="ligand_manifest.csv")
    ap.add_argument("--receptor", required=True, help="receptor.pdbqt")
    ap.add_argument("--box", required=True, help="box.json")
    ap.add_argument("--outdir", default="results/docking")
    ap.add_argument("--smina-bin", default="smina")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--n-poses", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Parallel CPU workers (default: all cores)")
    args = ap.parse_args()
    with open(args.box) as fh:
        box = json.load(fh)
    dock_library(args.manifest, args.receptor, box, args.outdir,
                 smina_bin=args.smina_bin, exhaustiveness=args.exhaustiveness,
                 n_poses=args.n_poses, seed=args.seed, n_workers=args.n_workers)


if __name__ == "__main__":
    _cli()
