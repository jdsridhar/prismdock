#!/usr/bin/env python
"""
run_dockathon.py — ONE-COMMAND driver for the PrismDock consensus docking pipeline.

    python run_dockathon.py --config config.yaml

Pipeline (all steps automatic):
  1. Ingest + prepare the target receptor (prepare_target).
  2. Auto-detect the docking box (detect_pocket).
  3. Prepare the ligand library (prepare_ligands).
  4. Consensus docking with Vina + Smina (dock_consensus).
  5. Optional GPU rescoring (GNINA / DiffDock) if enabled + available.
  6. Self-validation: redock RMSD + retrospective enrichment (validate).
  7. Trust layer: run-level validation gate + per-hit confidence tiers (trust).
  8. Auto-report: figures + report.md (report).

On event day you edit config.yaml (target PDB + library) and run this once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import prepare_target as pt
import detect_pocket as dp
import prepare_ligands as pl
import dock_consensus as dc
import validate as val
import trust as tr
import report as rep


def _versions():
    v = {}
    try:
        import vina; v["vina"] = vina.__version__
    except Exception: v["vina"] = "?"
    try:
        out = subprocess.run(["smina", "--version"], capture_output=True, text=True)
        v["smina"] = out.stdout.strip().split("\n")[0]
    except Exception: v["smina"] = "?"
    try:
        out = subprocess.run(["obabel", "-V"], capture_output=True, text=True)
        v["openbabel"] = out.stdout.strip().split("\n")[0]
    except Exception: v["openbabel"] = "?"
    try:
        v["p2rank"] = "installed" if dp.find_p2rank() else "not installed (fpocket fallback used)"
    except Exception: v["p2rank"] = "?"
    try:
        import rescore_gpu as _rg
        gnina_path = _rg.find_gnina()
        v["gnina"] = f"installed ({gnina_path})" if gnina_path else "not installed (optional 3rd engine)"
        dd_repo = _rg.find_diffdock_repo()
        dd_py = _rg.find_diffdock_python()
        v["diffdock"] = (f"installed (repo={dd_repo}, env={dd_py})"
                         if dd_repo and dd_py != "python3"
                         else "not installed (optional pocket sanity check)")
    except Exception: pass
    for mod in ("rdkit", "meeko", "sklearn", "numpy", "pandas"):
        try:
            m = __import__(mod); v[mod] = getattr(m, "__version__", "?")
        except Exception: v[mod] = "?"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    outdir = cfg["output"]["outdir"]
    os.makedirs(outdir, exist_ok=True)
    print("=" * 70)
    print(f"PRISMDOCK PIPELINE  ->  {outdir}")
    print("=" * 70)

    # ---- 1. target prep
    print("\n[1/7] Target preparation")
    tprep = pt.prepare_target(pdb_id=cfg["target"].get("pdb"),
                              infile=cfg["target"].get("infile"),
                              outdir=os.path.join(outdir, "target"),
                              ph=cfg["target"].get("ph", 7.4))

    # ---- 2. pocket detection
    print("\n[2/7] Binding-site detection")
    pc = cfg.get("pocket", {})
    if pc.get("force_center") and pc.get("force_size"):
        box = {"center": pc["force_center"], "size": pc["force_size"],
               "method": "user-forced"}
        json.dump(box, open(os.path.join(outdir, "target", "box.json"), "w"), indent=2)
    else:
        box = dp.detect_pocket(tprep["receptor_clean"], tprep["ligand_native"],
                               os.path.join(outdir, "target"),
                               padding=pc.get("padding", 5.0),
                               apo_size=pc.get("apo_size", 22.0))

    # ---- 3. ligand prep
    print("\n[3/7] Ligand library preparation")
    library_csv = cfg["library"]["csv"]
    
    if cfg["target"].get("dock_native") and tprep.get("ligand_native") and os.path.exists(tprep["ligand_native"]):
        try:
            import pandas as pd
            from rdkit import Chem
            mol = Chem.MolFromPDBFile(tprep["ligand_native"])
            if mol:
                smi = Chem.MolToSmiles(mol)
                resname = tprep.get("ligand_resname", "NATIVE")
                orig_df = pd.read_csv(library_csv)
                new_row = {"id": f"CO_CRYSTAL_{resname}", "smiles": smi}
                if "label" in orig_df.columns:
                    new_row["label"] = 1
                orig_df = pd.concat([orig_df, pd.DataFrame([new_row])], ignore_index=True)
                
                library_csv = os.path.join(outdir, "library", "augmented_library.csv")
                os.makedirs(os.path.dirname(library_csv), exist_ok=True)
                orig_df.to_csv(library_csv, index=False)
                print(f"  [native] Extracted SMILES {smi} from {resname}, added to library as CO_CRYSTAL_{resname}")
            else:
                print("  [native] Failed to extract SMILES from native ligand PDB")
        except Exception as e:
            print(f"  [native] Error extracting SMILES: {e}")

    manifest = pl.prepare_library(library_csv,
                                  os.path.join(outdir, "library"),
                                  limit=cfg["library"].get("limit"))
    manifest_csv = os.path.join(outdir, "library", "ligand_manifest.csv")
    prep_stats = {
        "n_total": int(len(manifest)),
        "n_ok": int((manifest["status"] == "ok").sum()),
        "n_meeko": int((manifest["prep_tool"] == "meeko").sum()) if "prep_tool" in manifest else 0,
        "n_obabel": int((manifest["prep_tool"] == "obabel").sum()) if "prep_tool" in manifest else 0,
        "n_failed": int((manifest["status"] != "ok").sum()),
    }

    # ---- 4. consensus docking
    print("\n[4/7] Consensus docking (Vina + Smina)")
    ranked = dc.dock_library(manifest_csv, tprep["receptor_pdbqt"], box,
                             os.path.join(outdir, "docking"),
                             smina_bin=cfg["docking"].get("smina_bin", "smina"),
                             exhaustiveness=cfg["docking"].get("exhaustiveness", 8),
                             n_poses=cfg["docking"].get("n_poses", 9),
                             seed=cfg["docking"].get("seed", 42),
                             n_workers=cfg["docking"].get("n_workers", None))
    results_csv = os.path.join(outdir, "docking", "consensus_results.csv")

    # ---- 5. optional GPU rescoring
    print("\n[5/8] Optional GPU rescoring")
    rs = cfg.get("rescoring", {})
    if rs.get("gnina_cnn") or rs.get("diffdock"):
        print("  (GPU rescoring flags set — see src/rescore_gpu.py for the hook)")
        try:
            import rescore_gpu
            rescore_gpu.run(results_csv, tprep, box, os.path.join(outdir, "docking"), rs)
            # gnina_rescore rewrites results_csv on disk with a merged, re-ranked
            # 3-engine table (see rescore_gpu.run docstring) -- reload it so every
            # step below (validation, trust, report) sees gnina_score too, instead
            # of the stale 2-engine `ranked` still held from dock_library() above.
            import pandas as _pd
            ranked = _pd.read_csv(results_csv)
        except Exception as e:
            print(f"  [skip] GPU rescoring unavailable: {e}")
    else:
        print("  (disabled — CPU consensus only)")

    # ---- 6. validation
    print("\n[6/8] Self-validation")
    redock = None
    if cfg["validation"].get("redock_native", True) and tprep["ligand_native"]:
        redock = val.redock_rmsd(tprep["ligand_native"], tprep["receptor_pdbqt"],
                                 box, os.path.join(outdir, "validation"))
    metrics = {}
    has_labels = "label" in ranked.columns and ranked["label"].notna().any()
    if cfg["validation"].get("enrichment", True) and has_labels:
        metrics, _ = val.compute_metrics(results_csv, os.path.join(outdir, "validation"))
    else:
        print("  (no labels in library -> skipping enrichment; prospective mode)")

    # ---- 7. trust layer: validation gate + per-hit confidence
    print("\n[7/8] Trust & confidence layer")
    run_trust = tr.run_trust_verdict(redock, metrics)
    print(f"  [trust] {run_trust['emoji']} {run_trust['status']}")
    for r in run_trust["reasons"]:
        print(f"    - {r}")

    n_engines_total = tr.n_engines_from_columns(ranked)
    contacts_by_id = {}
    sort_col = "final_rank" if "final_rank" in ranked.columns else ranked.columns[-1]
    top_k_ids = ranked.sort_values(sort_col).head(15)["id"].astype(str).tolist()
    pose_dir = os.path.join(outdir, "docking", "poses")
    try:
        import interactions as inter
        for lid in top_k_ids:
            sdf = os.path.join(pose_dir, f"{lid}_smina.sdf")
            if os.path.exists(sdf) and tprep.get("receptor_clean") and os.path.exists(tprep["receptor_clean"]):
                try:
                    res = inter.analyze_pose(tprep["receptor_clean"], sdf, pose_index=0)
                    if res["ok"]:
                        contacts_by_id[lid] = res["table"]
                except Exception:
                    pass  # one bad pose shouldn't drop the contact-specificity bonus for the rest
        print(f"  [trust] contact-specificity computed for {len(contacts_by_id)}/{len(top_k_ids)} top hits")
    except Exception as e:
        print(f"  [trust] contact-specificity bonus unavailable ({e}); "
              "confidence tiers still computed from agreement + validation status")

    ranked = tr.annotate_hits(ranked, run_trust, n_engines_total, contacts_by_id=contacts_by_id)
    ranked.to_csv(results_csv, index=False)
    trust_path = os.path.join(outdir, "validation", "trust.json")
    os.makedirs(os.path.dirname(trust_path), exist_ok=True)
    with open(trust_path, "w") as fh:
        json.dump(run_trust, fh, indent=2)
    print(f"  [trust] confidence tiers -> {results_csv}")
    print(f"  [trust] verdict -> {trust_path}")

    # ---- 8. report
    print("\n[8/8] Report")
    if cfg["output"].get("report", True):
        figdir = os.path.join(outdir, "report")
        figs = {}
        if has_labels:
            figs = rep.make_figures(results_csv, figdir)
        target_id = cfg["target"].get("pdb") or os.path.basename(
            cfg["target"].get("infile", "target"))
        rep.build_report(target_id, metrics, redock, figs, ranked,
                         _versions(), figdir, box=box, prep_stats=prep_stats,
                         run_trust=run_trust)

    print("\n" + "=" * 70)
    print("DONE. Key outputs:")
    print(f"  ranked hits : {results_csv}")
    print(f"  report      : {os.path.join(outdir, 'report', 'report.md')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
