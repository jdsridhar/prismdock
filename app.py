"""
PrismDock — P2Rank-Guided Rank-Integrated Smina/Vina Molecular Docking & Validation Suite.
Streamlit front-end for the consensus docking pipeline.

Run with:  streamlit run app.py
"""
import os
import sys
import json
import shutil
import subprocess
import time

# Ensure venv Scripts & openbabel directories are in PATH for subprocesses (obabel, vina, etc.)
venv_scripts = os.path.join(sys.prefix, "Scripts")
if os.path.isdir(venv_scripts) and venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

try:
    import openbabel
    ob_bin = os.path.join(os.path.dirname(openbabel.__file__), "bin")
    if os.path.isdir(ob_bin) and ob_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ob_bin + os.pathsep + os.environ.get("PATH", "")
    data = os.path.join(ob_bin, "..", "data")
    if os.path.isdir(data) and not os.environ.get("BABEL_DATADIR"):
        os.environ["BABEL_DATADIR"] = data
except Exception:
    pass

import yaml
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

try:
    import rescore_gpu as _rg
except Exception:
    _rg = None

st.set_page_config(page_title="PrismDock — Consensus Molecular Docking", layout="wide",
                   page_icon="\U0001F9EC")

DEFAULT_RESULTS = os.path.join(APP_DIR, "results", "cdk2_rehearsal")


# ---------------------------------------------------------------------------
# Sidebar — run selector (points the whole app at a results directory)
# ---------------------------------------------------------------------------
st.sidebar.title("PrismDock")
st.sidebar.markdown("**P2Rank-Guided Rank-Integrated Smina/Vina Molecular Docking & Validation Suite**")
st.sidebar.caption("Author: Sridhar J D, M.Pharm (Pharmaceutical Chemistry) | GitHub: [github.com/jdsridhar/prismdock](https://github.com/jdsridhar/prismdock) | ORCID: 0009-0009-5196-7100")

def _is_run_dir(d):
    """A real docking run has a consensus_results.csv somewhere under it."""
    for sub in ("docking/consensus_results.csv", "consensus_results.csv"):
        if os.path.exists(os.path.join(results_root, d, sub)):
            return True
    return False

results_root = os.path.join(APP_DIR, "results")
all_dirs = sorted([d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))]) if os.path.isdir(results_root) else []
available_runs = [d for d in all_dirs if _is_run_dir(d)]
if not available_runs:
    st.sidebar.warning("No completed runs found under results/. Launch one in the 'Run' tab.")
    available_runs = all_dirs  # fall back so the app doesn't crash on empty
run_name = st.sidebar.selectbox("Results to display", available_runs,
                                index=available_runs.index("cdk2_rehearsal")
                                if "cdk2_rehearsal" in available_runs else 0)
run_dir = os.path.join(results_root, run_name)

docking_csv_candidates = [
    os.path.join(run_dir, "docking", "consensus_results.csv"),
    os.path.join(run_dir, "consensus_results.csv"),
]
docking_csv = next((p for p in docking_csv_candidates if os.path.exists(p)), None)

# Run-level trust verdict (src/trust.py) — the headline differentiator:
# whether THIS run's protocol has demonstrated it works on THIS target,
# computed automatically by run_dockathon.py before any hit is shown.
trust_path = os.path.join(run_dir, "validation", "trust.json")
trust_data = json.load(open(trust_path)) if os.path.exists(trust_path) else None

st.sidebar.markdown("---")
if trust_data:
    _trust_title = trust_data["status"].replace("_", " ").title()
    st.sidebar.markdown(f"**{trust_data['emoji']} {_trust_title}**")
    st.sidebar.caption(trust_data["headline"] + " See the Validation tab for details.")
else:
    st.sidebar.caption("No trust verdict for this run yet (older run, or trust layer didn't run).")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline stages**\n"
    "1. Target ingestion + receptor prep\n"
    "2. Auto binding-site detection\n"
    "3. Ligand library prep\n"
    "4. Consensus docking (Vina + Smina)\n"
    "5. Self-validation (redock RMSD, AUC/EF/BEDROC)\n"
    "6. Trust & confidence layer\n"
    "7. Report + figures"
)

(tab_run, tab_prep, tab_hits, tab_inter, tab_val, tab_pose, tab_compare, tab_report, tab_present, tab_dl, tab_tech) = st.tabs(
    ["\u25B6 Run", "\U0001F6A6 Prep Check", "\U0001F4CA Hits", "\U0001F517 Interactions",
     "\u2705 Validation", "\U0001F9EA Pose Viewer", "\U0001F52C Compare", "\U0001F4C4 Report",
     "\U0001F3A4 Present", "\u2B07 Downloads", "\U0001F6E0 Tech Stack"]
)

# ---------------------------------------------------------------------------
# Tab 1: Run
# ---------------------------------------------------------------------------
with tab_run:
    st.header("Configure & launch a docking run")
    st.markdown("On event day: set the assigned target and screening library here, "
               "then launch. Everything downstream (pocket detection, docking, "
               "validation, report) runs automatically.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target")
        target_mode = st.radio("Source", ["RCSB PDB ID", "Upload structure file"], horizontal=True)
        if target_mode == "RCSB PDB ID":
            pdb_id = st.text_input("PDB ID", value="1H00")
            uploaded_struct = None
        else:
            pdb_id = None
            uploaded_struct = st.file_uploader("Structure (.pdb/.cif)", type=["pdb", "cif"])
        ph = st.slider("Protonation pH", 5.0, 9.0, 7.4, 0.1)
        dock_native = st.checkbox("Extract & dock co-crystal ligand", value=False)
        auto_pocket = st.checkbox("Auto-detect binding pocket", value=True)
        if not auto_pocket:
            st.caption("Manual Pocket Box")
            cp1, cp2, cp3 = st.columns(3)
            cx = cp1.number_input("Center X", value=0.0)
            cy = cp2.number_input("Center Y", value=0.0)
            cz = cp3.number_input("Center Z", value=0.0)
            sp1, sp2, sp3 = st.columns(3)
            sx = sp1.number_input("Size X", value=22.0)
            sy = sp2.number_input("Size Y", value=22.0)
            sz = sp3.number_input("Size Z", value=22.0)
            force_center = [cx, cy, cz]
            force_size = [sx, sy, sz]
        else:
            force_center = None
            force_size = None

    with col2:
        st.subheader("Screening library")
        lib_mode = st.radio("Source", ["Nucleus (scaffold) \u2192 enumerate", "Bundled library", "Upload CSV", "Paste SMILES", "Upload TXT"], horizontal=True)
        bundled = sorted([f for f in os.listdir(os.path.join(APP_DIR, "data"))
                          if f.endswith(".csv")]) if os.path.isdir(os.path.join(APP_DIR, "data")) else []
        lib_path = None
        def _parse_smiles_line(line, i):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                parts = [p for p in parts if p]
                if not parts: return None
                lid = parts[0]
                smi = parts[1] if len(parts) > 1 else parts[0]
                label = int(parts[2]) if len(parts) > 2 and parts[2] in ["0", "1"] else None
            else:
                parts = line.strip().split()
                if not parts: return None
                smi = parts[0]
                lid = parts[1] if len(parts) > 1 else f"lig_{i+1}"
                label = int(parts[2]) if len(parts) > 2 and parts[2] in ["0", "1"] else None
            row = {"id": lid, "smiles": smi}
            if label is not None:
                row["label"] = label
            return row

        if lib_mode == "Nucleus (scaffold) \u2192 enumerate":
            import enumerate_scaffold as _es
            st.caption(
                "Dockathon **nucleus** mode: paste the assigned core scaffold as SMILES. "
                "Every candidate keeps that core \u2014 it's decorated at its open positions with the "
                "R-groups you pick below. Use `[*]` in the SMILES to mark exact attachment points "
                "(e.g. `O=C(N)c1ccc([*])cc1`); with no `[*]`, every aromatic C\u2013H is scanned."
            )
            nucleus_smiles = st.text_input(
                "Nucleus SMILES", value="Nc1ncncc1",
                help="The fixed core scaffold. Draw it in the Ketcher tile if you prefer, then paste the SMILES here.")
            ec1, ec2 = st.columns(2)
            all_rg = list(_es.RGROUP_LIBRARY.keys())
            chosen_rg = ec1.multiselect(
                "R-groups to attach", all_rg, default=_es.DEFAULT_RGROUPS,
                help="Substituents decorated onto the open positions of the core.")
            max_analogs = ec2.number_input("Max analogs", 5, 2000, 200, step=5,
                help="Cap on the enumerated library size.")
            ec3, ec4 = st.columns(2)
            aromatic_only = ec3.checkbox("Aromatic C\u2013H only", value=True,
                help="Decorate only aromatic ring C\u2013H sites (off = scan all C\u2013H). Ignored when the SMILES has [*].")
            di_sub = ec4.checkbox("Add di-substitution", value=False,
                help="Also enumerate pairs of identical substituents at two sites (larger library).")

            if nucleus_smiles and chosen_rg:
                try:
                    from rdkit import Chem as _Chem
                    _core = _Chem.MolFromSmiles(nucleus_smiles)
                    if _core is None:
                        st.error("Could not parse that nucleus SMILES \u2014 check it.")
                    else:
                        enum_df = _es.enumerate_analogs(
                            nucleus_smiles, rgroup_names=chosen_rg,
                            max_analogs=int(max_analogs),
                            aromatic_only=aromatic_only, di_substitution=di_sub)
                        enum_df = _es.add_properties(enum_df)
                        lib_path = os.path.join(APP_DIR, "data", "enumerated_scaffold.csv")
                        _es.write_library(enum_df, lib_path)
                        n_pass = int(enum_df["Lipinski_pass"].sum())
                        st.success(
                            f"Enumerated **{len(enum_df)}** analogs around the nucleus "
                            f"({n_pass} pass Lipinski). They keep the core and will be docked as the library.")
                        # 2D grid preview of the first analogs
                        try:
                            grid_png = os.path.join(APP_DIR, "data", "_enum_grid.png")
                            _es.grid_image(enum_df, grid_png, mols_per_row=5, max_mols=15)
                            if os.path.exists(grid_png):
                                st.image(grid_png, caption="Enumerated analogs (first 15)", use_container_width=True)
                        except Exception:
                            pass
                        with st.expander("Enumerated library table"):
                            st.dataframe(
                                enum_df[["id", "r_group", "sites", "smiles",
                                         "MW", "logP", "HBD", "HBA", "TPSA", "Lipinski_pass"]],
                                use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Enumeration failed: {e}")
        elif lib_mode == "Bundled library":
            lib_csv = st.selectbox("Library CSV (id, smiles[, label])", bundled)
            lib_path = os.path.join(APP_DIR, "data", lib_csv) if lib_csv else None
        elif lib_mode == "Upload CSV":
            up = st.file_uploader("Library CSV", type=["csv"])
            if up:
                lib_path = os.path.join(APP_DIR, "data", "uploaded_library.csv")
                with open(lib_path, "wb") as fh:
                    fh.write(up.getbuffer())
        elif lib_mode == "Paste SMILES":
            smiles_text = st.text_area("Paste SMILES (one per line, formats: 'SMILES ID' or 'ID,SMILES')", height=150)
            if smiles_text:
                lines = smiles_text.strip().split("\n")
                rows = [_parse_smiles_line(line, i) for i, line in enumerate(lines)]
                rows = [r for r in rows if r is not None]
                if rows:
                    lib_path = os.path.join(APP_DIR, "data", "pasted_smiles.csv")
                    pd.DataFrame(rows).to_csv(lib_path, index=False)
                    st.success(f"Parsed {len(rows)} SMILES.")
        elif lib_mode == "Upload TXT":
            up_txt = st.file_uploader("SMILES TXT file (one per line, formats: 'SMILES ID' or 'ID,SMILES')", type=["txt", "smi", "csv"])
            if up_txt:
                lines = up_txt.read().decode("utf-8").strip().split("\n")
                rows = [_parse_smiles_line(line, i) for i, line in enumerate(lines)]
                rows = [r for r in rows if r is not None]
                if rows:
                    lib_path = os.path.join(APP_DIR, "data", "uploaded_smiles.csv")
                    pd.DataFrame(rows).to_csv(lib_path, index=False)
                    st.success(f"Parsed {len(rows)} SMILES.")

    st.subheader("Docking parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    exhaustiveness = c1.number_input("Exhaustiveness", 1, 32, 8)
    n_poses = c2.number_input("Poses per ligand", 1, 20, 9)
    seed = c3.number_input("Seed", 0, 999999, 42)
    limit = c4.number_input("Ligand cap (0 = all)", 0, 10000, 0)
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count() or 1
    # Default reserves 2 cores rather than maxing out every core: an
    # event-day machine also has to keep running this Streamlit UI, SSH,
    # and whatever else is open, and a fully-saturated machine gets
    # sluggish/unresponsive right when you need to demo it. Still
    # selectable up to all cores if you want maximum throughput instead.
    default_workers = max(1, cpu_cores - 2)
    n_workers = c5.number_input("Parallel CPU workers", 1, max(1, cpu_cores), default_workers,
                                help=f"{cpu_cores} cores detected. Defaults to {default_workers} "
                                     "(reserves 2 for the UI/OS) so the machine stays responsive "
                                     "during a big run; raise it for maximum throughput.")

    with st.expander("\u26A1 Optional GPU rescoring (third consensus member)", expanded=False):
        st.caption(
            "The Vina+Smina consensus above is the validated core and needs no GPU. "
            "On a machine with a CUDA GPU and the extra binaries installed, you can fold in "
            "a third, independent opinion. Both are no-ops if the binary/GPU isn't found -- "
            "the CPU consensus still runs and the pipeline won't fail."
        )
        gc1, gc2 = st.columns(2)
        gnina_found_path = _rg.find_gnina() if _rg else None
        gnina_available = gnina_found_path is not None
        gnina_label = "GNINA CNN rescoring -- 3rd consensus engine"
        if gnina_available:
            gnina_label += f" (found: {gnina_found_path})"
        else:
            gnina_label += " (not found -- see below)"
        use_gnina = gc1.checkbox(
            gnina_label,
            value=False, key="rescore_gnina",
            help="Re-scores the existing Vina poses with GNINA's CNN pose/affinity model, "
                 "then MERGES the result into consensus_results.csv and re-ranks with all "
                 "three engines (Vina+Smina+GNINA) -- a genuine third vote, benchmark-backed "
                 "(see README). The pre-GNINA ranking is kept as consensus_results_2engine.csv. "
                 "Works fine on CPU only (auto-fallback, just slower) -- a GPU is NOT required. "
                 "Detected on PATH, in this app's working/repo directory, or at $GNINA_PATH. "
                 "Install: conda install -c gnina gnina, or download the standalone binary "
                 "(see README).",
        )
        use_diffdock = gc2.checkbox(
            "DiffDock-L blind poses -- pocket sanity check (not a ranking input)",
            value=False, key="rescore_diffdock",
            help="Box-free diffusion docking used ONLY to sanity-check the pocket: does its "
                 "blind top pose land inside the box the rest of the pipeline assumed "
                 "(diffdock_in_box)? Its poses are informational and never folded into "
                 "consensus_z -- benchmarks show it trails Vina/GNINA on pose accuracy when "
                 "the pocket is already known (see README). Requires a DiffDock clone + model "
                 "weights on a GPU machine.",
        )
        if not gnina_available:
            manual_gnina = st.text_input(
                "Or point directly at a gnina binary (full path)",
                value="", key="gnina_manual_path",
                placeholder="/home/you/prismdock/gnina",
                help="Use this if your gnina binary isn't on PATH and isn't sitting in this "
                     "app's own folder -- e.g. you downloaded it somewhere else.")
            if manual_gnina:
                os.environ["GNINA_PATH"] = manual_gnina
                gnina_available = (_rg.find_gnina() is not None) if _rg else False
        if use_gnina and not gnina_available:
            st.warning("gnina checked but not found on PATH, in the working/repo directory, "
                       "or at $GNINA_PATH -- the run will skip CNN rescoring and continue "
                       "with the CPU consensus only. If you downloaded the binary into this "
                       "project's folder, that's checked automatically; otherwise paste the "
                       "full path above, or add its folder to PATH and relaunch.")
        elif use_gnina and gnina_available:
            st.caption("No GPU needed -- gnina auto-detects the absence of one and runs the "
                       "CNN scoring on CPU (slower, same result). CPU-only comparison of "
                       "Vina+Smina (the validated core) is always available regardless of "
                       "GNINA/GPU status.")

    run_label = st.text_input("Run name (output folder under results/)", value="live_run")

    launch = st.button("\U0001F680 Launch pipeline", type="primary")

    log_box = st.empty()
    if launch:
        if not pdb_id and not uploaded_struct:
            st.error("Provide a PDB ID or upload a structure.")
        elif not lib_path:
            st.error("Provide a screening library CSV.")
        else:
            out_run_dir = os.path.join(results_root, run_label)
            os.makedirs(out_run_dir, exist_ok=True)

            infile_path = None
            if uploaded_struct is not None:
                infile_path = os.path.join(out_run_dir, uploaded_struct.name)
                with open(infile_path, "wb") as fh:
                    fh.write(uploaded_struct.getbuffer())

            cfg = {
                "target": {"pdb": pdb_id, "infile": infile_path, "ph": float(ph), "dock_native": dock_native},
                "pocket": {"padding": 5.0, "apo_size": 22.0,
                          "force_center": force_center, "force_size": force_size},
                "library": {"csv": lib_path, "limit": None if limit == 0 else int(limit)},
                "docking": {"engines": ["vina", "smina"], "exhaustiveness": int(exhaustiveness),
                           "n_poses": int(n_poses), "seed": int(seed), "smina_bin": "smina",
                           "n_workers": int(n_workers)},
                "rescoring": {"gnina_cnn": bool(use_gnina), "diffdock": bool(use_diffdock)},
                "validation": {"redock_native": True, "enrichment": True},
                "output": {"outdir": out_run_dir, "report": True},
            }
            cfg_path = os.path.join(out_run_dir, "run_config.yaml")
            import yaml
            yaml.safe_dump(cfg, open(cfg_path, "w"))

            st.info(f"Launching: `python run_dockathon.py --config {cfg_path}`")
            proc = subprocess.Popen(
                [sys.executable, os.path.join(APP_DIR, "run_dockathon.py"),
                 "--config", cfg_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd=APP_DIR,
            )
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-40:]))
            proc.wait()
            if proc.returncode == 0:
                st.success(f"Done. Select '{run_label}' in the sidebar to view results.")
            else:
                st.error(f"Pipeline exited with code {proc.returncode}. See log above.")

# ---------------------------------------------------------------------------
# Tab: Prep Check — docking-readiness QC (traffic light)
# ---------------------------------------------------------------------------
with tab_prep:
    st.header(f"Docking-readiness QC — {run_name}")
    st.caption("Confirms the receptor and every ligand were prepared correctly "
               "*before* docking. A broken PDBQT still produces a score — a meaningless one. "
               "This catches missing charges, unprotonated receptors, and malformed torsion trees.")

    import prep_check as _pc

    qc = _pc.check_run(run_dir)
    summary = qc["summary"]

    _emoji = {"pass": "\U0001F7E2", "warn": "\U0001F7E1", "fail": "\U0001F534"}

    # Overall verdict banner
    if summary["overall"] == "pass":
        st.success(f"{_emoji['pass']} {summary['verdict']}")
    elif summary["overall"] == "warn":
        st.warning(f"{_emoji['warn']} {summary['verdict']}")
    else:
        st.error(f"{_emoji['fail']} {summary['verdict']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ligands", summary["n_ligands"])
    m2.metric("\U0001F7E2 Pass", summary["n_pass"])
    m3.metric("\U0001F7E1 Warn", summary["n_warn"])
    m4.metric("\U0001F534 Fail", summary["n_fail"])

    # Target card
    st.subheader("Target (receptor)")
    trec = qc["target"]
    st.markdown(f"**{_emoji[trec['status']]} {trec['name']}** — {trec['status'].upper()}")
    for reason in trec["reasons"]:
        st.markdown(f"- {reason}")
    if trec["metrics"]:
        st.caption(" | ".join(f"{k}: {v}" for k, v in trec["metrics"].items()))

    # Ligand table
    st.subheader("Ligand library")
    if not qc["ligands"]:
        st.info("No prepared ligand PDBQT files found for this run "
                "(run the pipeline first, or check results/<run>/library/pdbqt/).")
    else:
        lig_rows = []
        for r in qc["ligands"]:
            lig_rows.append({
                "": _emoji[r["status"]],
                "ligand": r["name"],
                "status": r["status"].upper(),
                "atoms": r["metrics"].get("n_atoms"),
                "rot_bonds": r["metrics"].get("n_branches"),
                "net_charge": r["metrics"].get("net_charge"),
                "notes": "; ".join(r["reasons"]),
            })
        lig_df = pd.DataFrame(lig_rows)

        def _row_color(row):
            c = {"PASS": "#123d1f", "WARN": "#5a4a12", "FAIL": "#5a1a1a"}.get(row["status"], "")
            return [f"background-color: {c}"] * len(row)

        st.dataframe(lig_df.style.apply(_row_color, axis=1),
                     width='stretch', hide_index=True)

        st.download_button(
            "\u2B07 Download prep-check report (CSV)",
            lig_df.to_csv(index=False).encode(),
            file_name=f"{run_name}_prep_check.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 2: Hits table
# ---------------------------------------------------------------------------
with tab_hits:
    st.header(f"Consensus hit ranking — {run_name}")
    if trust_data:
        _tier_note = ("Every row below also carries a `confidence_tier` — not the same as the "
                     "score, but whether the score should be believed (see Validation tab)."
                     if trust_data["status"] in ("validated", "partially_validated")
                     else "This run is **" + trust_data["status"].replace("_", " ") +
                     "** (see Validation tab) — every hit's confidence is capped accordingly, "
                     "regardless of how well it scores.")
        st.caption(f"{trust_data['emoji']} {_tier_note}")
    if docking_csv is None:
        st.warning("No consensus_results.csv found for this run yet.")
    else:
        df = pd.read_csv(docking_csv)
        has_labels = "label" in df.columns and df["label"].notna().any()
        has_gnina = "gnina_score" in df.columns and df["gnina_score"].notna().any()
        if has_gnina:
            st.caption("GNINA CNN rescoring is folded into this ranking as a third consensus "
                      "engine (`n_engines_scored` shows how many of Vina/Smina/GNINA scored "
                      "each ligand; `agreement` is the max-min spread across all engines that scored it).")

        colf1, colf2, colf3 = st.columns(3)
        min_agree = colf1.slider("Min agreement", 0.0, 1.0, 0.0, 0.05)
        only_actives = colf2.checkbox("Known actives only", value=False, disabled=not has_labels)
        n_rows = len(df)
        top_n_min = min(5, n_rows) if n_rows > 0 else 0
        top_n_default = min(25, n_rows) if n_rows > 0 else 0
        top_n = colf3.number_input("Show top N", top_n_min, max(n_rows, top_n_min), top_n_default,
                                   disabled=(n_rows == 0))

        fdf = df.copy()
        if "agreement" in fdf.columns:
            fdf = fdf[fdf["agreement"] >= min_agree]
        if only_actives and has_labels:
            fdf = fdf[fdf["label"] == 1]
        sort_col = "consensus_z" if "consensus_z" in fdf.columns else fdf.columns[-1]
        fdf = fdf.sort_values(sort_col).head(int(top_n))

        # Add ADMET properties if manifest is available
        manifest_csv = os.path.join(run_dir, "library", "ligand_manifest.csv")
        if os.path.exists(manifest_csv):
            man_df = pd.read_csv(manifest_csv)
            if "smiles" in man_df.columns:
                fdf = fdf.merge(man_df[["id", "smiles"]], on="id", how="left")
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Descriptors, rdMolDescriptors
                    def get_admet(smi):
                        if not isinstance(smi, str): return (None, None, None, None, None)
                        m = Chem.MolFromSmiles(smi)
                        if not m: return (None, None, None, None, None)
                        return (
                            round(Descriptors.MolWt(m), 2),
                            round(Descriptors.MolLogP(m), 2),
                            rdMolDescriptors.CalcNumLipinskiHBD(m),
                            rdMolDescriptors.CalcNumLipinskiHBA(m),
                            round(Descriptors.TPSA(m), 2)
                        )
                    admet_vals = fdf["smiles"].apply(get_admet)
                    fdf["MW"] = [x[0] for x in admet_vals]
                    fdf["LogP"] = [x[1] for x in admet_vals]
                    fdf["HBD"] = [x[2] for x in admet_vals]
                    fdf["HBA"] = [x[3] for x in admet_vals]
                    fdf["TPSA"] = [x[4] for x in admet_vals]
                except Exception as e:
                    st.caption(f"Could not compute ADMET properties: {e}")

        if "confidence_tier" in fdf.columns:
            def _confidence_color(row):
                c = {"High": "#123d1f", "Medium": "#5a4a12", "Low": "#5a1a1a"}.get(row["confidence_tier"], "")
                return [f"background-color: {c}"] * len(row)
            st.dataframe(fdf.style.apply(_confidence_color, axis=1), width='stretch', height=520)
        else:
            st.dataframe(fdf, width='stretch', height=520)
        st.caption(f"{len(fdf)} of {len(df)} ligands shown \u00b7 sorted by consensus score (lower = better)")
        if "confidence_reason" in fdf.columns and not fdf.empty:
            with st.expander(f"Why is {fdf.iloc[0]['id']} (top row) rated "
                             f"{fdf.iloc[0].get('confidence_tier', '?')} confidence?"):
                st.caption(fdf.iloc[0]["confidence_reason"])

        csv_bytes = fdf.to_csv(index=False).encode()
        st.download_button("Download filtered hits (CSV)", csv_bytes,
                          file_name=f"{run_name}_top_hits.csv")

# ---------------------------------------------------------------------------
# Tab: Interactions — 2D protein-ligand interaction diagram (LigPlot+ analogue)
# ---------------------------------------------------------------------------
with tab_inter:
    st.header(f"Protein–ligand interactions — {run_name}")
    st.caption("2D interaction network (open-source LigPlot+ analogue via ProLIF): "
               "H-bonds, hydrophobic contacts, pi-stacking, salt bridges and van-der-Waals "
               "contacts between a docked pose and the receptor.")

    try:
        import interactions as _inter
        import streamlit.components.v1 as _components

        pose_map = _inter.list_pose_sdfs(run_dir)
        receptor_pdb = os.path.join(run_dir, "target", "receptor_clean.pdb")

        if not pose_map:
            st.warning("No SDF pose files found for this run "
                       "(interactions need results/<run>/docking/poses/*_smina.sdf).")
        elif not os.path.exists(receptor_pdb):
            st.warning("Prepared receptor (target/receptor_clean.pdb) not found for this run.")
        else:
            # order ligands by consensus rank if we have it
            if docking_csv is not None:
                _rank = pd.read_csv(docking_csv)
                _order = [i for i in _rank.sort_values(
                    "consensus_z" if "consensus_z" in _rank.columns else _rank.columns[-1]
                )["id"].tolist() if i in pose_map]
                lig_options = _order + [i for i in pose_map if i not in _order]
            else:
                lig_options = list(pose_map)

            csel, psel = st.columns([2, 1])
            chosen_lig = csel.selectbox("Ligand", lig_options, key="inter_lig")
            # how many poses in that sdf (Smina side -- Vina/DiffDock comparison
            # always uses each engine's own top pose, noted below)
            import prolif as _plf
            _n_poses = len(list(_plf.sdf_supplier(pose_map[chosen_lig])))
            pose_idx = psel.number_input("Pose #", 1, max(_n_poses, 1), 1, key="inter_pose") - 1

            with st.spinner("Computing interactions for every engine's pose..."):
                multi_res = _inter.analyze_pose_multi(receptor_pdb, run_dir, chosen_lig,
                                                       pose_index=int(pose_idx))
            if int(pose_idx) > 0 and "vina" in multi_res:
                st.caption("Note: Pose # only changes the Smina/DiffDock pose shown -- the Vina "
                          "column always compares Vina's own top-ranked pose.")

            ok_engines = {e: r for e, r in multi_res.items() if r["ok"]}
            if not ok_engines:
                for e, r in multi_res.items():
                    st.warning(f"{e}: {r['message']}")
            else:
                cmp_table = _inter.contact_comparison_table(multi_res)
                n_max = len(ok_engines)
                n_shared = int((cmp_table["n_engines"] == n_max).sum()) if not cmp_table.empty and n_max > 1 else 0
                summary_bits = [f"**{e.title()}**: {r['table'].shape[0]} contacts, "
                                f"{r['table']['residue'].nunique()} residues"
                                for e, r in ok_engines.items()]
                st.markdown(" · ".join(summary_bits))
                if n_max > 1:
                    st.caption(
                        f"{n_shared} contact(s) are made by **all {n_max} engines** -- the "
                        "geometrically robust core of the binding mode, independent of which "
                        "search algorithm found it. Rows made by only one engine reflect a real "
                        "difference in the pose each engine settled on, not just a scoring "
                        "difference -- Vina and Smina search independently and don't always agree "
                        "on geometry even when their scores agree.")

                st.markdown("**2D diagrams to show**")
                engine_names = list(ok_engines.keys())
                check_cols = st.columns(len(engine_names))
                show_engine = {}
                for col, eng in zip(check_cols, engine_names):
                    show_engine[eng] = col.checkbox(eng.title(), value=True, key=f"inter_show_{eng}")

                for eng in engine_names:
                    if not show_engine[eng]:
                        continue
                    res = ok_engines[eng]
                    st.markdown(f"##### {eng.title()}")
                    st.caption(res["message"])
                    html = _inter.lignetwork_html(res)
                    if html:
                        _components.html(html, height=500, scrolling=True)
                    else:
                        st.info(res.get("message", "Diagram unavailable; see comparison table below."))

                st.subheader("Contact comparison across engines")
                tbl = cmp_table
                def _agree_color(row):
                    if row["n_engines"] == n_max and n_max > 1:
                        c = "#123d1f"
                    elif row["n_engines"] > 1:
                        c = "#5a4a12"
                    else:
                        c = "#5a1a1a"
                    return [f"background-color: {c}"] * len(row)
                if tbl.empty:
                    st.info("No contacts to compare.")
                else:
                    st.dataframe(tbl.style.apply(_agree_color, axis=1), width='stretch', hide_index=True)
                st.download_button(
                    "\u2B07 Download comparison (CSV)",
                    tbl.to_csv(index=False).encode(),
                    file_name=f"{run_name}_{chosen_lig}_contact_comparison.csv", mime="text/csv")
                st.caption("GNINA isn't shown separately: it re-scores Vina's exact pose rather "
                          "than searching for its own, so its contacts would just duplicate the "
                          "Vina column.")
    except ImportError as e:
        st.error(f"Interaction analysis dependencies not available: {e}. "
                 "Install with: pip install prolif")


# ---------------------------------------------------------------------------
# Tab 3: Validation (interactive Plotly)
# ---------------------------------------------------------------------------
with tab_val:
    st.header("Self-validation")

    if trust_data:
        _banner = {"validated": st.success, "partially_validated": st.warning,
                  "failed": st.error, "unvalidated": st.info}.get(trust_data["status"], st.info)
        _title = trust_data["status"].replace("_", " ").title()
        _banner(f"{trust_data['emoji']} **{_title}** — {trust_data['headline']}")
        with st.expander("Why this verdict — the checks behind it"):
            for r in trust_data["reasons"]:
                st.markdown(f"- {r}")
            st.caption(
                "Computed automatically before any hit is shown, the same way on every run. "
                "Every ranked compound's own confidence tier (High/Medium/Low, see Hits tab) "
                "is capped by this verdict — strong cross-engine agreement on a hit can't "
                "outrank the protocol's own demonstrated reliability on this target.")
        st.divider()

    redock_path_candidates = [
        os.path.join(run_dir, "validation", "redock_rmsd.json"),
        os.path.join(APP_DIR, "results", "1H00", "redock", "redock_rmsd.json"),
    ]
    redock_path = next((p for p in redock_path_candidates if os.path.exists(p)), None)

    if docking_csv is None:
        st.warning("No docking results for this run yet.")
    else:
        df = pd.read_csv(docking_csv)
        if "consensus_z" in df.columns:
            df = df.dropna(subset=["consensus_z"])
        has_labels = "label" in df.columns and df["label"].notna().any()

        st.subheader("Pose accuracy — redock native ligand")
        if redock_path:
            rd = json.load(open(redock_path))
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Top-pose RMSD", f"{rd.get('rmsd_top_pose')} \u00c5",
                      help="RMSD to the crystal ligand pose; <2 \u00c5 = validated protocol")
            rc2.metric(f"Best-of-{rd.get('n_poses','?')} RMSD", f"{rd.get('rmsd_best_pose')} \u00c5")
            rc3.metric("Validated (<2\u00c5)?", "\u2705 Yes" if rd.get("success_2A") else "\u26a0\ufe0f No")
        else:
            st.info("No redock validation found for this run — run redock_rmsd() to populate this panel.")

        if has_labels:
            labels = df["label"].astype(int).values
            st.subheader("Screening power — actives vs decoys")
            has_gnina = "gnina_score" in df.columns and df["gnina_score"].notna().any()
            m1, m2, m3, m4, m5 = st.columns(5)
            auc_c = roc_auc_score(labels, -df["consensus_z"].values)
            n_eng = 3 if has_gnina else 2
            m1.metric(f"AUC-ROC (consensus, {n_eng}-engine)", f"{auc_c:.3f}")
            if "vina_score" in df:
                m2.metric("AUC-ROC (Vina)", f"{roc_auc_score(labels, -df['vina_score'].values):.3f}")
            if "smina_score" in df:
                m3.metric("AUC-ROC (Smina)", f"{roc_auc_score(labels, -df['smina_score'].values):.3f}")
            if has_gnina:
                gdf = df.dropna(subset=["gnina_score"])
                m4.metric("AUC-ROC (GNINA)",
                         f"{roc_auc_score(gdf['label'].astype(int).values, -gdf['gnina_score'].values):.3f}")
            m5.metric("N actives / decoys", f"{int(labels.sum())} / {int((labels==0).sum())}")
            if has_gnina:
                st.caption("GNINA CNN rescoring is enabled for this run — the consensus above is a genuine "
                          "3-engine vote (Vina + Smina + GNINA), not just Vina+Smina. See "
                          "`consensus_results_2engine.csv` in this run's folder for the pre-GNINA ranking.")

            fig_roc = go.Figure()
            curves = [("Consensus", labels, -df["consensus_z"].values, "#1b4f9c", 3)]
            if "vina_score" in df:
                curves.append(("Vina", labels, -df["vina_score"].values, "#e07b39", 2))
            if "smina_score" in df:
                curves.append(("Smina", labels, -df["smina_score"].values, "#3a9c6e", 2))
            if has_gnina:
                gdf = df.dropna(subset=["gnina_score"])
                curves.append(("GNINA", gdf["label"].astype(int).values,
                              -gdf["gnina_score"].values, "#9c3a8e", 2))
            for name, curve_labels, score, color, w in curves:
                fpr, tpr, _ = roc_curve(curve_labels, score)
                auc = roc_auc_score(curve_labels, score)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                            name=f"{name} (AUC={auc:.3f})",
                                            line=dict(color=color, width=w)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                        name="Random", line=dict(color="grey", dash="dash")))
            fig_roc.update_layout(title="ROC — actives vs decoys", xaxis_title="False positive rate",
                                 yaxis_title="True positive rate", height=450,
                                 legend=dict(x=0.55, y=0.05))
            st.plotly_chart(fig_roc, width='stretch')

            colA, colB = st.columns(2)
            with colA:
                d = df.sort_values("consensus_z").reset_index(drop=True)
                n = len(d); n_act = int(d["label"].sum())
                frac_screened = np.arange(1, n + 1) / n * 100
                frac_found = np.cumsum(d["label"].values) / n_act * 100
                fig_enr = go.Figure()
                fig_enr.add_trace(go.Scatter(x=frac_screened, y=frac_found, mode="lines",
                                            name="Consensus", line=dict(color="#1b4f9c", width=3)))
                fig_enr.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                            name="Random", line=dict(color="grey", dash="dash")))
                fig_enr.update_layout(title="Enrichment", xaxis_title="% library screened",
                                     yaxis_title="% actives recovered", height=400)
                st.plotly_chart(fig_enr, width='stretch')
            with colB:
                fig_dist = go.Figure()
                for lab, name, c in [(1, "Actives", "#c0392b"), (0, "Decoys", "#7f8c8d")]:
                    vals = df.loc[df["label"] == lab, "consensus_z"]
                    fig_dist.add_trace(go.Histogram(x=vals, name=f"{name} (n={len(vals)})",
                                                   marker_color=c, opacity=0.65, histnorm="probability density"))
                fig_dist.update_layout(barmode="overlay", title="Score separation",
                                      xaxis_title="Consensus score (lower=better)", height=400)
                st.plotly_chart(fig_dist, width='stretch')

            fig_agree = px.scatter(df, x="vina_score", y="smina_score",
                                  color=df["label"].map({1: "Active", 0: "Decoy"}),
                                  color_discrete_map={"Active": "#c0392b", "Decoy": "#7f8c8d"},
                                  title=f"Cross-engine agreement (r={np.corrcoef(df['vina_score'], df['smina_score'])[0,1]:.2f})",
                                  labels={"vina_score": "Vina affinity (kcal/mol)",
                                         "smina_score": "Smina affinity (kcal/mol)"},
                                  hover_data=["id"])
            st.plotly_chart(fig_agree, width='stretch')
        else:
            st.info("This library has no active/decoy labels (prospective screening mode) — "
                   "enrichment metrics require a labeled benchmark set.")

# ---------------------------------------------------------------------------
# Tab 4: 3D pose viewer
# ---------------------------------------------------------------------------
with tab_pose:
    st.header("Advanced 3D pose viewer")
    st.caption("Docked pose(s) in the receptor pocket — surface, interaction lines, "
               "pocket residues (3-letter labels) and multi-ligand overlay in distinct colors.")
    try:
        import stmol
        from stmol import showmol
        import py3Dmol
        import numpy as _np

        receptor_pdb_candidates = [
            os.path.join(run_dir, "target", "receptor_clean.pdb"),
            os.path.join(APP_DIR, "results", "1H00", "receptor_clean.pdb"),
        ]
        receptor_pdb = next((p for p in receptor_pdb_candidates if os.path.exists(p)), None)
        pose_dir = os.path.join(run_dir, "docking", "poses")

        # 3Dmol.js built-in carbon colorschemes -- one per ligand so multi-select
        # overlays are visually distinguishable at a glance.
        LIG_PALETTE = ["greenCarbon", "cyanCarbon", "magentaCarbon", "orangeCarbon",
                       "purpleCarbon", "yellowCarbon", "pinkCarbon", "blueCarbon"]
        LIG_HEX = ["#2ecc71", "#17becf", "#e91e8c", "#f39c12",
                   "#9b59b6", "#f1c40f", "#ff6fae", "#3498db"]

        if docking_csv is None or receptor_pdb is None:
            st.warning("Receptor structure or docking results not found for this run.")
        else:
            df = pd.read_csv(docking_csv)
            sort_col = "consensus_z" if "consensus_z" in df.columns else df.columns[-1]
            df = df.sort_values(sort_col).reset_index(drop=True)
            options = df["id"].tolist()

            # --- controls --------------------------------------------------
            cc1, cc2, cc3 = st.columns(3)
            default_sel = options[:1]
            chosen_ligs = cc1.multiselect(
                "Ligand(s) — select multiple to compare poses side-by-side",
                options, default=default_sel, key="pose_ligs")
            receptor_style = cc2.selectbox("Receptor style",
                                           ["cartoon", "surface", "cartoon + surface", "sticks"],
                                           key="pose_style")
            show_pocket = cc3.checkbox("Pocket residues (sticks + 3-letter labels)",
                                       value=True, key="pose_pocket")

            cc4, cc5, cc6 = st.columns(3)
            show_interactions = cc4.checkbox("Interaction lines", value=True, key="pose_inter")
            multipose = cc5.checkbox("Overlay all poses (single ligand only)",
                                     value=False, key="pose_multi",
                                     disabled=(len(chosen_ligs) != 1))
            spin = cc6.checkbox("Spin", value=False, key="pose_spin")

            if not chosen_ligs:
                st.info("Select at least one ligand above to render its pose.")
            else:
                view = py3Dmol.view(width=900, height=600)
                view.addModel(open(receptor_pdb).read(), "pdb")

                # receptor styling (model 0)
                if receptor_style in ("cartoon", "cartoon + surface"):
                    view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
                elif receptor_style == "sticks":
                    view.setStyle({"model": 0}, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.1}})
                else:
                    view.setStyle({"model": 0}, {})
                if receptor_style in ("surface", "cartoon + surface"):
                    view.addSurface(py3Dmol.VDW, {"opacity": 0.55, "color": "white"}, {"model": 0})

                legend_rows = []       # (ligand, color_hex) for the on-screen legend
                all_resis = {}         # (chain, resi) -> resn, union across all chosen ligands
                per_lig_contacts = {}  # ligand -> ProLIF contact table (for "why best" + lines)
                per_lig_lines = {}     # ligand -> [(start_xyz, end_xyz, interaction_type), ...]
                model_idx = 1          # model 0 is the receptor
                _SPECIFIC_ITYPES = {"HBDonor", "HBAcceptor", "Cationic", "Anionic",
                                    "CationPi", "PiCation", "PiStacking"}
                _ITYPE_COLOR = {"HBDonor": "cyan", "HBAcceptor": "cyan",
                                "Cationic": "magenta", "Anionic": "magenta",
                                "CationPi": "orange", "PiCation": "orange", "PiStacking": "orange"}

                for i, lid in enumerate(chosen_ligs):
                    color = LIG_PALETTE[i % len(LIG_PALETTE)]
                    hexcolor = LIG_HEX[i % len(LIG_HEX)]
                    legend_rows.append((lid, hexcolor))
                    pose_pdbqt = os.path.join(pose_dir, f"{lid}_vina.pdbqt")
                    if not os.path.exists(pose_pdbqt):
                        st.info(f"No saved pose file for {lid}.")
                        continue
                    txt = open(pose_pdbqt).read()
                    if multipose and len(chosen_ligs) == 1:
                        view.addModelsAsFrames(txt, "pdbqt")
                        view.setStyle({"model": -1}, {"stick": {"colorscheme": color}})
                        view.zoomTo({"model": -1})
                    else:
                        first = []
                        for ln in txt.splitlines():
                            first.append(ln)
                            if ln.startswith("ENDMDL"):
                                break
                        view.addModel("\n".join(first), "pdbqt")
                        view.setStyle({"model": model_idx},
                                      {"stick": {"colorscheme": color, "radius": 0.2}})
                        model_idx += 1

                    # pocket residues + contacts, per ligand. Analyzed against
                    # VINA's pose (converted to SDF), not Smina's -- the
                    # sticks rendered above are Vina's geometry, so contacts
                    # (and the interaction lines drawn from them) need to
                    # come from the SAME pose or they won't align with what's
                    # actually on screen.
                    try:
                        import interactions as _inter
                        vina_sdf = _inter.vina_pose_to_sdf(pose_pdbqt)
                        res = _inter.analyze_pose(receptor_pdb, vina_sdf, pose_index=0) if vina_sdf else None
                        if res and res["ok"]:
                            per_lig_contacts[lid] = res["table"]
                            lig_conf = res["ligand"].GetConformer()
                            prot_conf = res["protein"].GetConformer()
                            lines = []
                            for _, crow in res["table"].iterrows():
                                if crow["interaction"] not in _SPECIFIC_ITYPES:
                                    continue
                                li, pi = crow["ligand_atom_idx"], crow["protein_atom_idx"]
                                if li is None or pi is None:
                                    continue
                                lp = lig_conf.GetAtomPosition(int(li))
                                pp = prot_conf.GetAtomPosition(int(pi))
                                lines.append(((lp.x, lp.y, lp.z), (pp.x, pp.y, pp.z), crow["interaction"]))
                            per_lig_lines[lid] = lines
                            for r in res["table"]["residue"].tolist():
                                # ProLIF residue strings look like "ASP25.A"
                                base = r.split(".")[0]
                                chain = r.split(".")[1] if "." in r else "A"
                                resn = "".join(ch for ch in base if ch.isalpha())
                                resi_digits = "".join(ch for ch in base if ch.isdigit())
                                if resi_digits:
                                    all_resis[(chain, int(resi_digits))] = resn
                    except Exception as _e:  # noqa: BLE001
                        st.caption(f"Interaction overlay unavailable for {lid}: {_e}")

                if not multipose or len(chosen_ligs) != 1:
                    view.zoomTo({"model": list(range(1, model_idx))})
                if model_idx == 1 and not multipose:
                    view.zoomTo()

                if show_pocket and all_resis:
                    for (chain, resi), resn in all_resis.items():
                        sel = {"model": 0, "chain": chain, "resi": resi}
                        view.addStyle(sel, {"stick": {"colorscheme": "grayCarbon", "radius": 0.12}})
                        view.addLabel(f"{resn}{resi}",
                                      {"fontSize": 10, "backgroundOpacity": 0.55,
                                       "backgroundColor": "black", "fontColor": "white"},
                                      sel)
                    st.caption(f"{len(all_resis)} pocket residue(s) in contact with the selected "
                               "pose(s), labeled with 3-letter code + residue number.")

                n_lines_drawn = 0
                if show_interactions and per_lig_lines:
                    for lid, lines in per_lig_lines.items():
                        for start, end, itype in lines:
                            color = _ITYPE_COLOR.get(itype, "yellow")
                            view.addLine({
                                "start": {"x": start[0], "y": start[1], "z": start[2]},
                                "end": {"x": end[0], "y": end[1], "z": end[2]},
                                "color": color, "dashed": True, "linewidth": 3,
                            })
                            n_lines_drawn += 1

                if spin:
                    view.spin(True)
                view.setBackgroundColor("white")
                showmol(view, height=600, width=900)

                if legend_rows:
                    legend_html = " &nbsp;&nbsp; ".join(
                        f'<span style="color:{c}; font-weight:600;">\u25CF</span> {lid}'
                        for lid, c in legend_rows)
                    st.markdown(legend_html, unsafe_allow_html=True)
                caption_bits = ["Gray sticks = pocket residues (3-letter label)"]
                if show_interactions:
                    caption_bits.append(
                        f"dashed lines = specific contacts ({n_lines_drawn} drawn: "
                        "cyan=H-bond, magenta=ionic, orange=pi-type). Generic "
                        "hydrophobic/van-der-Waals contacts are not lined -- too many "
                        "to stay readable -- see the Interactions tab full table.")
                st.caption(" -- ".join(caption_bits))

                # --- "why is this the best pose / ligand" explanation ------
                if len(chosen_ligs) >= 1:
                    st.subheader("Why this pose ranks where it does")
                    for lid in chosen_ligs:
                        row = df[df["id"] == lid]
                        if row.empty:
                            continue
                        row = row.iloc[0]
                        bits = []
                        if "final_rank" in row:
                            bits.append(f"rank **#{int(row['final_rank'])}** of {len(df)}")
                        engine_scores = []
                        for col, label in [("vina_score", "Vina"), ("smina_score", "Smina"),
                                           ("gnina_score", "GNINA")]:
                            if col in row and pd.notna(row[col]):
                                engine_scores.append(f"{label} {row[col]:.2f}")
                        if engine_scores:
                            bits.append("scores: " + ", ".join(engine_scores))
                        if "agreement" in row and pd.notna(row["agreement"]):
                            agree = row["agreement"]
                            agree_txt = ("engines strongly agree" if agree > 0.8 else
                                        "engines mostly agree" if agree > 0.5 else
                                        "engines disagree noticeably")
                            bits.append(f"{agree_txt} (agreement={agree:.2f})")
                        if "n_engines_scored" in row and pd.notna(row["n_engines_scored"]):
                            bits.append(f"scored by {int(row['n_engines_scored'])} engine(s)")
                        contacts = per_lig_contacts.get(lid)
                        contact_txt = ""
                        if contacts is not None and not contacts.empty:
                            top_types = contacts["interaction"].value_counts()
                            n_res = contacts["residue"].nunique()
                            key_types = ", ".join(f"{n}\u00d7{t}" for t, n in top_types.head(3).items())
                            contact_txt = (f" Makes {len(contacts)} contact(s) across {n_res} "
                                          f"pocket residues ({key_types}).")
                        st.markdown(f"**{lid}** — " + "; ".join(bits) + "." + contact_txt)
                    if len(chosen_ligs) > 1:
                        st.caption(
                            "Consensus rank is the primary tie-breaker (mean of per-engine ranks, "
                            "z-scored — lower is better). When two ligands rank closely, prefer "
                            "the one with higher `agreement` (engines concur) and more specific "
                            "contacts (H-bonds/ionic > hydrophobic/van der Waals) over the raw "
                            "score gap alone — see the Validation tab for how reliable that gap "
                            "typically is against this target's known actives/decoys.")

    except ImportError as e:
        st.error(f"3D viewer dependencies not available: {e}")

# ---------------------------------------------------------------------------
# Tab: Compare — Standard / Co-crystal / Lead-hit + any other molecules,
# overlaid in one 3D view with a side-by-side "why is this the best" verdict.
# ---------------------------------------------------------------------------
with tab_compare:
    st.header(f"Compare candidates — {run_name}")
    st.caption(
        "Put a **Standard** (known reference active), the **Co-crystal** ligand, the "
        "current **Lead hit**, and any other molecules you pick side by side: one 3D "
        "view with every pose in its own color, a scores/ADMET table, and a "
        "plain-language verdict on which one wins and why the others don't.")
    try:
        import stmol as _stmol2
        from stmol import showmol as _showmol2
        import py3Dmol as _p3d2
        import compare as _cmp
        import trust as _trust

        receptor_pdb_cmp = os.path.join(run_dir, "target", "receptor_clean.pdb")
        pose_dir_cmp = os.path.join(run_dir, "docking", "poses")

        if docking_csv is None or not os.path.exists(receptor_pdb_cmp):
            st.warning("Receptor structure or docking results not found for this run.")
        else:
            cdf = pd.read_csv(docking_csv)
            cdf["id"] = cdf["id"].astype(str)
            manifest_csv_cmp = os.path.join(run_dir, "library", "ligand_manifest.csv")
            manifest_df_cmp = pd.read_csv(manifest_csv_cmp) if os.path.exists(manifest_csv_cmp) else None

            defaults = _cmp.default_roles(cdf)
            all_ids = cdf.sort_values("final_rank" if "final_rank" in cdf.columns else cdf.columns[-1])["id"].tolist()
            none_opt = "— none —"

            rc1, rc2, rc3 = st.columns(3)
            std_id = rc1.selectbox(
                "Standard (reference active)", [none_opt] + all_ids,
                index=(all_ids.index(defaults["standard"]) + 1) if defaults["standard"] in all_ids else 0,
                help="A known/reference compound to benchmark the lead hit against — "
                     "defaults to the best-ranked labeled active (label=1) not already "
                     "used as co-crystal/lead hit, if this library has labels.")
            cox_id = rc2.selectbox(
                "Co-crystal (native ligand)", [none_opt] + all_ids,
                index=(all_ids.index(defaults["co_crystal"]) + 1) if defaults["co_crystal"] in all_ids else 0,
                help="The ligand extracted from the crystal structure — only dockable "
                     "if 'Extract & dock co-crystal ligand' was checked on the Run tab.")
            lead_id = rc3.selectbox(
                "Lead hit", [none_opt] + all_ids,
                index=(all_ids.index(defaults["lead_hit"]) + 1) if defaults["lead_hit"] in all_ids else 0,
                help="Defaults to consensus rank #1 (excluding the co-crystal ligand).")

            extra_ids = st.multiselect("+ Add any other molecule(s) to compare", all_ids, default=[])

            role_of_id = {}
            selected = []
            for val, role in [(std_id, "Standard"), (cox_id, "Co-crystal"), (lead_id, "Lead hit")]:
                if val != none_opt and val not in selected:
                    selected.append(val)
                    role_of_id[val] = role
            for val in extra_ids:
                if val not in selected:
                    selected.append(val)

            if not selected:
                st.info("Pick at least one molecule above (Standard / Co-crystal / Lead hit / other).")
            else:
                # --- comparison table -----------------------------------------
                table = _cmp.comparison_table(cdf, selected, manifest_df_cmp)
                show_table = table.copy()
                show_table.insert(1, "role", show_table["id"].map(role_of_id).fillna("—"))
                st.subheader("Comparison table")
                st.dataframe(show_table, width='stretch', hide_index=True)
                st.download_button("⬇ Download comparison (CSV)", show_table.to_csv(index=False).encode(),
                                   file_name=f"{run_name}_compare.csv", key="dl_compare_csv")

                # --- shared 3D overlay ------------------------------------------
                st.subheader("3D overlay")
                oc1, oc2, oc3 = st.columns(3)
                cmp_surface = oc1.checkbox("Receptor surface", value=False, key="cmp_surface")
                cmp_pocket = oc2.checkbox("Pocket residues (sticks + labels)", value=True, key="cmp_pocket")
                cmp_spin = oc3.checkbox("Spin", value=False, key="cmp_spin")

                color_of_id = {}
                extra_i = 0
                for mid in selected:
                    role_key = {"Standard": "standard", "Co-crystal": "co_crystal",
                               "Lead hit": "lead_hit"}.get(role_of_id.get(mid))
                    if role_key:
                        color_of_id[mid] = _cmp.ROLE_COLORS[role_key]
                    else:
                        color_of_id[mid] = _cmp.EXTRA_PALETTE[extra_i % len(_cmp.EXTRA_PALETTE)]
                        extra_i += 1

                view = _p3d2.view(width=900, height=600)
                view.addModel(open(receptor_pdb_cmp).read(), "pdb")
                view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
                if cmp_surface:
                    view.addSurface(_p3d2.VDW, {"opacity": 0.5, "color": "white"}, {"model": 0})

                all_resis_cmp = {}
                contacts_by_id = {}
                model_idx = 1
                missing_pose = []
                for mid in selected:
                    pose_pdbqt = os.path.join(pose_dir_cmp, f"{mid}_vina.pdbqt")
                    pose_sdf = os.path.join(pose_dir_cmp, f"{mid}_smina.sdf")
                    if not os.path.exists(pose_pdbqt):
                        missing_pose.append(mid)
                        continue
                    first = []
                    for ln in open(pose_pdbqt).read().splitlines():
                        first.append(ln)
                        if ln.startswith("ENDMDL"):
                            break
                    view.addModel("\n".join(first), "pdbqt")
                    view.setStyle({"model": model_idx},
                                  {"stick": {"color": color_of_id[mid], "radius": 0.22}})
                    model_idx += 1
                    if os.path.exists(pose_sdf):
                        try:
                            import interactions as _inter
                            res = _inter.analyze_pose(receptor_pdb_cmp, pose_sdf, pose_index=0)
                            if res["ok"]:
                                contacts_by_id[mid] = res["table"]
                                for r in res["table"]["residue"].tolist():
                                    base = r.split(".")[0]
                                    chain = r.split(".")[1] if "." in r else "A"
                                    resn = "".join(ch for ch in base if ch.isalpha())
                                    resi_digits = "".join(ch for ch in base if ch.isdigit())
                                    if resi_digits:
                                        all_resis_cmp[(chain, int(resi_digits))] = resn
                        except Exception:
                            pass

                if model_idx > 1:
                    view.zoomTo({"model": list(range(1, model_idx))})
                else:
                    view.zoomTo()

                if cmp_pocket and all_resis_cmp:
                    for (chain, resi), resn in all_resis_cmp.items():
                        sel = {"model": 0, "chain": chain, "resi": resi}
                        view.addStyle(sel, {"stick": {"colorscheme": "grayCarbon", "radius": 0.12}})
                        view.addLabel(f"{resn}{resi}",
                                      {"fontSize": 10, "backgroundOpacity": 0.55,
                                       "backgroundColor": "black", "fontColor": "white"}, sel)

                if cmp_spin:
                    view.spin(True)
                view.setBackgroundColor("white")
                _showmol2(view, height=600, width=900)

                legend_html = " &nbsp;&nbsp; ".join(
                    f'<span style="color:{color_of_id[mid]}; font-weight:600;">●</span> '
                    f'{mid}' + (f' <em>({role_of_id[mid]})</em>' if mid in role_of_id else '')
                    for mid in selected)
                st.markdown(legend_html, unsafe_allow_html=True)
                if missing_pose:
                    st.caption(f"No docked pose found for: {', '.join(missing_pose)} — it may not have "
                              "been part of this run's library (e.g. co-crystal docking was off), or "
                              "prep/docking failed for it (check the Prep Check tab).")

                # --- verdict ------------------------------------------------
                st.subheader("Why one ranks best, and why the others don't")
                run_trust_cmp = trust_data  # same run-level verdict shown in Validation/Hits
                n_eng_cmp = _trust.n_engines_from_columns(cdf)
                explanation = _cmp.explain_comparison(show_table.drop(columns=["role"]), role_of_id,
                                                       contacts_by_id, run_trust=run_trust_cmp,
                                                       n_engines_total=n_eng_cmp)
                for mid in selected:
                    text = explanation["per_molecule"].get(mid)
                    if text:
                        st.markdown(f"- {text}")
                st.success(explanation["verdict"])

    except ImportError as e:
        st.error(f"Compare tab dependencies not available: {e}")

# ---------------------------------------------------------------------------
# Tab 5: Report
# ---------------------------------------------------------------------------
with tab_report:
    st.header("Auto-generated report")
    report_md = os.path.join(run_dir, "report", "report.md")
    if os.path.exists(report_md):
        text = open(report_md).read()
        report_fig_dir = os.path.join(run_dir, "report")

        # A bare local filesystem path in markdown image syntax is not
        # something the browser can fetch (st.markdown renders as HTML in
        # the browser, which has no access to server-side paths) — embed
        # each referenced figure as a base64 data URI instead.
        import base64
        import re as _re

        def _embed_image(match):
            fname = match.group(1)
            fpath = os.path.join(report_fig_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                return f"](data:image/png;base64,{b64}"
            return match.group(0)

        text = _re.sub(r"\]\((fig_[\w.]+\.png)", _embed_image, text)
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.warning("No report.md found for this run yet. Launch a run or check the results folder.")

# ---------------------------------------------------------------------------
# Tab: Present — slide-by-slide judge-facing walkthrough of the selected run
# ---------------------------------------------------------------------------
with tab_present:
    def _present_target_id():
        cfg_path = os.path.join(run_dir, "run_config.yaml")
        if os.path.exists(cfg_path):
            try:
                cfg = yaml.safe_load(open(cfg_path))
                t = cfg.get("target", {})
                return t.get("pdb") or os.path.basename(t.get("infile") or run_name)
            except Exception:
                pass
        report_md_path = os.path.join(run_dir, "report", "report.md")
        if os.path.exists(report_md_path):
            first_line = open(report_md_path).readline()
            if "—" in first_line:
                return first_line.split("—")[-1].strip().lstrip("#").strip()
        return run_name

    def _present_json(*parts):
        p = os.path.join(run_dir, *parts)
        return json.load(open(p)) if os.path.exists(p) else None

    def _slide_title():
        st.markdown("<div style='text-align:center; padding-top:40px;'>", unsafe_allow_html=True)
        st.title(f"Consensus Docking — {_present_target_id()}")
        st.markdown("##### 100% open-source pipeline — AutoDock Vina + Smina consensus, "
                    "with automatic self-validation")
        if trust_data:
            st.markdown(f"### {trust_data['emoji']} {trust_data['status'].replace('_', ' ').title()}")
        st.caption(f"Run: {run_name}")
        st.markdown("</div>", unsafe_allow_html=True)

    def _slide_trust():
        st.markdown("## Can this run's ranking be trusted?")
        if not trust_data:
            st.info("No trust verdict recorded for this run (older run, or trust layer didn't execute).")
            return
        banner = {"validated": st.success, "partially_validated": st.warning,
                 "failed": st.error, "unvalidated": st.info}.get(trust_data["status"], st.info)
        banner(f"{trust_data['emoji']}  **{trust_data['status'].replace('_', ' ').title()}**")
        st.markdown(f"##### {trust_data['headline']}")
        st.markdown("**Why:**")
        for r in trust_data["reasons"]:
            st.markdown(f"- {r}")

    def _slide_summary():
        st.markdown("## Protocol validation, at a glance")
        redock = _present_json("validation", "redock_rmsd.json")
        metrics = _present_json("validation", "metrics.json")
        c1, c2, c3, c4 = st.columns(4)
        if redock and redock.get("rmsd_top_pose") is not None:
            c1.metric("Redock RMSD (top pose)", f"{redock['rmsd_top_pose']:.2f} Å",
                      help="Native ligand re-docked blind; <2 Å means the search & scoring "
                           "reproduce the known binding mode.")
        else:
            c1.metric("Redock RMSD", "n/a")
        if metrics:
            c2.metric("AUC-ROC", f"{metrics.get('auc_roc', float('nan')):.3f}",
                      help="Probability a random active outranks a random decoy. 0.5=random, 1.0=perfect.")
            c3.metric("EF @ 1%", f"{metrics.get('EF_1pct', float('nan')):.2f}×",
                      help="Fold-enrichment for true actives in the top 1% of the ranking.")
            c4.metric("BEDROC (α=20)", f"{metrics.get('BEDROC_a20', float('nan')):.3f}",
                      help="Early-recognition score — weights actives found near the top of the list.")
        else:
            c2.metric("AUC-ROC", "n/a")
        st.caption("Every number on this slide is produced by this run itself — nothing here is "
                  "entered by hand.")

    def _slide_box():
        st.markdown("## Search space")
        box = _present_json("target", "box.json")
        if not box:
            st.info("No docking box recorded for this run.")
            return
        st.markdown(f"**Method:** `{box.get('method', 'unknown')}`")
        c1, c2 = st.columns(2)
        c1.metric("Center (Å)", str([round(x, 2) for x in box.get("center", [])]))
        c2.metric("Size (Å)", str([round(x, 2) for x in box.get("size", [])]))
        if box.get("method") in ("p2rank", "fpocket") and box.get("all_pockets"):
            st.caption(f"{len(box['all_pockets'])} candidate pocket(s) were found; the "
                      "highest-confidence one was used for the box above.")

    def _slide_hits():
        st.markdown("## Top consensus hits")
        if docking_csv is None:
            st.info("No docking results for this run yet.")
            return
        df = pd.read_csv(docking_csv)
        sort_col = "consensus_z" if "consensus_z" in df.columns else df.columns[-1]
        top = df.sort_values(sort_col).head(8)
        cols = [c for c in ["final_rank", "id", "vina_score", "smina_score", "gnina_score",
                            "consensus_z", "agreement", "confidence_tier"] if c in top.columns]
        st.dataframe(top[cols], hide_index=True, width='stretch')
        st.caption("Ranked by consensus_z (lower = better). confidence_tier reflects both "
                  "cross-engine agreement and this run's own validation status.")

    def _slide_spotlight():
        st.markdown("## Lead hit spotlight")
        if docking_csv is None:
            st.info("No docking results for this run yet.")
            return
        df = pd.read_csv(docking_csv)
        sort_col = "consensus_z" if "consensus_z" in df.columns else df.columns[-1]
        top_row = df.sort_values(sort_col).iloc[0]
        lid = str(top_row["id"])
        st.markdown(f"### {lid}  ·  consensus rank #{int(top_row.get('final_rank', 1))}")
        if "confidence_tier" in top_row and "confidence_reason" in top_row:
            tier_color = {"High": st.success, "Medium": st.warning, "Low": st.error}.get(
                top_row["confidence_tier"], st.info)
            tier_color(f"**{top_row['confidence_tier']} confidence** — {top_row['confidence_reason']}")
        pose_dir = os.path.join(run_dir, "docking", "poses")
        receptor_pdb = os.path.join(run_dir, "target", "receptor_clean.pdb")
        pose_pdbqt = os.path.join(pose_dir, f"{lid}_vina.pdbqt")
        if os.path.exists(receptor_pdb) and os.path.exists(pose_pdbqt):
            try:
                import py3Dmol as _p3d3
                from stmol import showmol as _showmol3
                view = _p3d3.view(width=800, height=450)
                view.addModel(open(receptor_pdb).read(), "pdb")
                view.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
                first = []
                for ln in open(pose_pdbqt).read().splitlines():
                    if ln.startswith("ENDMDL"):
                        break
                    first.append(ln)
                view.addModel("\n".join(first), "pdbqt")
                view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
                view.zoomTo({"model": 1})
                _showmol3(view, height=450, width=800)
            except Exception as e:
                st.caption(f"3D view unavailable: {e}")

    def _slide_figures():
        st.markdown("## Validation figures")
        figdir = os.path.join(run_dir, "report")
        shown = False
        for fname, label in [("fig_roc.png", "ROC curves"), ("fig_enrichment.png", "Enrichment")]:
            fpath = os.path.join(figdir, fname)
            if os.path.exists(fpath):
                st.markdown(f"**{label}**")
                st.image(fpath, width='stretch')
                shown = True
        if not shown:
            st.info("No labeled benchmark in this run's library, so no ROC/enrichment figures "
                    "were generated (expected for a prospective/blind screen).")

    def _slide_closing():
        st.markdown("<div style='text-align:center; padding-top:60px;'>", unsafe_allow_html=True)
        st.markdown("## Every other tool gives you a docking score.")
        st.markdown("## We give you a docking score you can defend.")
        st.caption("Validated · Confidence-scored · Explained — automatically, every run.")
        st.markdown("</div>", unsafe_allow_html=True)

    _slides = [
        ("Title", _slide_title),
        ("Trust Verdict", _slide_trust),
        ("Validation Summary", _slide_summary),
        ("Search Space", _slide_box),
        ("Top Hits", _slide_hits),
        ("Lead Hit Spotlight", _slide_spotlight),
        ("Figures", _slide_figures),
        ("Closing", _slide_closing),
    ]

    st.session_state.setdefault("present_idx", 0)
    st.session_state["present_idx"] = min(st.session_state["present_idx"], len(_slides) - 1)

    def _goto_slide(new_idx):
        # Streamlit keeps a keyed widget's own session_state value across
        # reruns and ignores a newly-passed `index=` once that key already
        # has a stored value -- so the "Jump to slide" selectbox must be
        # updated here too, not just present_idx, or it silently goes stale
        # after Previous/Next (confirmed: it kept showing "Title" on slide 2).
        st.session_state["present_idx"] = new_idx
        st.session_state["present_jump"] = _slides[new_idx][0]

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    if nav1.button("← Previous", key="present_prev", disabled=(st.session_state["present_idx"] == 0)):
        _goto_slide(st.session_state["present_idx"] - 1)
        st.rerun()
    nav2.markdown(
        f"<div style='text-align:center'>Slide {st.session_state['present_idx'] + 1} / {len(_slides)} "
        f"&nbsp;·&nbsp; {_slides[st.session_state['present_idx']][0]}</div>",
        unsafe_allow_html=True)
    if nav3.button("Next →", key="present_next",
                   disabled=(st.session_state["present_idx"] == len(_slides) - 1)):
        _goto_slide(st.session_state["present_idx"] + 1)
        st.rerun()

    st.session_state.setdefault("present_jump", _slides[st.session_state["present_idx"]][0])
    jump = st.selectbox("Jump to slide", [s[0] for s in _slides],
                        key="present_jump", label_visibility="collapsed")
    jump_idx = [s[0] for s in _slides].index(jump)
    if jump_idx != st.session_state["present_idx"]:
        st.session_state["present_idx"] = jump_idx
        st.rerun()

    st.divider()
    _slides[st.session_state["present_idx"]][1]()

# ---------------------------------------------------------------------------
# Tab: Downloads — structures, figures, tables + full-run ZIP
# ---------------------------------------------------------------------------
with tab_dl:
    st.header(f"Downloads — {run_name}")
    st.caption("Grab individual files by category, or the whole run as a single ZIP.")

    import io
    import zipfile
    import mimetypes

    def _collect(run_dir):
        """Return {category: [(label, abspath)]} for downloadable outputs."""
        cats = {"Structures": [], "Figures": [], "Tables": [], "Reports": []}
        target = os.path.join(run_dir, "target")
        if os.path.isdir(target):
            for f in sorted(os.listdir(target)):
                if f.endswith((".pdb", ".pdbqt")):
                    cats["Structures"].append((f, os.path.join(target, f)))
        pose_dir = os.path.join(run_dir, "docking", "poses")
        if os.path.isdir(pose_dir):
            for f in sorted(os.listdir(pose_dir)):
                if f.endswith((".pdbqt", ".sdf")):
                    cats["Structures"].append((os.path.join("poses", f), os.path.join(pose_dir, f)))
        rep = os.path.join(run_dir, "report")
        if os.path.isdir(rep):
            for f in sorted(os.listdir(rep)):
                if f.endswith(".png"):
                    cats["Figures"].append((f, os.path.join(rep, f)))
                elif f.endswith(".md"):
                    cats["Reports"].append((f, os.path.join(rep, f)))
        # tables: any CSV/JSON under the run
        for root, _dirs, files in os.walk(run_dir):
            for f in files:
                if f.endswith((".csv", ".json")):
                    rel = os.path.relpath(os.path.join(root, f), run_dir)
                    cats["Tables"].append((rel, os.path.join(root, f)))
        return cats

    cats = _collect(run_dir)

    # full-run ZIP
    def _build_zip(run_dir):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(run_dir):
                for f in files:
                    ap = os.path.join(root, f)
                    zf.write(ap, os.path.join(run_name, os.path.relpath(ap, run_dir)))
        buf.seek(0)
        return buf.getvalue()

    st.subheader("Full run")
    with st.spinner("Packing ZIP…"):
        zip_bytes = _build_zip(run_dir)
    st.download_button(f"\u2B07 Download entire run ({run_name}.zip)", zip_bytes,
                       file_name=f"{run_name}.zip", mime="application/zip",
                       key="dl_full_zip")
    st.caption(f"{len(zip_bytes)/1024:.0f} KB compressed")

    st.divider()
    for cat, items in cats.items():
        if not items:
            continue
        st.subheader(cat)
        # lay out in up to 3 columns of download buttons
        ncol = 3
        cols = st.columns(ncol)
        for i, (label, path) in enumerate(items):
            with cols[i % ncol]:
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    st.download_button(label, data, file_name=os.path.basename(path),
                                       mime=mime, key=f"dl_{cat}_{i}")
                except Exception as e:  # noqa: BLE001
                    st.caption(f"{label}: unavailable ({e})")
    if not any(cats.values()):
        st.info("No downloadable outputs found for this run yet.")

# ---------------------------------------------------------------------------
# Tab 6: Tech Stack
# ---------------------------------------------------------------------------
with tab_tech:
    st.markdown("**PrismDock — P2Rank-Guided Rank-Integrated Smina/Vina Molecular Docking & Validation Suite** is powered by a variety of cutting-edge open-source tools and libraries.")
    st.markdown("*Author: Sridhar J D, M.Pharm (Pharmaceutical Chemistry) | GitHub: [github.com/jdsridhar/prismdock](https://github.com/jdsridhar/prismdock) | ORCID: 0009-0009-5196-7100*")
    
    st.subheader("Core Pipeline Engines")
    st.markdown("""
    - **[AutoDock Vina](https://vina.scripps.edu/)**: The primary empirical scoring function and docking engine. It predicts the binding affinity and binding pose of ligands.
    - **[Smina](https://github.com/mwojcikowski/smina)**: A fork of Vina that provides custom scoring functions (we use **Vinardo**) and robust CLI tools. It acts as the second consensus engine, ensuring robustness against single-engine biases.
    - **[GNINA](https://github.com/gnina/gnina)** (Optional): A molecular docking program with integrated deep learning scoring functions. Used for a third consensus vote via CNN pose and affinity scoring.
    - **[DiffDock](https://github.com/gcorso/DiffDock)** (Optional): A diffusion model for blind docking. Used purely for sanity-checking the auto-detected binding pocket.
    - **[P2Rank](https://github.com/rdk/p2rank)**: Machine-learning cavity predictor (primary) for blind pocket detection when no reference ligand is provided — preferred over fpocket because published benchmarks show better top-1 pocket accuracy. Every candidate pocket it finds is recorded, not just the winner.
    - **[fpocket](https://github.com/fpocket/fpocket)**: Geometry-based cavity detection — automatic fallback if P2Rank/Java isn't available on this machine.
    - **[Meeko](https://github.com/forli-lab/Meeko)**: Primary tool for preparing ligands and the receptor, converting molecules to the PDBQT format required by Vina, using AutoDock-native atom typing.
    - **[OpenBabel](https://openbabel.org/)**: Automatic fallback PDBQT converter for any molecule Meeko fails to prepare, so a single edge case can't silently drop a ligand from the run.
    - **[RDKit](https://www.rdkit.org/)**: The backbone for cheminformatics. Responsible for reading SMILES, standardization, 3D conformer embedding (ETKDGv3), and MMFF optimization.
    - **[PDB2PQR](https://github.com/Electrostatics/pdb2pqr)** / **[PROPKA](https://github.com/jensengroup/propka)**: Used to predict pKa values and properly protonate the receptor target at the desired pH (e.g., physiological pH 7.4).
    """)

    st.subheader("Analysis & Visualization")
    st.markdown("""
    - **[ProLIF](https://github.com/chemosim-lab/ProLIF)** (Protein-Ligand Interaction Fingerprints): Analyzes docking poses to generate 2D interaction network diagrams (identifying H-bonds, salt bridges, pi-stacking, etc.).
    - **[py3Dmol](https://3dmol.csb.pitt.edu/)** & **[stmol](https://github.com/napoles-uach/stmol)**: Provide the advanced, interactive 3D WebGL viewer for visualizing docked poses directly in the browser.
    - **[Plotly](https://plotly.com/python/)**: Powers the interactive validation plots (ROC curves, enrichment metrics, and score distributions).
    - **[Scikit-learn](https://scikit-learn.org/)**: Used for calculating validation metrics like ROC-AUC when known actives and decoys are provided.
    """)

    st.subheader("Application Framework")
    st.markdown("""
    - **[Streamlit](https://streamlit.io/)**: The frontend framework that glues the pipeline together into an accessible web dashboard, allowing event-day operation without dropping to the command line.
    - **[Pandas](https://pandas.pydata.org/)**: Core data structure handling. Manages the ligand manifest, parses engine scores, computes the consensus ranking (z-scores and agreement), and exports results.
    """)
