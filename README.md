# PrismDock — Consensus Molecular Docking Dashboard

A Streamlit front-end over the open-source consensus docking pipeline
(AutoDock Vina + Smina, with automatic self-validation). See
[README_PIPELINE.md](README_PIPELINE.md) for the pipeline internals,
[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the comprehensive architecture summary, and
[DEMO_SCRIPT.md](DEMO_SCRIPT.md) for the presentation pitch.

## First-time setup (once per machine)

You need conda. If you don't have it yet, this one script installs Miniconda
and creates the environment for you:

```bash
cd prismdock
./setup_env.sh
```

This installs Miniconda to `~/miniconda3` (skipped if conda is already present
under `~/miniconda3`, `~/anaconda3`, `~/miniforge3`, or `/opt/conda`) and then
creates the `docking` conda environment from `environment.yml` — the same
package set (RDKit, AutoDock Vina, Smina, OpenBabel, Meeko, fpocket,
scikit-learn, Streamlit, Plotly, py3Dmol, ProLIF, ...) used to build and
validate this pipeline. It also downloads P2Rank (apo pocket prediction,
preferred over fpocket — see [README_PIPELINE.md](README_PIPELINE.md)) into
`vendor/p2rank/`, skipped automatically if Java isn't found (fpocket is used
instead). Takes a few minutes.

## Every time after: launch the dashboard

```bash
cd prismdock
./launch_app.sh
```

Opens on http://localhost:8501 — nine tabs:

1. **Run** — configure + launch a docking job (fetch a PDB by ID or upload a structure, pick a library, set docking parameters).
2. **Prep Check** — docking-readiness QC. Traffic-light pass/warn/fail for the receptor and every ligand (charges parsed, hydrogens added, torsion tree valid), with a downloadable QC table.
3. **Hits** — sortable, filterable consensus ranking table with CSV download.
4. **Interactions** — 2D protein–ligand interaction diagram (open-source LigPlot+ analogue via ProLIF): H-bonds, hydrophobic, pi-stacking, salt-bridge and van-der-Waals contacts, plus a tidy contact table (CSV download).
5. **Validation** — interactive ROC / enrichment / score-separation / cross-engine-agreement Plotly plots, redock RMSD, and AUC / EF / BEDROC metrics.
6. **Pose Viewer** — advanced interactive 3D viewer: receptor cartoon/surface, pocket residues as labelled sticks, contact-residue highlights, multi-pose overlay, and spin.
7. **Compare** — pick a Standard (reference active), the Co-crystal ligand, the current Lead hit, and any other molecules; see every pose overlaid in one 3D view in distinct colors, a scores/ADMET comparison table, and an auto-generated verdict on which one wins and why the others don't.
8. **Report** — auto-generated Markdown writeup with embedded figures.
9. **Downloads** — per-category download buttons (structures, figures, tables, report) and a single full-run ZIP.

## Manual setup (if you prefer not to run the scripts)

```bash
conda env create -f environment.yml
conda activate docking
streamlit run app.py --server.port 8501
```

## Event-day checklist

1. `./launch_app.sh`
2. In the **Run** tab: enter the assigned PDB ID (or upload a structure),
   choose a screening-library source, set exhaustiveness (8+ recommended
   for the real run), and launch.
3. Once it finishes, pick the new run from the sidebar to see hits, validation
   plots, poses, and the report.

### Nucleus (scaffold) mode — for scaffold-based design challenges

When the committee hands you a fixed **core scaffold** ("nucleus") and every
candidate must keep it, pick **Nucleus (scaffold) → enumerate** as the library
source in the Run tab:

1. Paste the core as SMILES (or draw it in the Ketcher tile and paste the SMILES).
   - Mark exact attachment points with `[*]` — e.g. `O=C(N)c1ccc([*])cc1` — and each
     `[*]` is combinatorially replaced by every chosen R-group.
   - With **no** `[*]`, every aromatic C–H is a candidate site (mono-substitution;
     tick *Add di-substitution* for pairs).
2. Pick the R-groups to attach (a curated 15-member medchem set is selected by
   default; ~30 are available) and a max-analog cap.
3. The app enumerates the analog library, shows a 2D grid preview + a table with
   MW / logP / HBD / HBA / TPSA / Lipinski pass, and writes it to
   `data/enumerated_scaffold.csv`. Every analog is guaranteed to contain the
   nucleus as a substructure (the "keep the core" rule is enforced).
4. Launch — the enumerated library feeds the same pocket-detection → consensus
   docking → validation → report pipeline as any other library.

The enumeration engine is `src/enumerate_scaffold.py` (RDKit only) and can also
be run headless:

```bash
python src/enumerate_scaffold.py --nucleus "O=C(N)c1ccc([*])cc1" \
    --out data/my_scaffold.csv --max 200 --grid data/my_scaffold_grid.png
```

## Optional GPU rescoring (third consensus member)

The Vina + Smina consensus is the validated core and needs no GPU — it's what
every result above is built on. If you're running the dashboard on a machine
with a CUDA GPU (e.g. a laptop with an RTX 3050+), the **Run** tab has an
"⚡ Optional GPU rescoring" expander with two checkboxes that fold in a third,
independent opinion. Both are safe to leave unchecked; if checked but the
binary/GPU isn't available, the pipeline logs a skip message and continues
with the CPU consensus only — it never fails the run.

### GNINA (CNN pose/affinity rescoring) — a genuine third consensus engine

GNINA re-scores the poses Vina already found with a convolutional neural
network — it does not re-search, just re-scores. This is not a cosmetic
add-on: published benchmarks show GNINA's CNN scoring out-ranks Vina's
empirical scoring on virtual-screening enrichment (AUC/EF/BEDROC — the same
axis this pipeline already validates against) on most DUD-E/LIT-PCBA
targets tested. Open-source (Apache-2.0).

```bash
conda install -c gnina gnina         # installs the gnina CLI binary
# or build from source: https://github.com/gnina/gnina
gnina --version                       # confirm it's on PATH
```

Once installed, check "GNINA CNN rescoring" in the Run tab (or set
`rescoring.gnina_cnn: true` in a config YAML). The pipeline calls
`gnina --score_only --cnn_scoring rescore` on each existing Vina pose,
converts the result to a `gnina_score` column (GNINA's CNNaffinity negated,
so it sorts lower-is-better like `vina_score`/`smina_score`), then **merges
it straight into `consensus_results.csv` and re-ranks with all three
engines** via `dock_consensus.consensus_rank(..., engines=["vina","smina",
"gnina"])` — GNINA becomes a real consensus vote, not a side file. The
pre-GNINA (Vina+Smina only) ranking is preserved alongside it as
`consensus_results_2engine.csv` so you can show how the ranking shifted.
The Hits/Validation tabs and the generated report automatically detect the
`gnina_score` column and switch to reporting 3-engine AUC/agreement.
Implementation in [`src/rescore_gpu.py`](src/rescore_gpu.py)
(`gnina_rescore`, `run`) and [`src/dock_consensus.py`](src/dock_consensus.py)
(`consensus_rank`, now generalized to N engines).

### DiffDock-L (blind pose prediction) — a pocket sanity check, not a ranking input

DiffDock-L samples ligand placements over the whole protein surface with a
diffusion model, with no binding-site box. That is exactly what makes it
useful here as a **sanity check** rather than a ranking engine: since this
pipeline already knows the pocket (from the co-crystallised ligand or
fpocket), DiffDock's headline advantage over box-based docking — finding
the site blind — doesn't apply. A fair benchmark comparison at the 2 Å RMSD
threshold found Vina/GNINA's top-1 pose-accuracy success rates beat
DiffDock's by roughly 20-25 percentage points when the pocket is already
known, so DiffDock's poses are written for inspection only and never folded
into `consensus_z`. Its practical value is flagging whether its independent,
box-free guess agrees with the box the rest of the pipeline assumed — a
mismatch is worth a human look (e.g. a cryptic or alternate site). Open-source
(MIT).

```bash
git clone https://github.com/gcorso/DiffDock.git
# follow the repo's README to create its conda env + download model weights
export DIFFDOCK_REPO=/path/to/DiffDock
```

Once installed and `DIFFDOCK_REPO` is set, check "DiffDock-L blind poses" in
the Run tab (or set `rescoring.diffdock: true`). The pipeline builds a
per-ligand batch CSV from the run's manifest and calls DiffDock's
`inference.py` against the prepared receptor, writing
`{run}/docking/diffdock/<ligand_id>/rank1.sdf` per ligand plus a
`diffdock_summary.csv` that includes a `diffdock_in_box` flag — whether
DiffDock's own top blind pose centroid falls inside the docking box (+2 Å
slack) — implementation in [`src/rescore_gpu.py`](src/rescore_gpu.py)
(`diffdock_poses`, `_pose_center_in_box`). The first inference call is
silent for several minutes while DiffDock precomputes lookup tables — this
is normal, not a hang.

Neither dependency is in `environment.yml` — both need a CUDA build and a
GPU driver, so they're kept out of the shared conda env and installed
separately only on a GPU-equipped machine.
