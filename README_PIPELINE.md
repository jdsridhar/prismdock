# Consensus Docking Pipeline — Dockathon

A **100% open-source**, reproducible molecular-docking pipeline built to be
*pointed at an unknown target on event day*. You edit one config file, run one
command, and get ranked hits **plus an automatic self-validation** that proves
the protocol works.

## The novelty (what to tell the judges)

Most teams run a single engine (Vina) on a single structure and show a picture.
This pipeline is different on three axes that judges actually score:

1. **Consensus over two independent scoring functions** (AutoDock Vina +
   Smina/Vinardo). A hit is credible when *independent* engines agree — we rank
   by cross-engine consensus, not one score we happen to trust. Optional GPU
   third opinion (GNINA CNN / DiffDock) folds in automatically when available.

2. **Built-in self-validation** — the part almost nobody does:
   - **Pose accuracy:** redock the native crystal ligand and report **RMSD to
     the crystal pose** (<2 Å = the protocol reproduces known geometry).
   - **Screening power:** seed the library with known actives + decoys and
     report **AUC-ROC, enrichment factor (EF@1/5/10%), and BEDROC**. This
     *proves* the score separates binders from non-binders instead of asserting
     it.

3. **Automation + reproducibility** — target in → validated report out, one
   command, versions pinned, no paid software anywhere in the stack.

## Install

```bash
conda create -n docking -c conda-forge -c bioconda python=3.11 \
    rdkit openbabel vina smina fpocket pdbfixer openmm \
    scikit-learn matplotlib numpy pandas scipy requests pyyaml
conda activate docking
pip install meeko gemmi
```

## Run

```bash
# edit config.yaml -> set target.pdb (the assigned target) + library.csv
python run_dockathon.py --config config.yaml
```

Outputs (under `results/run/`):
- `docking/consensus_results.csv` — ranked hits with per-engine + consensus scores
- `validation/redock_rmsd.json`, `validation/metrics.json`
- `report/report.md` + figures (ROC, enrichment, score separation, agreement)

## Event-day checklist

1. `conda activate docking`
2. Edit `config.yaml`: set `target.pdb` to the assigned PDB ID (or `target.infile`
   for an uploaded / AlphaFold structure). Point `library.csv` at your screening set.
3. If the target is **apo** (no bound ligand), pocket detection falls back to
   fpocket automatically — sanity-check the detected box, or set `force_center`.
4. If a GPU is on the machine, set `rescoring.gnina_cnn: true` for a third opinion.
5. `python run_dockathon.py --config config.yaml`
6. Show `report/report.md`.

## Module map (`src/`)

| File | Role |
|------|------|
| `prepare_target.py` | fetch + clean + protonate receptor, extract native ligand, PDBQT |
| `detect_pocket.py`  | ligand-guided box, else P2Rank cavity prediction (fpocket fallback) |
| `prepare_ligands.py`| SMILES → standardize → 3D → PDBQT (Meeko, OpenBabel fallback) |
| `dock_consensus.py` | Vina + Smina docking + consensus rank/agreement |
| `validate.py`       | redock RMSD + AUC/EF/BEDROC |
| `rescore_gpu.py`    | optional GNINA CNN / DiffDock hooks (GPU) |
| `compare.py`        | Standard/Co-crystal/Lead-hit comparison table + narrative (Compare tab) |
| `report.py`         | figures + Markdown report |
| `run_dockathon.py`  | one-command driver over `config.yaml` |

All components are open-source (Vina Apache-2.0, Smina Apache/GPL, RDKit BSD,
fpocket MIT, P2Rank Apache-2.0, OpenBabel GPL, Meeko LGPL, PDBFixer/OpenMM
MIT). See `environment.md` for pinned versions.

## Why Meeko (not OpenBabel) for ligand/receptor PDBQT prep

**Meeko is primary; OpenBabel is an automatic fallback if Meeko fails on a
given molecule.** Meeko is built and maintained by the Forli Lab — the same
group behind AutoDock/Vina itself — so it assigns AutoDock atom types and
builds torsion trees the way Vina/Smina/GNINA actually expect, and correctly
handles macrocycles and flexible rings that older tools mishandle. It's the
modern recommended replacement for the legacy AutoDockTools/OpenBabel prep
workflow. OpenBabel remains valuable as a fallback: it's a more permissive,
general-purpose cheminformatics converter that succeeds on some inputs Meeko
rejects, so one molecule's edge case can no longer silently drop it from an
entire screening run (this mirrors the pattern `prepare_target.py` already
used for the receptor). Each run's `ligand_manifest.csv` records a
`prep_tool` column (`meeko` or `obabel`) per ligand, and the generated
report states how many ligands needed the fallback.

## Why P2Rank (not fpocket) for apo pocket detection

**P2Rank is primary; fpocket is an automatic fallback if P2Rank/Java isn't
installed.** For a target with a co-crystallised ligand, the box is always
centred on that ligand directly (`ligand-guided`) — no predictor is
involved, and this remains the most reliable option. For an **apo** target
with nothing bound, the pocket must be predicted, and here P2Rank
(a machine-learning model — random forest over solvent-accessible-surface
points, trained on known binding sites) is preferred over fpocket's
pure-geometry alpha-sphere clustering: published benchmarks (Krivák &
Hoksza, *J Cheminform* 2018, and subsequent P2Rank papers) show it
consistently beats fpocket on top-1/top-3 pocket identification accuracy,
which matters most exactly when there's no ligand to hint at the site.
fpocket is kept as the automatic fallback (it needs no Java runtime and
ships via conda), so a missing optional dependency never blocks a run.
Both apo methods now record **every** ranked candidate pocket, not just the
winner, in `box.json` → `all_pockets` — useful when the top-ranked cavity
turns out to be a crystallographic artifact or the real site is cryptic.
P2Rank isn't on conda; `setup_env.sh` downloads it straight from its GitHub
release into `vendor/p2rank/` (needs a JRE — already present on most
machines; skipped automatically if Java is missing).
