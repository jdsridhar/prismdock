# Dockathon — Demo & Presentation Script

A tight narrative for the judges. The core message: **we don't just dock, we
prove our docking is trustworthy — automatically, with 100% open-source tools.**

---

## 30-second pitch (open with this)

> "Anyone can run one docking engine and show a binding pose. The problem is a
> single score is unreliable. Our pipeline does two things differently: it takes
> a **consensus of two independent engines** so a hit only ranks high when they
> agree, and it **validates itself** — it redocks the crystal ligand to prove
> pose accuracy, and it screens known actives against decoys to prove it can
> tell binders from non-binders. Target in, validated report out, one command,
> zero paid software."

---

## The 4 things to show (in order)

1. **Reproducibility / automation** — show `config.yaml` and the single command.
   *"On this morning's target we changed one line and re-ran."* Judges love that
   you were producing results while others were installing software.

2. **Pose validation (redock RMSD)** — show the redock number.
   *"We redocked the native crystal ligand: RMSD X.XX Å to the crystal pose.
   Under 2 Å means our box + protocol reproduce a known answer — so when we dock
   unknowns, the geometry is trustworthy."*

3. **Screening validation (enrichment)** — show the ROC + enrichment plot.
   *"We seeded the library with known CDK2 actives and property-matched decoys.
   AUC-ROC = 0.XX, and we recover Y% of actives in the top 5% screened —
   enrichment factor Z×. This is retrospective proof the score works, not a
   claim."*

4. **Consensus advantage** — show the ROC with all three curves.
   *"Notice the consensus curve sits above either single engine. Cross-engine
   agreement is our credibility filter — the scatter shows actives cluster where
   both engines score strongly."*

---

## Anticipated judge questions + answers

- **"Why not just use Glide/GOLD?"** → Paid + closed. Our stack is 100%
  open-source and fully reproducible — anyone can rerun it, which is the point
  of a benchmark.
- **"How do you know your hits are real?"** → We don't assert it, we measured it:
  redock RMSD proves pose accuracy, enrichment proves ranking power. Both are in
  the report, computed automatically.
- **"What if the target has no known ligand / is apo?"** → Pocket detection falls
  back to fpocket cavity detection; we sanity-check the box. Prospective mode
  skips enrichment (no labels) but still reports consensus + redock-style checks
  where a reference exists.
- **"Isn't consensus just averaging?"** → It's rank aggregation + an *agreement*
  score. A ligand both engines rank top gets high confidence; disagreement is
  flagged, not hidden.
- **"Could you add ML scoring?"** → Yes — GNINA CNN and DiffDock hooks are wired
  behind a flag; on a GPU machine they fold in as a third consensus member.

---

## Event-day runbook (do this live)

```bash
conda activate docking
# edit config.yaml: target.pdb = <assigned PDB>, library.csv = <screening set>
python run_dockathon.py --config config.yaml
# open results/run/report/report.md and present the 4 figures
```

If the assigned target is apo or the box looks wrong, set `pocket.force_center`
in the config and re-run — takes seconds to re-detect.

---

## What makes this "novel" (the scoring rubric translation)

| Judge criterion | Our answer |
|---|---|
| Scientific rigor | redock RMSD + AUC/EF/BEDROC self-validation |
| Method novelty | multi-engine consensus + agreement scoring |
| Reproducibility | one config, one command, pinned open-source versions |
| Engineering | automated target→report, resumable, GPU-optional |
| Communication | auto-generated report with quantitative figures |
