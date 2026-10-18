"""
report.py — Figures + auto-generated report for the docking run.

Produces publication-quality figures:
  * ROC curves (consensus vs each single engine) — the "consensus wins" story.
  * Enrichment plot (actives recovered vs % of library screened).
  * Score-distribution (actives vs decoys) for the consensus score.
  * Vina-vs-Smina agreement scatter (cross-engine correlation).
  * Redock RMSD panel (pose overlay is rendered separately if PyMOL/py3Dmol).

Then assembles a Markdown/HTML report tying figures + metrics + tool versions
together into one reproducible deliverable.

Open-source: matplotlib, numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "figure.autolayout": True,
    })


def plot_roc(df, outpath):
    """ROC for consensus + each single engine (2- or 3-engine, GNINA optional)."""
    _style()
    labels = df["label"].astype(int).values
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    engine_cols = [("Vina", "vina_score", "#e07b39"), ("Smina", "smina_score", "#3a9c6e"),
                   ("GNINA", "gnina_score", "#9c3a8e")]
    present = [e for e, col, _ in engine_cols if col in df and df[col].notna().any()]
    consensus_label = f"Consensus ({'+'.join(present)})" if present else "Consensus"
    curves = [(consensus_label, -df["consensus_z"].values, "#1b4f9c", 2.4)]
    for eng, col, c in engine_cols:
        if col in df and df[col].notna().any():
            sub = df.dropna(subset=[col])
            curves.append((eng, -sub[col].values, c, 1.6, sub["label"].astype(int).values))
    for item in curves:
        if len(item) == 5:
            name, score, color, lw, lab = item
        else:
            name, score, color, lw = item
            lab = labels
        fpr, tpr, _ = roc_curve(lab, score)
        auc = roc_auc_score(lab, score)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random (0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Retrospective ROC — actives vs decoys")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.savefig(outpath); plt.close(fig)
    return outpath


def plot_enrichment(df, outpath):
    """% actives recovered vs % library screened (early recognition)."""
    _style()
    d = df.sort_values("consensus_z").reset_index(drop=True)
    labels = d["label"].astype(int).values
    n = len(labels); n_act = labels.sum()
    frac_screened = np.arange(1, n + 1) / n
    frac_found = np.cumsum(labels) / n_act
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(frac_screened * 100, frac_found * 100, color="#1b4f9c", lw=2.4, label="Consensus")
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=1, label="Random")
    # ideal
    ideal_x = [0, n_act / n * 100, 100]; ideal_y = [0, 100, 100]
    ax.plot(ideal_x, ideal_y, ":", color="black", lw=1, label="Ideal")
    ax.set_xlabel("% of library screened"); ax.set_ylabel("% of actives recovered")
    ax.set_title("Enrichment (early recognition)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    fig.savefig(outpath); plt.close(fig)
    return outpath


def plot_score_dist(df, outpath):
    """Consensus score distribution: actives vs decoys."""
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for lab, name, c in [(1, "Actives", "#c0392b"), (0, "Decoys", "#7f8c8d")]:
        vals = df.loc[df["label"] == lab, "consensus_z"]
        ax.hist(vals, bins=20, alpha=0.6, label=f"{name} (n={len(vals)})",
                color=c, density=True)
    ax.set_xlabel("Consensus score (lower = better)"); ax.set_ylabel("Density")
    ax.set_title("Score separation: actives vs decoys")
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(outpath); plt.close(fig)
    return outpath


def plot_agreement(df, outpath):
    """Vina vs Smina score scatter, colored by label."""
    _style()
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for lab, name, c in [(0, "Decoys", "#7f8c8d"), (1, "Actives", "#c0392b")]:
        sub = df[df["label"] == lab]
        ax.scatter(sub["vina_score"], sub["smina_score"], s=26, alpha=0.7,
                   label=name, color=c, edgecolor="white", linewidth=0.4)
    r = np.corrcoef(df["vina_score"], df["smina_score"])[0, 1]
    ax.set_xlabel("Vina affinity (kcal/mol)"); ax.set_ylabel("Smina affinity (kcal/mol)")
    ax.set_title(f"Cross-engine agreement (r = {r:.2f})")
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(outpath); plt.close(fig)
    return outpath


def make_figures(results_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(results_csv).dropna(subset=["label", "consensus_z"])
    df["label"] = df["label"].astype(int)
    figs = {
        "roc": plot_roc(df, os.path.join(outdir, "fig_roc.png")),
        "enrichment": plot_enrichment(df, os.path.join(outdir, "fig_enrichment.png")),
        "score_dist": plot_score_dist(df, os.path.join(outdir, "fig_score_dist.png")),
        "agreement": plot_agreement(df, os.path.join(outdir, "fig_agreement.png")),
    }
    return figs


# --------------------------------------------------------------------------
# Interpretation helpers — turn raw numbers into plain-language verdicts so the
# report explains what each result *means*, not just what it is.
# --------------------------------------------------------------------------
def _fnum(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _interpret_rmsd(rmsd):
    try:
        r = float(rmsd)
    except (TypeError, ValueError):
        return "RMSD unavailable."
    if r < 1.0:
        return (f"An RMSD of {r:.2f} Å is essentially a perfect reproduction of the "
                "crystallographic binding mode — the docking engine places the ligand "
                "exactly where experiment says it sits.")
    if r < 2.0:
        return (f"An RMSD of {r:.2f} Å is below the 2 Å threshold conventionally used "
                "to define a *correct* pose, so the protocol reproduces the known "
                "binding mode. This is the standard evidence that the search box, "
                "scoring function and preparation are all working for this target.")
    if r < 3.0:
        return (f"An RMSD of {r:.2f} Å is above the 2 Å success threshold — the pose is "
                "in the right pocket but the orientation drifts. Worth widening "
                "exhaustiveness or re-checking the box before trusting rankings.")
    return (f"An RMSD of {r:.2f} Å means the redocked pose does not reproduce the "
            "crystal binding mode; treat the screening results with caution and "
            "re-examine box placement and ligand preparation.")


def _interpret_auc(auc):
    try:
        a = float(auc)
    except (TypeError, ValueError):
        return "AUC unavailable."
    if a >= 0.8:
        q = "strong"
    elif a >= 0.7:
        q = "good"
    elif a >= 0.6:
        q = "moderate"
    elif a > 0.55:
        q = "weak but above chance"
    else:
        q = "no better than random"
    return (f"AUC-ROC = {a:.3f} indicates **{q}** separation of known actives from "
            "decoys. AUC is the probability that a randomly chosen active is ranked "
            "above a randomly chosen decoy: 0.5 is random, 1.0 is perfect.")


def _interpret_ef(ef1):
    try:
        e = float(ef1)
    except (TypeError, ValueError):
        return "EF unavailable."
    return (f"EF@1% = {e:.2f} means the top 1% of the ranked library is enriched in "
            f"true actives {e:.1f}× over what random picking would give — this is the "
            "number that matters most in a real campaign, where you only assay the top "
            "handful of compounds.")


def _consensus_vs_single(metrics):
    """Explain whether consensus beat the single engines (2- or 3-engine)."""
    try:
        c = float(metrics.get("auc_roc"))
        singles = {}
        for eng in ("vina", "smina", "gnina"):
            v = metrics.get(f"auc_{eng}")
            if v is not None:
                singles[eng.capitalize() if eng != "gnina" else "GNINA"] = float(v)
        if len(singles) < 2:
            return ""
    except (TypeError, ValueError):
        return ""
    best_eng, best_single = max(singles.items(), key=lambda kv: kv[1])
    single_str = ", ".join(f"{k} {v:.3f}" for k, v in singles.items())
    n_eng = len(singles)
    if c >= best_single:
        return (f"The {n_eng}-engine consensus AUC ({c:.3f}) is at least as good as any "
                f"single engine alone ({single_str}) — combining independent scoring "
                "functions cancels their individual failure modes and gives a more "
                "robust ranking than trusting one engine.")
    return (f"On this run a single engine (best: {best_eng} at {best_single:.3f}) edged out "
            f"the {n_eng}-engine consensus ({c:.3f}; other engines: {single_str}). With small "
            "or noisy libraries the engines occasionally disagree in a way that hurts the "
            "average; the consensus advantage is most reliable on larger, well-powered "
            "screens. The value of consensus is robustness across targets, not a "
            "guaranteed win on every run.")


def _pocket_method_note(box):
    """Plain-language rationale + this run's actual pocket-detection outcome."""
    if not box:
        return ""
    method = box.get("method")
    if method == "ligand-guided":
        return (" This run had a co-crystallised ligand, so the box is centred on it "
                "directly — the single most reliable box definition, and no cavity "
                "predictor is needed.")
    n_alt = max(0, len(box.get("all_pockets", [])) - 1)
    if method == "p2rank":
        prob = box.get("probability")
        prob_txt = f" (top-pocket probability {prob:.2f})" if prob is not None else ""
        alt_txt = (f" {n_alt} lower-ranked candidate pocket(s) were also recorded "
                   "(see `box.json` → `all_pockets`) in case the top pick is wrong for "
                   "a cryptic or unusual site." if n_alt else "")
        return (f" **This run used P2Rank**{prob_txt} — a machine-learning predictor "
                "(random forest over solvent-accessible surface points, trained on "
                "known binding sites) chosen over pure-geometry cavity detection "
                "(fpocket) because published benchmarks (Krivák & Hoksza, "
                "*J Cheminform* 2018, and later P2Rank work) show it beats fpocket on "
                "top-1 pocket identification accuracy, particularly for apo structures "
                "with no bound ligand to hint at the site." + alt_txt)
    if method == "fpocket":
        alt_txt = (f" {n_alt} lower-ranked candidate pocket(s) were also recorded." if n_alt else "")
        return (" **This run used fpocket** (pure-geometry alpha-sphere clustering) — "
                "the automatic fallback used because P2Rank (or its Java runtime) "
                "wasn't available on this machine. It still ranks cavities by a "
                "druggability score and remains a reasonable apo-structure fallback, "
                "just with lower published top-1 accuracy than P2Rank." + alt_txt)
    return ""


def _prep_tool_note(prep_stats):
    """Plain-language rationale + this run's actual ligand-prep tool split."""
    if not prep_stats or not prep_stats.get("n_ok"):
        return ""
    n_ok = prep_stats["n_ok"]
    n_obabel = prep_stats.get("n_obabel", 0)
    if n_obabel == 0:
        outcome = f"Meeko successfully prepared all {n_ok} docked ligand(s) — no fallback was needed."
    else:
        outcome = (f"Of {n_ok} docked ligand(s), {n_obabel} needed the OpenBabel fallback "
                   f"(Meeko rejected them — usually a macrocycle or unusual valence) and "
                   f"{n_ok - n_obabel} used Meeko directly.")
    return (" **Ligand/receptor PDBQT prep uses Meeko first, OpenBabel as a fallback.** "
           "Meeko is built and maintained by the Forli Lab — the same group behind "
           "AutoDock/Vina — so it assigns AutoDock atom types and torsion trees the way "
           "Vina/GNINA actually expect, and is the modern recommended replacement for "
           "older AutoDockTools-based prep. OpenBabel is a more permissive general "
           "cheminformatics converter: less precise on AutoDock-specific typing, but a "
           "good safety net that succeeds on some inputs Meeko rejects, so one "
           "molecule's edge case no longer silently drops it from the whole run. "
           + outcome)


def _trust_section(run_trust):
    """
    The headline differentiator: a run-level trust verdict, shown before
    anything else. Most docking tools hand you a score; this section is
    what lets you decide whether to believe it -- computed automatically,
    the same way on every run, not asserted by whoever is presenting it.
    """
    if not run_trust:
        return ""
    md = ["## \U0001F512 Trust verdict — can this run's ranking be trusted?\n"]
    title = run_trust["status"].replace("_", " ").title()
    md.append(f"### {run_trust['emoji']} {title}\n")
    md.append(run_trust["headline"] + "\n")
    md.append("**Why:**\n")
    for r in run_trust["reasons"]:
        md.append(f"- {r}\n")
    md.append(
        "\n*This verdict is computed the same way on every run, before any hit is shown — "
        "it is not a claim added after the fact. Every ranked compound below also carries "
        "its own confidence tier (High/Medium/Low), capped by this verdict: no individual "
        "hit can be rated more trustworthy than the protocol itself has been shown to be on "
        "this target — see §3.*\n")
    return "".join(md)


def build_report(target_id, metrics, redock, figs, top_hits_df, versions,
                 outdir, box=None, prep_stats=None, run_trust=None):
    """Assemble an explained Markdown report.

    Beyond the raw numbers, every section now carries plain-language text that
    (a) explains what the method/metric is and (b) interprets the *actual*
    values from this run, so a judge or teammate reading it cold understands
    both the approach and whether it worked.
    """
    os.makedirs(outdir, exist_ok=True)
    md = []
    md.append(f"# PrismDock Consensus Docking Report — {target_id}\n")
    md.append("**PrismDock — P2Rank-Guided Rank-Integrated Smina/Vina Molecular Docking & Validation Suite**\n\n"
              "**Author:** Sridhar J D, M.Pharm (Pharmaceutical Chemistry) | **GitHub:** [github.com/jdsridhar/prismdock](https://github.com/jdsridhar/prismdock) | **ORCID:** [0009-0009-5196-7100](https://orcid.org/0009-0009-5196-7100)\n\n"
              "_100% open-source pipeline: AutoDock Vina + Smina consensus, with P2Rank ML pocket prediction and automatic self-validation._\n")

    md.append(_trust_section(run_trust))

    # ---- Executive summary -------------------------------------------------
    md.append("## Executive summary\n")
    summary_bits = []
    if redock and redock.get("rmsd_top_pose") is not None:
        ok = redock.get("success_2A")
        summary_bits.append(
            f"the protocol {'reproduces' if ok else 'does not reproduce'} the known "
            f"crystal pose (redock RMSD {_fnum(redock.get('rmsd_top_pose'), 2)} Å)")
    if metrics.get("auc_roc") is not None:
        summary_bits.append(
            f"it separates actives from decoys with AUC-ROC {_fnum(metrics.get('auc_roc'))} "
            f"and {_fnum(metrics.get('EF_1pct'), 1)}× enrichment in the top 1%")
    if summary_bits:
        md.append("On **" + str(target_id) + "**, " + "; ".join(summary_bits) + ".\n")
    md.append(
        "This pipeline's novelty is not a single new score but a **defensible workflow**: "
        "two independent open-source engines are combined into a consensus ranking, and "
        "the same run automatically checks itself two ways — by re-docking the native "
        "ligand (does it recover the real binding mode?) and by screening known actives "
        "against decoys (does the ranking actually enrich for binders?). Every number "
        "below is produced by the run itself, not entered by hand.\n")

    # ---- How to read this report ------------------------------------------
    has_gnina_metric = metrics.get("auc_gnina") is not None
    md.append("## How the pipeline works\n")
    if has_gnina_metric:
        step3 = ("3. **Dock/score with three engines** — AutoDock Vina (physics-empirical "
                "score), Smina (Vinardo score), and GNINA (CNN pose/affinity model rescoring "
                "the Vina poses) — three methodologically distinct scoring functions, so "
                "agreement between them is meaningful signal.\n")
        step4 = ("4. **Consensus rank** each ligand by combining its rank and its normalised "
                "score across all three engines (`consensus_z`, lower = better), plus an "
                "`agreement` term (max-min spread) that flags where the engines concur, and "
                "`n_engines_scored` recording how many engines actually scored each ligand.\n")
    else:
        step3 = ("3. **Dock with two engines** — AutoDock Vina (physics-empirical score) and "
                "Smina (Vinardo score) — which use different scoring functions, so agreement "
                "between them is meaningful signal.\n")
        step4 = ("4. **Consensus rank** each ligand by combining its rank and its normalised "
                "score across both engines (`consensus_z`, lower = better), plus an "
                "`agreement` term that flags where the two engines concur.\n")
    md.append(
        "1. **Prepare** the target (clean, protonate at physiological pH, assign "
        "AutoDock atom types) and each ligand (standardise, 3D embed, protonate)."
        + _prep_tool_note(prep_stats) + "\n"
        "2. **Locate the pocket** from the co-crystallised ligand when present, or with "
        "P2Rank (falling back to fpocket) for an apo structure, and build the search box."
        + _pocket_method_note(box) + "\n"
        + step3 + step4 +
        "5. **Self-validate** with redock RMSD and retrospective enrichment metrics "
        "(below).\n")

    # ---- Section 1: protocol validation -----------------------------------
    md.append("## 1. Protocol validation — does the method reproduce known truth?\n")
    if redock and redock.get("rmsd_top_pose") is not None:
        ok = redock.get("success_2A")
        md.append(f"**Pose reproduction (native-ligand redock):** top-pose RMSD "
                  f"= **{_fnum(redock.get('rmsd_top_pose'), 2)} Å**, "
                  f"best-of-N = {_fnum(redock.get('rmsd_best_pose'), 2)} Å "
                  f"→ {'**PASS** (<2 Å)' if ok else '**FAIL** (>2 Å)'}.\n")
        md.append("*What this is:* we take the ligand from the crystal structure, strip "
                  "it out, and dock it back blind. If the top pose lands within 2 Å "
                  "(heavy-atom, symmetry-corrected RMSD) of where it really sits, the "
                  "search and scoring are trustworthy for this target.\n")
        md.append("*Reading this run:* " + _interpret_rmsd(redock.get("rmsd_top_pose")) + "\n")
    else:
        md.append("_No native ligand was available to redock (apo target), so pose "
                  "reproduction could not be measured for this run._\n")

    n_eng = metrics.get("n_engines", 2)
    md.append("**Retrospective screening (known actives vs. decoys):**\n")
    engine_rows = ""
    if metrics.get("auc_vina") is not None:
        engine_rows += f"| AUC-ROC (Vina only) | {_fnum(metrics.get('auc_vina'))} | single-engine baseline |\n"
    if metrics.get("auc_smina") is not None:
        engine_rows += f"| AUC-ROC (Smina only) | {_fnum(metrics.get('auc_smina'))} | single-engine baseline |\n"
    if metrics.get("auc_gnina") is not None:
        engine_rows += (f"| AUC-ROC (GNINA CNN only) | {_fnum(metrics.get('auc_gnina'))} | "
                        "single-engine baseline (CNN pose/affinity model) |\n")
    md.append(f"| Metric | Value | What it measures |\n|---|---|---|\n"
              f"| AUC-ROC (consensus, {n_eng}-engine) | **{_fnum(metrics.get('auc_roc'))}** | overall ranking power (0.5 = random) |\n"
              f"{engine_rows}"
              f"| EF @ 1% | {_fnum(metrics.get('EF_1pct'), 2)} | fold-enrichment in the top 1% |\n"
              f"| EF @ 5% | {_fnum(metrics.get('EF_5pct'), 2)} | fold-enrichment in the top 5% |\n"
              f"| EF @ 10% | {_fnum(metrics.get('EF_10pct'), 2)} | fold-enrichment in the top 10% |\n"
              f"| BEDROC (α=20) | {_fnum(metrics.get('BEDROC_a20'))} | early-recognition (weights top of list) |\n"
              f"| Actives / Decoys | {metrics.get('n_actives')} / {metrics.get('n_decoys')} | benchmark composition |\n")
    md.append("*Reading this run:*\n")
    md.append("- " + _interpret_auc(metrics.get("auc_roc")) + "\n")
    md.append("- " + _interpret_ef(metrics.get("EF_1pct")) + "\n")
    cvs = _consensus_vs_single(metrics)
    if cvs:
        md.append("- " + cvs + "\n")
    md.append("- **BEDROC (α=20)** answers a different question from AUC: it weights the "
              "very top of the ranked list, matching how screening is used in practice "
              "(you only test the top few). A value near 1 means actives cluster at the "
              "top; near 0 means they are scattered.\n")

    # ---- Section 2: figures with captions ----------------------------------
    md.append("\n## 2. Figures\n")
    fig_captions = {
        "roc": ("ROC curves", "Each curve traces true-positive vs false-positive rate as "
                "the score threshold sweeps. A curve bowing toward the top-left is better; "
                "the diagonal is random. Compare the consensus curve against the two "
                "single-engine curves to see the effect of combining them."),
        "enrichment": ("Enrichment", "Fraction of actives recovered vs. fraction of the "
                "library screened. The steeper the early rise, the more actives you catch "
                "while testing only a small slice — the practical payoff of a good ranking. "
                "The black dotted line is the theoretical ideal, grey is random."),
        "score_dist": ("Score separation", "Distribution of the consensus score for actives "
                "(red) vs decoys (grey). Good separation — actives shifted toward better "
                "(lower) scores with little overlap — is what drives the AUC and EF above."),
        "agreement": ("Cross-engine agreement", "Vina vs Smina affinity for every compound, "
                "coloured by label. Points on a tight diagonal mean the two independent "
                "engines concur; the annotated Pearson r quantifies it. Consensus is most "
                "trustworthy where the engines agree."),
    }
    for key in ["roc", "enrichment", "score_dist", "agreement"]:
        if key in figs:
            title, cap = fig_captions[key]
            md.append(f"### {title}\n![{title}]({os.path.basename(figs[key])})\n")
            md.append(f"*{cap}*\n")

    # ---- Section 3: top hits ----------------------------------------------
    md.append("\n## 3. Top consensus hits\n")
    md.append("Ranked by `consensus_z` (lower = better). `agreement` near 1 means Vina and "
              "Smina independently favour the compound — the hits you can trust most. "
              "`label` (if present) marks known actives (1) vs decoys (0) from the "
              "benchmark; on a real event-day target every compound is unlabelled.\n")
    has_confidence = "confidence_tier" in top_hits_df.columns
    if has_confidence:
        md.append(
            "`confidence` is *not* the same thing as the score — it's whether the score "
            "should be believed: High requires full engine coverage, strong agreement, "
            "*and* this run's protocol passing its own validation (§Trust verdict above); "
            "a compound can score well and still be Low confidence (e.g. only one engine "
            "scored it, or the protocol itself is unvalidated on this target).\n")
    cols = [c for c in ["final_rank", "id", "label", "vina_score", "smina_score", "gnina_score",
                        "consensus_z", "agreement", "n_engines_scored", "confidence_tier"]
           if c in top_hits_df.columns]
    try:
        md.append(top_hits_df[cols].head(15).to_markdown(index=False))
    except ImportError:
        md.append(top_hits_df[cols].head(15).to_string(index=False))

    if has_confidence and not top_hits_df.empty:
        top = top_hits_df.iloc[0]
        md.append(f"\n\n**Worked example — why `{top['id']}` (rank #1) is rated "
                  f"{top.get('confidence_tier', '?')} confidence:** {top.get('confidence_reason', 'n/a')}.\n")

    if box:
        md.append(f"\n## 4. Docking box\n")
        md.append("The search space the engines explored. A `ligand-guided` box is centred "
                  "on the co-crystal ligand; a `p2rank`/`fpocket` box comes from apo cavity "
                  "prediction (P2Rank preferred, fpocket as automatic fallback — see "
                  "*How the pipeline works* above for why)." + _pocket_method_note(box) + "\n")
        md.append(f"- method: `{box.get('method')}`\n"
                  f"- center: {box.get('center')}\n- size (Å): {box.get('size')}\n")
        alt = box.get("all_pockets")
        if alt and len(alt) > 1:
            md.append(f"\n**All {len(alt)} candidate pockets found** (top-ranked is the box used above):\n\n")
            md.append("| rank | center | probability/score |\n|---|---|---|\n")
            for p in alt[:10]:
                pscore = p.get("probability", p.get("n_pocket_atoms", "—"))
                md.append(f"| {p.get('rank')} | {p.get('center')} | {pscore} |\n")

    # ---- Limitations -------------------------------------------------------
    md.append("\n## 5. Limitations & honest caveats\n")
    md.append(
        "- Docking scores are approximate free-energy estimates, not experimental "
        "affinities — rankings are far more reliable than absolute numbers.\n"
        "- Retrospective enrichment uses property-matched decoys; real prospective "
        "performance is usually lower.\n"
        "- A single receptor conformation is used (no explicit protein flexibility "
        "beyond side-chain sampling in the box).\n"
        "- Metrics on small libraries carry large error bars; trust them most on the "
        "full benchmark run.\n")

    # ---- Reproducibility ---------------------------------------------------
    md.append("\n## 6. Reproducibility\n")
    md.append("Every tool is open-source and every step is scripted and seeded. Versions "
              "used for this run:\n")
    md.append("```\n" + "\n".join(f"{k}: {v}" for k, v in versions.items()) + "\n```\n")

    report_md = os.path.join(outdir, "report.md")
    with open(report_md, "w") as fh:
        fh.write("\n".join(md))
    print(f"[report] wrote {report_md}")
    return report_md
