"""
trust.py — THE TRUST LAYER. Turns raw validation numbers and per-hit engine
agreement into two explicit, explained verdicts:

  1. RUN-LEVEL TRUST (`run_trust_verdict`): before anyone looks at a single
     hit, did THIS run's protocol demonstrate it actually works on THIS
     target? Combines redock RMSD (does it reproduce the known pose?) and
     retrospective enrichment (does the score separate actives from
     decoys?) into one status: validated / partially_validated / failed /
     unvalidated. "unvalidated" (no native ligand, no labels) is reported
     honestly as "nothing could be checked", never dressed up as a pass.

  2. HIT-LEVEL CONFIDENCE (`hit_confidence` / `annotate_hits`): for each
     ranked compound, is its rank corroborated by independent engines, or
     is one engine carrying it? Cross-engine `agreement` alone is not
     trustworthy on its own -- a ligand scored by only one engine shows
     agreement=1.0 (there's nothing to disagree with), which would read as
     maximum confidence for the least evidence. This module explicitly
     downgrades that case, and caps every hit's ceiling by the run-level
     trust status: no individual hit can be "High confidence" if the
     protocol itself hasn't been shown to work on this target, no matter
     how well its engines agree with each other.

Most docking tools report a score. This module is what lets the rest of
the pipeline report a score you can defend.

Open-source: pandas.
"""
from __future__ import annotations

import pandas as pd

# AUC threshold for "the ranking has real separating power" -- consistent
# with report.py's _interpret_auc, which calls 0.6+ "moderate or better".
_AUC_PASS = 0.6
# Agreement breakpoints -- consistent with the "strongly/mostly agree"
# language already used in app.py's Pose Viewer and compare.py.
_AGREEMENT_HIGH = 0.8
_AGREEMENT_MED = 0.5

_TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
_STATUS_CAP = {
    "validated": "High",
    "partially_validated": "Medium",
    "unvalidated": "Medium",
    "failed": "Low",
}
_STATUS_HEADLINE = {
    "validated": (
        "Validated — this run's protocol reproduces the known crystal pose "
        "AND separates known actives from decoys on this target. Hits below "
        "can be trusted at face value, modulated by each hit's own "
        "confidence tier."),
    "partially_validated": (
        "Partially validated — only some of the available self-checks "
        "passed (see reasons below). Treat hits as promising leads, not "
        "proven ones."),
    "failed": (
        "Validation failed — the self-checks that could run did not pass. "
        "Treat every hit here as exploratory only; this run has not "
        "demonstrated the protocol works on this target."),
    "unvalidated": (
        "Unvalidated — no native ligand was available to redock and this "
        "library has no labeled actives/decoys, so neither self-check "
        "could run. Expected for a genuinely blind prospective screen: "
        "hits are exploratory by construction, not because anything "
        "failed."),
}
_STATUS_EMOJI = {"validated": "✅", "partially_validated": "⚠️",
                 "failed": "🛑", "unvalidated": "❔"}


def run_trust_verdict(redock: dict | None, metrics: dict | None) -> dict:
    """
    Combine redock RMSD + retrospective enrichment into one run-level trust
    status. A check only counts if it actually ran (i.e. the target had a
    native ligand to redock, or the library had labels for enrichment).

      validated           — every check that could run, passed
      partially_validated — some checks passed, some failed or didn't run
      failed              — every check that ran, failed
      unvalidated         — no check could run at all
    """
    reasons = []
    checks_run, checks_passed = [], []

    redock_known = bool(redock and redock.get("rmsd_top_pose") is not None)
    if redock_known:
        # Deliberately graded on the TOP-ranked pose, not validate.py's
        # success_2A (which is best-of-N). In a real screen you only ever
        # see the top pose the engine hands you -- best-of-N answers "did
        # the right answer appear somewhere in the list", which you could
        # only confirm by already knowing the crystal answer. Top-pose is
        # the honest, actually-actionable bar.
        rmsd_top = redock["rmsd_top_pose"]
        redock_ok = rmsd_top < 2.0
        checks_run.append("pose_accuracy")
        if redock_ok:
            checks_passed.append("pose_accuracy")
        reasons.append(
            f"pose accuracy: top-pose redock RMSD {rmsd_top:.2f} Å "
            f"{'passes' if redock_ok else 'fails'} the 2 Å reproduction threshold "
            "(graded on the top-ranked pose, not best-of-N)")
    else:
        reasons.append("pose accuracy: no native ligand available to redock — unverified")

    auc = metrics.get("auc_roc") if metrics else None
    has_enrichment = auc is not None
    if has_enrichment:
        auc_ok = auc >= _AUC_PASS
        checks_run.append("ranking_power")
        if auc_ok:
            checks_passed.append("ranking_power")
        reasons.append(
            f"ranking power: retrospective AUC-ROC {auc:.3f} "
            f"{'passes' if auc_ok else 'is below'} the {_AUC_PASS} moderate-separation bar")
    else:
        reasons.append("ranking power: no labeled actives/decoys in this library — unverified "
                       "(prospective screening mode)")

    if not checks_run:
        status = "unvalidated"
    elif len(checks_passed) == len(checks_run):
        status = "validated"
    elif checks_passed:
        status = "partially_validated"
    else:
        status = "failed"

    return {
        "status": status,
        "emoji": _STATUS_EMOJI[status],
        "headline": _STATUS_HEADLINE[status],
        "reasons": reasons,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
    }


def n_engines_from_columns(df: pd.DataFrame) -> int:
    return sum(1 for e in ("vina", "smina", "gnina") if f"{e}_score" in df.columns)


def contact_summary(contacts: pd.DataFrame | None) -> dict:
    """Condense a ProLIF contact table into counts used for confidence/narrative."""
    if contacts is None or contacts.empty:
        return {"n_contacts": 0, "n_residues": 0, "n_specific": 0, "top_types": ""}
    specific_types = {"HBDonor", "HBAcceptor", "Cationic", "Anionic", "CationPi", "PiCation", "PiStacking"}
    n_specific = int(contacts["interaction"].isin(specific_types).sum())
    top_types = contacts["interaction"].value_counts()
    return {
        "n_contacts": int(len(contacts)),
        "n_residues": int(contacts["residue"].nunique()),
        "n_specific": n_specific,
        "top_types": ", ".join(f"{n}×{t}" for t, n in top_types.head(3).items()),
    }


def _raw_tier_from_agreement(agreement) -> tuple[str, str]:
    if agreement is None or (isinstance(agreement, float) and pd.isna(agreement)):
        return "Medium", "cross-engine agreement unavailable"
    if agreement >= _AGREEMENT_HIGH:
        return "High", f"engines strongly agree (agreement={agreement:.2f})"
    if agreement >= _AGREEMENT_MED:
        return "Medium", f"engines mostly agree (agreement={agreement:.2f})"
    return "Low", f"engines disagree noticeably (agreement={agreement:.2f})"


def hit_confidence(row, run_trust: dict, n_engines_total: int,
                   contacts_summary: dict | None = None) -> dict:
    """
    Per-hit confidence tier (High/Medium/Low) + a one-line, specific reason.

    Two corrections a naive "just show agreement" approach misses:
      * A ligand scored by only one engine has nothing to disagree with, so
        its `agreement` column reads 1.0 — that's an artifact of missing
        data, not evidence of consensus. Partial engine coverage is
        explicitly detected and forces Low.
      * No hit can rank above the run's own validation ceiling: if the
        protocol hasn't been shown to reproduce known truth on this target
        (see run_trust_verdict), even perfect cross-engine agreement is
        capped — agreement is necessary but not sufficient for trust.
    """
    agreement = row.get("agreement")
    n_scored = row.get("n_engines_scored")
    reasons = []

    partial_coverage = (n_scored is not None and not pd.isna(n_scored)
                        and n_engines_total and n_scored < n_engines_total)
    if partial_coverage:
        tier = "Low"
        reasons.append(f"only {int(n_scored)}/{int(n_engines_total)} engine(s) scored this "
                       "compound — agreement isn't meaningful evidence with partial coverage")
    else:
        tier, why = _raw_tier_from_agreement(agreement)
        reasons.append(why)
        cap = _STATUS_CAP[run_trust["status"]]
        if _TIER_RANK[tier] > _TIER_RANK[cap]:
            reasons.append(
                f"capped {tier}→{cap} because this run's protocol is "
                f"'{run_trust['status'].replace('_', ' ')}' (see Validation) — strong "
                "engine agreement doesn't matter if the protocol itself is unproven here")
            tier = cap

    if contacts_summary:
        if contacts_summary.get("n_specific", 0) > 0:
            reasons.append(f"makes {contacts_summary['n_specific']} specific "
                           "(H-bond/ionic/π) contact(s)")
        elif contacts_summary.get("n_contacts", 0) > 0:
            reasons.append("contacts are all generic (hydrophobic/VdW only) — no anchoring "
                           "polar interaction seen")

    return {"tier": tier, "reason": "; ".join(reasons)}


def annotate_hits(df: pd.DataFrame, run_trust: dict, n_engines_total: int | None = None,
                  contacts_by_id: dict | None = None) -> pd.DataFrame:
    """
    Add `confidence_tier` and `confidence_reason` columns to every row of a
    consensus results DataFrame. `contacts_by_id` (id -> ProLIF contact
    DataFrame) is optional and typically only supplied for a bounded top-K
    of hits (computing contacts for an entire library is expensive) —
    everything else still gets a numeric-only confidence tier.
    """
    if n_engines_total is None:
        n_engines_total = n_engines_from_columns(df)
    tiers, reasons = [], []
    for _, row in df.iterrows():
        csumm = None
        if contacts_by_id is not None:
            csumm_src = contacts_by_id.get(str(row.get("id")))
            if csumm_src is not None:
                csumm = contact_summary(csumm_src)
        result = hit_confidence(row, run_trust, n_engines_total, csumm)
        tiers.append(result["tier"])
        reasons.append(result["reason"])
    out = df.copy()
    out["confidence_tier"] = tiers
    out["confidence_reason"] = reasons
    return out
