"""
compare.py — Side-by-side comparison of selected molecules: Standard
(reference/known active), Co-crystal (native ligand), Lead hit (top
consensus rank), and any other hand-picked compounds.

Produces:
  * role defaults (best-guess Standard / Co-crystal / Lead hit ids)
  * a comparison table (scores, agreement, ADMET, contact summary)
  * a plain-language narrative explaining which compound is best and why
    the others fall short (score gap, engine disagreement, fewer/weaker
    contacts, or drug-likeness red flags)

Confidence tiers (High/Medium/Low) shown here come from trust.hit_confidence
-- the SAME engine that annotates the Hits table and the report, so "why is
this one best" reads consistently everywhere in the app, not as three
different heuristics wearing different labels.

Open-source: pandas, RDKit.
"""
from __future__ import annotations

import pandas as pd

from trust import contact_summary, hit_confidence, n_engines_from_columns

CO_CRYSTAL_PREFIX = "CO_CRYSTAL_"

ROLE_LABELS = {"standard": "Standard", "co_crystal": "Co-crystal", "lead_hit": "Lead hit"}
ROLE_COLORS = {"standard": "#3498db", "co_crystal": "#95a5a6", "lead_hit": "#2ecc71"}
EXTRA_PALETTE = ["#f39c12", "#9b59b6", "#e91e8c", "#f1c40f", "#ff6fae", "#17becf"]


def default_roles(df: pd.DataFrame) -> dict:
    """
    Guess sensible defaults for Standard / Co-crystal / Lead-hit from the
    consensus results DataFrame. Any role left as None means "ask the user".
    """
    roles = {"standard": None, "co_crystal": None, "lead_hit": None}
    if df is None or df.empty:
        return roles
    ids = df["id"].astype(str).tolist()
    sort_col = "final_rank" if "final_rank" in df.columns else df.columns[-1]
    ranked = df.assign(id=df["id"].astype(str)).sort_values(sort_col)

    co_crystal_ids = [i for i in ids if i.startswith(CO_CRYSTAL_PREFIX)]
    if co_crystal_ids:
        roles["co_crystal"] = co_crystal_ids[0]

    non_native = ranked[~ranked["id"].isin(co_crystal_ids)]
    lead_pool = non_native if not non_native.empty else ranked
    if not lead_pool.empty:
        roles["lead_hit"] = str(lead_pool.iloc[0]["id"])

    if "label" in ranked.columns and ranked["label"].notna().any():
        actives = ranked[(ranked["label"] == 1) &
                         (~ranked["id"].isin([roles["co_crystal"], roles["lead_hit"]]))]
        if not actives.empty:
            roles["standard"] = str(actives.iloc[0]["id"])

    return roles


def admet_properties(smi) -> dict:
    """Lipinski/ADMET descriptors for one SMILES. Returns Nones on failure."""
    empty = {"MW": None, "LogP": None, "HBD": None, "HBA": None, "TPSA": None}
    if not isinstance(smi, str) or not smi:
        return empty
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        m = Chem.MolFromSmiles(smi)
        if not m:
            return empty
        return {
            "MW": round(Descriptors.MolWt(m), 2),
            "LogP": round(Descriptors.MolLogP(m), 2),
            "HBD": rdMolDescriptors.CalcNumLipinskiHBD(m),
            "HBA": rdMolDescriptors.CalcNumLipinskiHBA(m),
            "TPSA": round(Descriptors.TPSA(m), 2),
        }
    except Exception:
        return empty


def lipinski_violations(admet: dict) -> list[str]:
    """Rule-of-5 flags (informational — a violation doesn't disqualify a hit)."""
    v = []
    if admet.get("MW") is not None and admet["MW"] > 500:
        v.append(f"MW {admet['MW']:.0f} > 500")
    if admet.get("LogP") is not None and admet["LogP"] > 5:
        v.append(f"LogP {admet['LogP']:.1f} > 5")
    if admet.get("HBD") is not None and admet["HBD"] > 5:
        v.append(f"{admet['HBD']} H-bond donors > 5")
    if admet.get("HBA") is not None and admet["HBA"] > 10:
        v.append(f"{admet['HBA']} H-bond acceptors > 10")
    return v


def comparison_table(df: pd.DataFrame, ids: list[str], manifest_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per selected id, ordered as given, with scores + ADMET."""
    work = df.assign(id=df["id"].astype(str))
    sub = work[work["id"].isin(ids)].copy()
    sub = sub.set_index("id").reindex([i for i in ids if i in sub["id"].tolist()] or ids).dropna(how="all").reset_index()

    cols = [c for c in ["id", "final_rank", "vina_score", "smina_score", "gnina_score",
                        "consensus_z", "agreement", "n_engines_scored", "label"] if c in sub.columns]
    out = sub[cols].copy() if cols else pd.DataFrame({"id": ids})

    if manifest_df is not None and "smiles" in manifest_df.columns:
        man = manifest_df.assign(id=manifest_df["id"].astype(str))
        out = out.merge(man[["id", "smiles"]], on="id", how="left")
        admet = out["smiles"].apply(admet_properties).apply(pd.Series)
        out = pd.concat([out, admet], axis=1)
    return out


def explain_comparison(table_df: pd.DataFrame, role_of_id: dict, contacts_by_id: dict,
                       run_trust: dict | None = None, n_engines_total: int | None = None) -> dict:
    """
    Build a plain-language comparison. Returns
    {"per_molecule": {id: str}, "verdict": str, "best_id": str|None}.
    Ranking for "best" uses consensus_z when present (lower=better), else
    final_rank, else the given order.

    When `run_trust` (from trust.run_trust_verdict) is given, each molecule's
    confidence tier is computed via trust.hit_confidence -- capped by this
    run's own validation status -- instead of a standalone agreement read,
    so the verdict matches what the Hits tab and report show for the same id.
    """
    per_molecule = {}
    if table_df.empty:
        return {"per_molecule": {}, "verdict": "No molecules selected.", "best_id": None}

    if "consensus_z" in table_df.columns and table_df["consensus_z"].notna().any():
        ranked = table_df.dropna(subset=["consensus_z"]).sort_values("consensus_z")
        rank_basis = "consensus_z"
    elif "final_rank" in table_df.columns and table_df["final_rank"].notna().any():
        ranked = table_df.dropna(subset=["final_rank"]).sort_values("final_rank")
        rank_basis = "final_rank"
    else:
        ranked = table_df
        rank_basis = None
    best_id = str(ranked.iloc[0]["id"]) if not ranked.empty else None
    best_row = ranked.iloc[0] if not ranked.empty else None

    for _, row in table_df.iterrows():
        mid = str(row["id"])
        role = role_of_id.get(mid)
        role_txt = f"**{role}** " if role else ""
        bits = []
        if pd.notna(row.get("final_rank")):
            bits.append(f"consensus rank #{int(row['final_rank'])}")
        engine_scores = [f"{lbl} {row[col]:.2f}" for col, lbl in
                         [("vina_score", "Vina"), ("smina_score", "Smina"), ("gnina_score", "GNINA")]
                         if col in row and pd.notna(row[col])]
        if engine_scores:
            bits.append("scores: " + ", ".join(engine_scores))
        if run_trust is not None:
            tier_info = hit_confidence(row, run_trust, n_engines_total or 2)
            bits.append(f"**{tier_info['tier']} confidence** — {tier_info['reason']}")
        elif pd.notna(row.get("agreement")):
            a = row["agreement"]
            bits.append(("engines strongly agree" if a > 0.8 else
                        "engines mostly agree" if a > 0.5 else
                        "engines disagree noticeably") + f" (agreement={a:.2f})")
        if row.get("id") not in table_df["id"].values or pd.isna(row.get("final_rank", pd.NA)):
            bits.append("not found in this run's docked results (no score available)")

        cs = contact_summary(contacts_by_id.get(mid))
        contact_txt = ""
        if cs["n_contacts"]:
            contact_txt = (f" Makes {cs['n_contacts']} contact(s) across {cs['n_residues']} "
                           f"pocket residues, {cs['n_specific']} of them specific "
                           f"(H-bond/ionic/π-type) ({cs['top_types']}).")

        admet = {k: row.get(k) for k in ("MW", "LogP", "HBD", "HBA", "TPSA")}
        violations = lipinski_violations(admet) if any(v is not None for v in admet.values()) else []
        admet_txt = f" Lipinski flags: {', '.join(violations)}." if violations else ""

        if best_id is not None and mid == best_id:
            bits.append("**ranks best of the compared set**")
        elif rank_basis and pd.notna(row.get(rank_basis)) and best_row is not None and pd.notna(best_row.get(rank_basis)):
            gap = row[rank_basis] - best_row[rank_basis]
            if rank_basis == "consensus_z" and gap > 0:
                bits.append(f"{gap:.2f} worse consensus_z than the top-ranked compound here")

        per_molecule[mid] = f"{role_txt}**{mid}** — " + "; ".join(bits) + "." + contact_txt + admet_txt

    verdict_bits = []
    if best_id:
        role = role_of_id.get(best_id)
        role_txt = f" ({role})" if role else ""
        tier_txt = ""
        if run_trust is not None:
            best_tier = hit_confidence(best_row, run_trust, n_engines_total or 2)["tier"]
            tier_txt = f" — rated **{best_tier} confidence**"
        verdict_bits.append(f"**{best_id}**{role_txt} ranks best of the {len(table_df)} compared "
                            f"by {('consensus score' if rank_basis == 'consensus_z' else 'consensus rank')}"
                            f"{tier_txt}.")
        worst = ranked.iloc[-1] if len(ranked) > 1 else None
        if worst is not None and str(worst["id"]) != best_id:
            worst_id = str(worst["id"])
            reasons = []
            if pd.notna(worst.get("agreement")) and pd.notna(best_row.get("agreement")) and worst["agreement"] < best_row["agreement"]:
                reasons.append("weaker cross-engine agreement")
            wc = contact_summary(contacts_by_id.get(worst_id))
            bc = contact_summary(contacts_by_id.get(best_id))
            if wc["n_specific"] < bc["n_specific"]:
                reasons.append("fewer specific (H-bond/ionic) contacts")
            w_admet = {k: worst.get(k) for k in ("MW", "LogP", "HBD", "HBA", "TPSA")}
            if lipinski_violations(w_admet):
                reasons.append("drug-likeness flags")
            reason_txt = f" — mainly due to {', '.join(reasons)}" if reasons else ""
            verdict_bits.append(f"**{worst_id}** ranks weakest here{reason_txt}.")
    verdict_bits.append(
        "Ranking is driven primarily by `consensus_z` (mean of normalized per-engine scores, "
        "lower=better); when two compounds are close, prefer the one with higher `agreement` "
        "(independent engines concur) and more specific contacts over the raw score gap alone.")

    return {"per_molecule": per_molecule, "verdict": " ".join(verdict_bits), "best_id": best_id}
