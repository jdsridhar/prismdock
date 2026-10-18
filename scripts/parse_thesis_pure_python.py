import csv
import os

pkg_dir = 'guide_packets/Package_A_PPARg_Diabetes'
out_path = 'guide_packets/Package_A_PPARg_Diabetes/parsed_thesis_summary.txt'

lines = []
def log(s=""):
    lines.append(s)

log("=== 1. ADMET DRUGLIKENESS SHORTLIST ===")
p_admet = os.path.join(pkg_dir, 'actual_admet_druglikeness_shortlist.csv')
if os.path.exists(p_admet):
    with open(p_admet, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            log(", ".join(row))
else:
    log("Not found: " + p_admet)

log("\n=== 2. SCREENING SET COMPOUNDS SAMPLE ===")
p_comp = os.path.join(pkg_dir, 'actual_phytochem_screening_set.csv')
if os.path.exists(p_comp):
    with open(p_comp, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        log("Headers: " + str(headers))
        count = 0
        for row in reader:
            count += 1
            if count <= 10:
                log(f"Row {count}: {row}")
        log(f"Total screening set compounds: {count}")
else:
    log("Not found: " + p_comp)

log("\n=== 3. DOCKING SCORES ANALYSIS ===")
p_dock = os.path.join(pkg_dir, 'actual_docking_scores.csv')
if os.path.exists(p_dock):
    with open(p_dock, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        log("Fieldnames: " + str(fieldnames))
        
        # Collect rows
        rows = []
        for r in reader:
            rows.append(r)
        log(f"Total docked rows: {len(rows)}")
        
        # Determine score field
        score_field = None
        for sf in ['vina_score', 'docking_score', 'score', 'affinity', 'binding_affinity', 'docking_score_kcal_mol', 'score_1zgy', 'score_2q5s']:
            if sf in fieldnames:
                score_field = sf
                break
        if not score_field:
            # check any field with score
            for fn in fieldnames:
                if 'score' in fn.lower() or 'affinity' in fn.lower() or 'vina' in fn.lower() or 'kcal' in fn.lower():
                    score_field = fn
                    break
        log(f"Selected score field: {score_field}")
        
        # Try sorting if score_field found
        if score_field:
            def get_score(row):
                try:
                    return float(row[score_field])
                except:
                    return 999.0
            rows.sort(key=get_score)
            log(f"\nTop 30 docked compounds by {score_field}:")
            for i, r in enumerate(rows[:30]):
                log(f"Rank {i+1}: {r}")
                
        # Also check reference compound
        ref_rows = [r for r in rows if any('rosiglitazone' in str(v).lower() or 'brl' in str(v).lower() or 'reference' in str(v).lower() for v in r.values())]
        log(f"\nReference rows found ({len(ref_rows)}):")
        for rr in ref_rows:
            log(str(rr))
else:
    log("Not found: " + p_dock)

log("\n=== 4. INTERACTION FINGERPRINT MATRIX SAMPLE ===")
p_ifp = os.path.join(pkg_dir, 'actual_interaction_fingerprint_matrix.csv')
if os.path.exists(p_ifp):
    with open(p_ifp, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        log(f"IFP Headers ({len(headers)} columns): {headers[:15]}...")
        count = 0
        for row in reader:
            count += 1
            if count <= 10:
                log(f"IFP Row {count}: {row[:15]}...")
        log(f"Total IFP compounds: {count}")
else:
    log("Not found: " + p_ifp)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

print("DONE! Wrote to", out_path)
