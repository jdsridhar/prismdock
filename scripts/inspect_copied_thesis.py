import os
import pandas as pd

pkg_dir = 'guide_packets/Package_A_PPARg_Diabetes'

print("=== ADMET SHORTLIST ===")
df_admet = pd.read_csv(os.path.join(pkg_dir, 'actual_admet_druglikeness_shortlist.csv'))
print(df_admet.to_string())

print("\n=== DOCKING SCORES ===")
df_dock = pd.read_csv(os.path.join(pkg_dir, 'actual_docking_scores.csv'))
print(f"Total docked records: {len(df_dock)}")
print("Columns:", df_dock.columns.tolist())
print("\nFirst 5 rows:")
print(df_dock.head())

# Find score column
score_col = None
for col in ['vina_score', 'docking_score', 'score', 'affinity', 'binding_affinity']:
    if col in df_dock.columns:
        score_col = col
        break
if not score_col:
    # Look for numeric columns
    numeric_cols = df_dock.select_dtypes(include=['float64', 'int64']).columns
    print("Numeric columns:", numeric_cols.tolist())
    score_col = numeric_cols[0]

print(f"\nSorting by {score_col} (ascending, best binding):")
df_sorted = df_dock.sort_values(by=score_col, ascending=True)
print(df_sorted.head(25).to_string())

print("\n=== SCREENING SET COMPOUNDS ===")
df_comp = pd.read_csv(os.path.join(pkg_dir, 'actual_phytochem_screening_set.csv'))
print(f"Total screening set compounds: {len(df_comp)}")
print("Columns:", df_comp.columns.tolist())
print(df_comp.head(5).to_string())

print("\n=== INTERACTION FINGERPRINT MATRIX ===")
df_ifp = pd.read_csv(os.path.join(pkg_dir, 'actual_interaction_fingerprint_matrix.csv'))
print(f"IFP shape: {df_ifp.shape}")
print(df_ifp.head(10).to_string())
