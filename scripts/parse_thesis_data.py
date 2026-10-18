import os
import pandas as pd

base = r'C:\Users\sridhar\OneDrive\Desktop\project1\PPARG_MetS_NetworkPharm'

print('=== ADMET SHORTLIST ===')
try:
    p_admet = os.path.join(base, '09_results', 'admet_druglikeness_shortlist.csv')
    if os.path.exists(p_admet):
        df_admet = pd.read_csv(p_admet)
        print(df_admet.to_string())
    else:
        print('admet file not found at', p_admet)
except Exception as e:
    print('ADMET error:', e)

print('\n=== DOCKING SCORES ===')
try:
    p_dock = os.path.join(base, '09_results', 'docking_scores.csv')
    if os.path.exists(p_dock):
        df_dock = pd.read_csv(p_dock)
        print('Shape:', df_dock.shape)
        print('Columns:', df_dock.columns.tolist())
        print('\nFirst 5 rows:')
        print(df_dock.head(5).to_string())
        
        # Identify score column
        score_cols = [c for c in df_dock.columns if any(k in c.lower() for k in ['score', 'affinity', 'vina', 'kcal'])]
        print('Score columns:', score_cols)
        if score_cols:
            col = score_cols[0]
            df_sorted = df_dock.sort_values(by=col, ascending=True)
            print(f'\nTop 20 compounds by {col}:')
            print(df_sorted.head(20).to_string())
    else:
        print('docking_scores.csv not found at', p_dock)
except Exception as e:
    print('Dock error:', e)

print('\n=== SCREENING SET COMPOUNDS ===')
try:
    p_comp = os.path.join(base, '03_compounds', 'phytochem_screening_set.csv')
    if os.path.exists(p_comp):
        df_comp = pd.read_csv(p_comp)
        print('Screening set shape:', df_comp.shape)
        print('Columns:', df_comp.columns.tolist())
        print(df_comp.head(5).to_string())
except Exception as e:
    print('Comp error:', e)

print('\n=== INTERACTION FINGERPRINT MATRIX ===')
try:
    p_ifp = os.path.join(base, '09_results', 'interaction_fingerprint_matrix.csv')
    if os.path.exists(p_ifp):
        df_ifp = pd.read_csv(p_ifp)
        print('IFP shape:', df_ifp.shape)
        print('Columns:', df_ifp.columns.tolist())
        print(df_ifp.head(10).to_string())
except Exception as e:
    print('IFP error:', e)
