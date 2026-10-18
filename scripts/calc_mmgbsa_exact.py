import csv
import os

def calculate_mmgbsa_stats(filepath, start_frame=200):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Locate "Delta Energy Terms"
    delta_idx = -1
    for i, line in enumerate(lines):
        if "delta energy terms" in line.lower():
            delta_idx = i
            break
            
    if delta_idx == -1:
        return None
        
    reader = csv.DictReader(lines[delta_idx+1:])
    terms = ['VDWAALS', 'EEL', 'EGB', 'ESURF', 'GGAS', 'GSOLV', 'TOTAL']
    data = {t: [] for t in terms}
    
    for row in reader:
        try:
            frame = int(row['Frame #'])
            if frame >= start_frame:
                for t in terms:
                    data[t].append(float(row[t]))
        except (ValueError, KeyError):
            continue
            
    results = {}
    for t in terms:
        vals = data[t]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean)**2 for x in vals) / len(vals)
            std = var ** 0.5
            results[t] = (mean, std)
    return results

print("=== ACTUAL 100 ns MM-GBSA BINDING FREE ENERGY (20-100 ns equilibrium) ===")
for sys_name in ['reference', 'ergosterol']:
    path = f'guide_packets/Package_A_PPARg_Diabetes/md_data/{sys_name}/FINAL_RESULTS_MMGBSA.csv'
    res = calculate_mmgbsa_stats(path, 200)
    print(f"\n--- {sys_name.upper()} ---")
    if res:
        print(f"Van der Waals (dVDW):     {res['VDWAALS'][0]:6.2f} +/- {res['VDWAALS'][1]:4.2f} kcal/mol")
        print(f"Electrostatic (dEEL):     {res['EEL'][0]:6.2f} +/- {res['EEL'][1]:4.2f} kcal/mol")
        print(f"Polar Solvation (dEGB):   {res['EGB'][0]:6.2f} +/- {res['EGB'][1]:4.2f} kcal/mol")
        print(f"Non-polar Solv (dESURF):  {res['ESURF'][0]:6.2f} +/- {res['ESURF'][1]:4.2f} kcal/mol")
        print(f"Gas-phase Energy (dGGAS): {res['GGAS'][0]:6.2f} +/- {res['GGAS'][1]:4.2f} kcal/mol")
        print(f"Solvation Energy (dGSOLV):{res['GSOLV'][0]:6.2f} +/- {res['GSOLV'][1]:4.2f} kcal/mol")
        print(f"TOTAL BINDING FREE ENERGY (dTOTAL): {res['TOTAL'][0]:6.2f} +/- {res['TOTAL'][1]:4.2f} kcal/mol")
