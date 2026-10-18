import csv
import os

pkg_dir = 'guide_packets/Package_A_PPARg_Diabetes'

def parse_csv_stats(filepath, time_col=0, val_cols=[1], skip_start_time=20.0):
    if not os.path.exists(filepath):
        return None
    data = {c: [] for c in val_cols}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[time_col])
                if t >= skip_start_time:
                    for c in val_cols:
                        data[c].append(float(row[c]))
            except (ValueError, IndexError):
                continue
    stats = {}
    for c, vals in data.items():
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = var ** 0.5
            stats[c] = (mean, std)
    return stats, header

print("=== GROMACS MD TRAJECTORY EQUILIBRIUM STATISTICS (20-100 ns) ===")
for sys_name in ['reference', 'ergosterol']:
    print(f"\n--- System: {sys_name.upper()} ---")
    sys_dir = os.path.join(pkg_dir, 'md_data', sys_name)
    
    # RMSD
    rmsd_file = os.path.join(sys_dir, 'rmsd.csv')
    res, hdr = parse_csv_stats(rmsd_file, 0, [1, 2], 20.0)
    if res:
        print(f"Backbone RMSD: {res[1][0]:.3f} +/- {res[1][1]:.3f} nm ({res[1][0]*10:.2f} +/- {res[1][1]*10:.2f} A)")
        if 2 in res:
            print(f"Ligand RMSD: {res[2][0]:.3f} +/- {res[2][1]:.3f} nm ({res[2][0]*10:.2f} +/- {res[2][1]*10:.2f} A)")
            
    # Gyration
    rg_file = os.path.join(sys_dir, 'gyrate.csv')
    res, hdr = parse_csv_stats(rg_file, 0, [1], 20.0)
    if res:
        print(f"Radius of Gyration (Rg): {res[1][0]:.3f} +/- {res[1][1]:.3f} nm ({res[1][0]*10:.2f} +/- {res[1][1]*10:.2f} A)")
        
    # SASA
    sasa_file = os.path.join(sys_dir, 'sasa.csv')
    res, hdr = parse_csv_stats(sasa_file, 0, [1], 20.0)
    if res:
        print(f"Total SASA: {res[1][0]:.1f} +/- {res[1][1]:.1f} nm^2 ({res[1][0]*100:.1f} +/- {res[1][1]*100:.1f} A^2)")
        
    # Hbonds
    hb_file = os.path.join(sys_dir, 'hbonds.csv')
    res, hdr = parse_csv_stats(hb_file, 0, [1], 20.0)
    if res:
        print(f"Intermolecular H-bonds: {res[1][0]:.2f} +/- {res[1][1]:.2f}")
        
    # MMGBSA
    mmgbsa_file = os.path.join(sys_dir, 'FINAL_RESULTS_MMGBSA.csv')
    if os.path.exists(mmgbsa_file):
        with open(mmgbsa_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            print("MM-GBSA Results sample lines:")
            for l in lines[:15]:
                print(" ", l)
