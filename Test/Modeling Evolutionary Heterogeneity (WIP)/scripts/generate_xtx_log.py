import pandas as pd
import datetime
import os

def generate_xtx(comparison_csv, output_dir, params, model_name, plot_file):
    if not os.path.exists(comparison_csv):
        return
        
    df = pd.read_csv(comparison_csv)
    df = df[df['Parameter'].isin(params)]
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = os.path.join(output_dir, "log.xtx")
    
    with open(output_path, "w") as f:
        f.write(f"% --- BI vs BEAST Diagnostic Log: {model_name} ---\n")
        f.write(f"% Generated: {now}\n\n")
        
        f.write(f"\\section{{Phylogenetic {model_name} Comparison}}\n")
        f.write("Comparison of posterior parameter recovery.\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|ccc}\n")
        f.write("\\hline\n")
        f.write("Parameter & BI Mean (SD) & BEAST Mean (SD) & Diff (\\%) \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            p = row['Parameter'].replace('_', '\\_')
            bi = row['BI Mean (SD)']
            beast = row['BEAST Mean (SD)']
            diff = row['Diff (%)']
            f.write(f"{p} & {bi} & {beast} & {diff} \\\\\n")
            
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Posterior estimation results for {model_name}.}}\n")
        f.write("\\end{table}\n\n")
        
        f.write("\\subsection{Diagnostic Plots}\n")
        f.write(f"Overlap density plots: \\texttt{{{plot_file}}}.\n")

if __name__ == "__main__":
    # Model 1
    generate_xtx("BI_vs_BEAST_comparison.csv", 
                 "Model_1_Spatial_Heterogeneity", 
                 ['kappa', 'alpha'], 
                 "Spatial Heterogeneity (+Gamma)",
                 "density_gamma.png")
    
    # Model 2
    generate_xtx("BI_vs_BEAST_comparison.csv", 
                 "Model_2_Temporal_Heterogeneity", 
                 ['kappa', 'alpha', 'mu_c', 'sigma_c'], 
                 "Temporal Heterogeneity (UCLN Revised)",
                 "density_ucln.png")
    
    # Root summary
    generate_xtx("BI_vs_BEAST_comparison.csv", 
                 ".", 
                 ['kappa', 'alpha', 'mu_c', 'sigma_c'], 
                 "Evolutionary Heterogeneity Full",
                 "density_posteriors_comparison.png")
    
    print("Distributed log.xtx files generated in respective folders.")
