import pandas as pd
import matplotlib.pyplot as plt
import re

log_path = "/home/sebastian_sosa/BI/Test/Network/Network edges/results/binary_full_directed_no_zi/binary_full_directed_no_zi_log.txt"
output_svg = "/home/sebastian_sosa/BI/Test/Network/Network edges/results/binary_full_directed_no_zi/summary_kl_comparison.svg"

data = []
with open(log_path, 'r') as f:
    lines = f.readlines()
    start_parsing = False
    for line in lines:
        if "-----------------------------------------------------------------------------------------------" in line:
            start_parsing = True
            continue
        if start_parsing:
            if "=== Summary KL stats ===" in line or not line.strip():
                break
            # Use regex to split by 2 or more spaces
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 5:
                param = parts[0]
                stan_mean = float(parts[1])
                bi_mean = float(parts[2])
                diff = float(parts[3])
                kl = float(parts[4])
                data.append({"Parameter": param, "Stan_mean": stan_mean, "BI_mean": bi_mean, "KL": kl})

df = pd.DataFrame(data)

# Sort by KL divergence and take top 20
df_top = df.sort_values(by="KL", ascending=False).head(20)

# Also include hyperpriors if not in top 20
hyperpriors = df[df['Parameter'].str.contains('random_group|edge_sigma')]
df_summary = pd.concat([df_top, hyperpriors]).drop_duplicates().sort_values(by="KL", ascending=False)

plt.figure(figsize=(12, 10))
y_pos = range(len(df_summary))

plt.barh([y - 0.2 for y in y_pos], df_summary['Stan_mean'], height=0.4, label='Stan Mean', color='#3498db', alpha=0.8)
plt.barh([y + 0.2 for y in y_pos], df_summary['BI_mean'], height=0.4, label='BI Mean', color='#e74c3c', alpha=0.8)

plt.yticks(y_pos, df_summary['Parameter'])
plt.xlabel('Mean Value')
plt.title('Comparison of Stan vs BI Mean Estimates (Highest KL Divergence + Hyperpriors)')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add KL values as text labels
for i, kl in enumerate(df_summary['KL']):
    plt.text(max(df_summary['Stan_mean'].iloc[i], df_summary['BI_mean'].iloc[i]) + 0.05, i, f'KL: {kl:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_svg)
print(f"Summary SVG generated at: {output_svg}")
