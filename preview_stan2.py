import json

with open(r'c:\Users\Sosa\Documents\BI\Test\1.Rethinking_np.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

out = []
stan_cells = []
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if 'STAN' in src or 'stan' in src.lower():
        # Let's just print surrounding cells to understand where the stan code is
        pass
    if cell['cell_type'] == 'markdown' and 'STAN' in src:
        out.append(f"MARKDOWN: {src}")
        # The next cell is likely the STAN code
        if idx + 1 < len(nb['cells']):
            next_cell = nb['cells'][idx+1]
            out.append(f"NEXT_CELL ({next_cell['cell_type']}): " + "".join(next_cell.get('source', [])))
            stan_cells.append(idx+1)

with open('preview_stan_cells.txt', 'w', encoding='utf-8') as f:
    f.write("\n\n---\n\n".join(out))
