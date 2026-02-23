import json

with open(r'c:\Users\Sosa\Documents\BI\Test\1.Rethinking_np.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

out = []
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', [])).strip()
    if 'code' not in cell['cell_type']: continue
    lower_src = src.lower()
    if 'stan' in lower_src and 'import' not in lower_src and 'library' not in lower_src:
        out.append(f"----- Cell {idx} (len {len(src)}) -----\n" + src)

with open('preview.txt', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(out))
