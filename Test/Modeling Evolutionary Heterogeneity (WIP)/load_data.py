import re
import numpy as np

def parse_nexus(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple regex to find the matrix block
    matrix_match = re.search(r'matrix\s*(.*?)\s*;', content, re.DOTALL | re.IGNORECASE)
    if not matrix_match:
        raise ValueError("Could not find matrix block in NEXUS file")
    
    matrix_text = matrix_match.group(1)
    # Remove lines starting with [ and ending with ] (comments/info)
    matrix_text = re.sub(r'\[.*?\]', '', matrix_text)
    
    lines = matrix_text.strip().split('\n')
    taxa_data = {}
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'N': 4, '?': 4}
    
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split(None, 1)
        if len(parts) < 2: continue
        
        taxon = parts[0]
        seq_str = parts[1].upper().replace(' ', '')
        
        # Handle matchchar '.' if necessary (standard in Nexus)
        # Assuming the first taxon is the reference
        if not taxa_data:
            ref_seq = seq_str
            encoded = []
            for char in seq_str:
                encoded.append(base_map.get(char, 4))
            taxa_data[taxon] = encoded
        else:
            encoded = []
            for i, char in enumerate(seq_str):
                if char == '.':
                    encoded.append(taxa_data[list(taxa_data.keys())[0]][i])
                else:
                    encoded.append(base_map.get(char, 4))
            taxa_data[taxon] = encoded
            
    # Convert to one-hot (excluding gaps for likelihood)
    N_taxa = len(taxa_data)
    L = len(list(taxa_data.values())[0])
    
    one_hot = np.zeros((N_taxa, L, 4))
    taxa_names = list(taxa_data.keys())
    
    for i, name in enumerate(taxa_names):
        seq = taxa_data[name]
        for j, base_idx in enumerate(seq):
            if base_idx < 4:
                one_hot[i, j, base_idx] = 1.0
            else:
                # Ambiguous/Gap: equal probability for all 4 bases
                one_hot[i, j, :] = 0.25
                
    return taxa_names, one_hot

if __name__ == "__main__":
    names, data = parse_nexus("primate-mtDNA.nexus")
    print(f"Parsed {len(names)} taxa: {names}")
    print(f"Data shape: {data.shape}")
    np.save("primate_data.npy", data)
    with open("taxa_names.txt", "w") as f:
        f.write("\n".join(names))
