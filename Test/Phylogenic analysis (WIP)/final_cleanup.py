import os
import shutil

models = {
    "Model_1_Simple": {
        "patterns": ["simple", "L_simple", "data_simple", "brms_fit_simple", "brms_post_simple", "model_simple.rds"],
        "extra": ["phylo.nex"]
    },
    "Model_2_Poisson": {
        "patterns": ["pois", "L_pois", "data_pois", "model_pois.rds", "output_pois.log"],
        "extra": ["phylo.nex"]
    },
    "Model_3_Repeated": {
        "patterns": ["repeat", "L_repeat", "data_repeat", "model_repeat.stan", "get_stan_repeat", "save_L.R"],
        "extra": ["phylo.nex"]
    },
    "Model_4_Meta": {
        "patterns": ["meta", "L_meta", "bi_post_meta", "model_meta.stan", "get_stan_meta"],
        "extra": ["phylo.nex"]
    },
    "Model_6_Slopes": {
        "patterns": ["slopes", "data_slopes", "phylo_slopes", "sim_data_effect", "model_slopes.stan", "get_stan_slopes", "save_L_slopes"],
        "extra": []
    }
}

base_dir = r"c:\Users\Sosa\Documents\BI\Test\Phylogenic analysis (WIP)"

# 1. Create subfolders and move files
for folder, config in models.items():
    dest_path = os.path.join(base_dir, folder)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # Patterns
    for f in os.listdir(base_dir):
        # Skip directories
        if os.path.isdir(os.path.join(base_dir, f)):
            continue
            
        for pat in config["patterns"]:
            if pat in f:
                dest_file = os.path.join(dest_path, f)
                if not os.path.exists(dest_file):
                    shutil.move(os.path.join(base_dir, f), dest_file)
                else:
                    # If already exists (from previous copy), just delete from root
                    try:
                        os.remove(os.path.join(base_dir, f))
                    except:
                        pass
                break
    
    # Extra files (copy them if shared)
    for extra in config["extra"]:
        src_extra = os.path.join(base_dir, extra)
        if os.path.exists(src_extra):
            shutil.copy(src_extra, os.path.join(dest_path, extra))

# 2. Final root cleanup
# Delete phylo.nex from root after copies are done
if os.path.exists(os.path.join(base_dir, "phylo.nex")):
    os.remove(os.path.join(base_dir, "phylo.nex"))

# Remove plots directory
plots_dir = os.path.join(base_dir, "plots")
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)

# Handle remaining specific files
remaining_to_delete = ["log.txt", "output.log", "SIM.R", "test_shapes.R", "test_trunc.R", "inspect_data.R", "save_L_base.R", "convert_pdf.py", "COMPARISON.md"]
for f in remaining_to_delete:
    p = os.path.join(base_dir, f)
    if os.path.exists(p):
        os.remove(p)

print("Root directory cleaned. All files moved to model subfolders.")
