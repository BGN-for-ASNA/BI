"""Run the R or Julia code blocks of the "Get started" pages, page by page.

The Python blocks are covered by ``tests/test_doc_examples.py`` (pytest). This
script does the same for the R and Julia tabs, which need R (BayesForge R
package) or Julia (PyBayesForge) installed:

    python Documentation/tools/run_doc_examples.py --lang r
    python Documentation/tools/run_doc_examples.py --lang julia --julia-project path/to/project

Installation commands (install.packages, setup_env, Pkg.add, add_pip) are
skipped. Pages that are fragments get a small setup (PRE), a fit between two
fragments (BETWEEN) or a final fit (POST), so that each page runs on its own.
Exit status is non-zero when a page fails.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

START = pathlib.Path(__file__).resolve().parents[1] / "start"

SNIPPETS = {
    "r": {
        "fence": r"^```\s*\{?[rR]\}?\s*\n(.*?)^```",
        "skip": ("install.packages", "setup_env("),
        "drop_lines": ("bayesplot::",),        # optional package
        "PRE": {"Define model": '''library(BayesForge)
m <- importBF(platform = "cpu", rand_seed = 1, print_devices_found = FALSE)
X <- seq(-1, 1, length.out = 30)
m$data_on_model <- list(X = jnp$array(X), Y = jnp$array(0.5 + 2 * X))
'''},
        "BETWEEN": {"Interoperability": '''model <- function(height, weight) {
  a <- bf.dist.normal(178, 20, name = "a")
  b <- bf.dist.log_normal(0, 1, name = "b")
  s <- bf.dist.uniform(0, 50, name = "s")
  bf.dist.normal(a + b * weight, s, obs = height)
}
m$fit(model, num_samples = 100L, num_warmup = 100L, progress_bar = FALSE)
'''},
        "POST": {"Define model": '''
m$fit(model, num_samples = 100L, num_warmup = 100L, num_chains = 1L, progress_bar = FALSE)
print(m$summary())
'''},
        "suffix": ".R",
        "cmd": lambda script, args: ["Rscript", script],
    },
    "julia": {
        "fence": r"^```\s*\{?julia\}?\s*\n(.*?)^```",
        "skip": ("Pkg.add", "add_pip"),
        "drop_lines": (),
        "PRE": {"Define model": '''using PyBayesForge
m = importBF(platform = "cpu", rand_seed = 1)
Xv = collect(range(-1, 1; length = 30))
m.data_on_model = pydict(X = jnp.array(Xv), Y = jnp.array(0.5 .+ 2 .* Xv))
'''},
        "BETWEEN": {"Interoperability": '''@BF function model(weight, height)
    a = m.dist.normal(178, 20)
    b = m.dist.log_normal(0, 1)
    s = m.dist.uniform(0, 50)
    m.dist.normal(a + b * weight, s, obs = height)
end
m.fit(model, num_samples = 100, num_warmup = 100, progress_bar = false)
'''},
        "POST": {"Define model": '''
m.fit(model, num_samples = 100, num_warmup = 100, num_chains = 1, progress_bar = false)
println(m.summary())
'''},
        "suffix": ".jl",
        "cmd": lambda script, args: ["julia", f"--project={args.julia_project}", script],
    },
}


def page_script(page, cfg):
    blocks = re.findall(cfg["fence"], page.read_text(), re.S | re.M)
    blocks = [b for b in blocks if not any(s in b for s in cfg["skip"])]
    blocks = ["\n".join(l for l in b.splitlines() if not any(d in l for d in cfg["drop_lines"]))
              for b in blocks]
    if not blocks:
        return None
    sep = "\n" + cfg["BETWEEN"].get(page.stem, "") + "\n"
    return cfg["PRE"].get(page.stem, "") + sep.join(blocks) + cfg["POST"].get(page.stem, "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=sorted(SNIPPETS), required=True)
    parser.add_argument("--julia-project", default=".", help="Julia project with PyBayesForge")
    parser.add_argument("-k", default="", help="only pages whose name contains this text")
    args = parser.parse_args()
    cfg = SNIPPETS[args.lang]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for page in sorted(START.glob("*.qmd")):
            if args.k not in page.stem:
                continue
            code = page_script(page, cfg)
            if code is None:
                continue
            script = pathlib.Path(tmp) / f"page{cfg['suffix']}"
            script.write_text(code)
            proc = subprocess.run(cfg["cmd"](str(script), args), capture_output=True, text=True,
                                  env=env, cwd=tmp, timeout=1800)
            ok = proc.returncode == 0
            print(f"{'PASS' if ok else 'FAIL'}  {page.stem}", flush=True)
            if not ok:
                failed.append(page.stem)
                print("\n".join("      " + l for l in proc.stderr.strip().splitlines()[-8:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
