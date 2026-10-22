"""Check the three BayesForge repositories against compat.toml.

Usage:
    python tools/check_compat.py [--python DIR] [--r SRC] [--julia SRC] [--only python,r,julia]

Each SRC is a local checkout (directory) or a GitHub branch written
owner/repo@branch (read from raw.githubusercontent.com). Defaults: this
repository for Python, BGN-for-ASNA/BFR@dev and BGN-for-ASNA/PyBayesForge.jl@dev.
compat.toml is read from the --python checkout. Prints one table row per
checked item and exits with 1 if any manifest disagrees with compat.toml.
"""
import argparse
import os
import re
import sys
import tomllib
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reader(src):
    """Return read(path) -> text for a local directory or owner/repo@branch."""
    if os.path.isdir(src):
        return lambda path: open(os.path.join(src, path), encoding="utf-8").read()
    repo, _, branch = src.partition("@")
    base = f"https://raw.githubusercontent.com/{repo}/{branch or 'main'}/"
    return lambda path: urllib.request.urlopen(base + path, timeout=30).read().decode()


def _dep(deps, name):
    for d in deps:
        m = re.match(rf"\s*{re.escape(name)}\s*(\[[^\]]*\])?\s*(.*)$", d)
        if m:
            return m.group(2).replace(" ", "")
    return None


def python_manifest(read):
    p = tomllib.loads(read("pyproject.toml"))["project"]
    deps = p.get("dependencies", [])
    return {
        "version": p["version"],
        "python": p["requires-python"].replace(" ", ""),
        "numpyro": _dep(deps, "numpyro"),
        "arviz": _dep(deps, "arviz"),
        "jax": _dep(p.get("optional-dependencies", {}).get("cpu", []), "jax"),
    }


def r_manifest(read):
    desc = read("DESCRIPTION")
    mgmt = read("R/BF_management.R")

    def r_string(var):
        m = re.search(rf"^{re.escape(var)}\s*<-\s*\"([^\"]*)\"", mgmt, re.M)
        return m.group(1).replace(" ", "") if m else None

    return {
        "version": re.search(r"^Version:\s*(\S+)", desc, re.M).group(1),
        "python": r_string(".bf_python_version"),
        "bayesforge": r_string(".bf_bayesforge_version"),
    }


def julia_manifest(read):
    project = tomllib.loads(read("Project.toml"))
    conda = tomllib.loads(read("CondaPkg.toml"))
    bf = conda.get("pip", {}).get("deps", {}).get("bayesforge")
    if isinstance(bf, dict):
        bf = bf.get("version")
    return {
        "version": project["version"],
        "python": conda.get("deps", {}).get("python", "").replace(" ", ""),
        "bayesforge": bf.replace(" ", "") if bf else None,
    }


MANIFESTS = {"python": python_manifest, "r": r_manifest, "julia": julia_manifest}


def check(compat, sources, only=("python", "r", "julia")):
    """Return a list of (package, key, expected, found, ok) rows."""
    rows = []
    for pkg in only:
        try:
            found = MANIFESTS[pkg](reader(sources[pkg]))
        except Exception as e:                      # unreadable manifest
            rows.append((pkg, "manifest", sources[pkg], f"error: {e}", False))
            continue
        for key, expected in compat[pkg].items():
            expected = str(expected).replace(" ", "")
            got = found.get(key)
            rows.append((pkg, key, expected, got, got == expected))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", default=HERE)
    ap.add_argument("--r", default="BGN-for-ASNA/BFR@dev")
    ap.add_argument("--julia", default="BGN-for-ASNA/PyBayesForge.jl@dev")
    ap.add_argument("--only", default="python,r,julia")
    args = ap.parse_args(argv)
    # Windows consoles default to cp1252, which cannot print the ✅/❌ marks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    compat = tomllib.loads(reader(args.python)("compat.toml"))
    rows = check(compat, {"python": args.python, "r": args.r, "julia": args.julia},
                 only=[s.strip() for s in args.only.split(",") if s.strip()])
    print("| package | item | compat.toml | manifest | ok |\n|---|---|---|---|---|")
    for pkg, key, exp, got, ok in rows:
        print(f"| {pkg} | {key} | {exp} | {got} | {'✅' if ok else '❌'} |")
    bad = [r for r in rows if not r[4]]
    if bad:
        print(f"\n{len(bad)} item(s) disagree with compat.toml.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
