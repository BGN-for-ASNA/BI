"""2.4 -- tools/check_compat.py: the manifests of the three repositories are
checked against compat.toml (one compatibility matrix)."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("check_compat", os.path.join(ROOT, "tools", "check_compat.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

COMPAT = {
    "python": {"version": "1.0", "python": ">=3.12", "numpyro": ">=0.21.0",
               "arviz": ">=1.2.0", "jax": ">=0.10.0"},
    "r": {"version": "0.3", "python": ">=3.12", "bayesforge": ">=1.0"},
    "julia": {"version": "0.2", "python": ">=3.12,<3.15", "bayesforge": ">=1.0"},
}


def _write(d, files):
    for path, text in files.items():
        p = os.path.join(d, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    return str(d)


def _repos(tmp_path, r_version="0.3", julia_bf=">=1.0"):
    py = _write(tmp_path / "py", {"pyproject.toml": (
        '[project]\nname="bayesforge"\nversion="1.0"\nrequires-python=">=3.12"\n'
        'dependencies=["numpyro >= 0.21.0", "arviz >= 1.2.0"]\n'
        '[project.optional-dependencies]\ncpu=["jax >= 0.10.0", "jaxlib >= 0.10.0"]\n')})
    r = _write(tmp_path / "r", {
        "DESCRIPTION": f"Package: BayesForge\nVersion: {r_version}\n",
        "R/BF_management.R": '.bf_python_version <- ">=3.12"\n.bf_bayesforge_version <- ">=1.0"\n'})
    jl = _write(tmp_path / "jl", {
        "Project.toml": 'name = "PyBayesForge"\nversion = "0.2"\n',
        "CondaPkg.toml": ('[deps]\npython = ">=3.12,<3.15"\n[pip.deps]\n'
                          f'bayesforge = {{ version = "{julia_bf}", extras = ["cpu"] }}\n')})
    return {"python": py, "r": r, "julia": jl}


def test_consistent_manifests_pass(tmp_path):
    rows = cc.check(COMPAT, _repos(tmp_path))
    assert len(rows) == 11 and all(ok for *_, ok in rows)


def test_every_disagreement_is_reported(tmp_path):
    rows = cc.check(COMPAT, _repos(tmp_path, r_version="0.4", julia_bf=">=0.9"))
    bad = {(pkg, key) for pkg, key, _, _, ok in rows if not ok}
    assert bad == {("r", "version"), ("julia", "bayesforge")}


def test_unreadable_manifest_is_reported(tmp_path):
    repos = _repos(tmp_path)
    repos["r"] = str(tmp_path / "missing")
    os.makedirs(repos["r"])
    rows = cc.check(COMPAT, repos, only=["r"])
    assert rows[0][1] == "manifest" and not rows[0][4]


def test_this_repository_agrees_with_compat_toml():
    assert cc.main(["--only", "python"]) == 0
