"""Run both causal-DAG and model-graph test suites and report a combined total.

    python "Test/Causal & DAG/run_all.py"

(pytest is optional: ``pytest "Test/Causal & DAG"`` works too.)
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(fname):
    print(f"\n=== {fname} ===")
    try:
        runpy.run_path(os.path.join(HERE, fname), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    rc = 0
    for f in ("test_causal_dag.py", "test_model_graph.py"):
        rc |= _run(f)
    print("\n==============================")
    print("ALL SUITES PASSED" if rc == 0 else "SOME SUITES FAILED")
    sys.exit(rc)
