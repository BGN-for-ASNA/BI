"""2.3 -- categorical columns (R factors arrive as pandas Categorical) are
indexed by m.index(), keeping the level order."""
import pandas as pd

from BayesForge import bf


def test_index_all_includes_categorical_and_keeps_level_order():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    df = pd.DataFrame({
        "g": pd.Categorical(["b", "a", "c", "a"], categories=["c", "a", "b"]),
        "site": ["x", "y", "x", "z"],          # text column
        "y": [1.5, 2.0, 3.0, 4.0],
    })
    m.data(df)
    m.index()
    assert m.df["index_g"].tolist() == [2, 1, 0, 1]          # c=0, a=1, b=2 (factor order)
    assert m.index_map["g"] == {0: "c", 1: "a", 2: "b"}
    assert m.df["index_site"].tolist() == [0, 1, 0, 2]        # alphabetical for text
    assert m.index_map["site"] == {0: "x", 1: "y", 2: "z"}
