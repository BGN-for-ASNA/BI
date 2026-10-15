"""m.data() accepts CSV paths and in-memory tables (R/Julia data frames arrive
as pandas DataFrames)."""
import pandas as pd
import pytest

from BayesForge import bf


@pytest.fixture(scope="module")
def m():
    return bf(platform="cpu", print_devices_found=False)


def test_csv_path(m):
    df = m.data(m.load.howell1(only_path=True), sep=";")
    assert {"height", "weight"} <= set(df.columns)


def test_dataframe_is_copied(m):
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.5]})
    m.data(table)
    m.scale(["x"])
    assert table["x"].tolist() == [1.0, 2.0, 3.0]  # caller's table untouched
    assert m.df["x"].mean() == pytest.approx(0.0)


def test_dict_of_columns(m):
    m.data({"x": [1, 2], "y": [3, 4]})
    assert list(m.df.columns) == ["x", "y"]


def test_csv_kwargs_rejected_for_tables(m):
    with pytest.raises(TypeError):
        m.data(pd.DataFrame({"x": [1]}), sep=";")


def test_fit_accepts_whole_number_floats():
    # R passes 200 as 200.0; fit() must accept it (and reject 200.5).
    import jax.numpy as jnp
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    x = jnp.linspace(-1, 1, 20)

    def model(x, y):
        a = m.dist.normal(0, 1, name="a")
        m.dist.normal(a * x, 1.0, obs=y)

    m.fit(model, obs=dict(x=x, y=2 * x), num_samples=50.0, num_warmup=50.0,
          num_chains=1.0, progress_bar=False)
    assert m.posteriors["a"].shape == (50,)
    with pytest.raises(ValueError, match="whole number"):
        m.fit(model, obs=dict(x=x, y=2 * x), num_samples=50.5, progress_bar=False)
