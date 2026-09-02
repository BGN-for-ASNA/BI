"""Typed result objects returned by the Workflow class.

Mirrors the ``m.summary()`` convention of returning something table-like
with its own printable form, rather than a bare dict.
"""
from dataclasses import dataclass, field


@dataclass
class RecoveryResult:
    """Result of :meth:`Workflow.recover`."""
    table: "object"        # pandas.DataFrame, one row per simulation
    metrics: dict           # {param_name: {bias, rmse, r2, coverage, wrg, grade}}
    param_names: list
    hdi_prob: float = 0.89
    n_jobs: int = 1

    def summary(self) -> str:
        lines = [f"Recovery over {len(self.table)} simulation(s) "
                 f"({self.hdi_prob:.0%} HDI, n_jobs={self.n_jobs}):"]
        for name in self.param_names:
            s = self.metrics[name]
            lines.append(
                f"  {name}: bias={s['bias']:.3g}  rmse={s['rmse']:.3g}  "
                f"R2={s['r2']:.3f}  coverage={s['coverage']:.2f}  "
                f"WRG={s['wrg']:.2f}  grade={s['grade']}"
            )
        failing = [n for n in self.param_names if self.metrics[n]["grade"] == "F"]
        if failing:
            lines.append(f"  FAILED (WRG<0.75): {', '.join(failing)} -- "
                          "do not trust this model yet; revisit priors/parameterization "
                          "(see iterative_model_improvement workflow).")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class SBCResult:
    """Result of :meth:`Workflow.sbc`."""
    table: "object"        # pandas.DataFrame, one row per SBC replication
    uniformity: dict        # {param_name: {statistic, p_value, n_bins}}
    param_names: list
    n_post_draws: int
    n_jobs: int = 1

    def summary(self) -> str:
        lines = [f"SBC over {len(self.table)} replication(s) "
                 f"({self.n_post_draws} posterior draws each, n_jobs={self.n_jobs}):"]
        for name in self.param_names:
            u = self.uniformity[name]
            verdict = "PASS" if (u["p_value"] == u["p_value"] and u["p_value"] > 0.05) else "FAIL"
            lines.append(f"  {name}: chi2={u['statistic']:.2f}  p={u['p_value']:.3f}  "
                         f"[{verdict}]")
        failing = [n for n in self.param_names
                   if self.uniformity[n]["p_value"] == self.uniformity[n]["p_value"]
                   and self.uniformity[n]["p_value"] <= 0.05]
        if failing:
            lines.append(f"  Non-uniform ranks for: {', '.join(failing)} -- "
                          "the fitting procedure is miscalibrated for these parameters; "
                          "check model code and priors before trusting it on real data.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class ContrastResult:
    """Result of :meth:`Workflow.contrast`, :meth:`Workflow.poststratify`."""
    name: str
    mean: float
    hdi_lo: float
    hdi_hi: float
    p_positive: float
    hdi_prob: float = 0.89

    def summary(self) -> str:
        return (f"{self.name}: mean={self.mean:.4g}  "
                f"{self.hdi_prob:.0%} HDI=[{self.hdi_lo:.4g}, {self.hdi_hi:.4g}]  "
                f"P(>0)={self.p_positive:.3f}")

    def __repr__(self) -> str:
        return self.summary()
