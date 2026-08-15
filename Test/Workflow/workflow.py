#%%
# BF_script_to_graph_traced.py

from __future__ import annotations

import ast
import os
import graphviz
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx


# =========================================================
# Node
# =========================================================

@dataclass
class Node:
    id: str
    kind: str
    label: str
    source: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# AST HELPERS
# =========================================================

def unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def call_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return call_path(node.func)
    return ""


def target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return f"{target_name(target.value)}.{target.attr}"
    return unparse(target)


def collect_names(expr: ast.AST) -> Set[str]:
    names = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node):
            names.add(node.id)

    V().visit(expr)
    return names


# =========================================================
# GRAPH BUILDER
# =========================================================

class ScriptGraphBuilder:
    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)

        self.nodes: List[Node] = []
        self.g = nx.DiGraph()

        self.counter = 0
        self.var_last_node: Dict[str, str] = {}
        self.data_profiles: Dict[str, Any] = {}
        self.summary_output: Optional[str] = None
        self.latex_output: Optional[str] = None
        self.real_model = None
        self.model_args = {}
        self.fitted_BF = None  # BF model object after fit, used for diagWIP
        self.ppc_samples = None

        # runtime hook
        self.model_fn = None

    def new_id(self):
        self.counter += 1
        return f"n{self.counter:03d}"

    def add_node(self, kind, label, source, inputs=None, outputs=None, meta=None):
        node = Node(
            id=self.new_id(),
            kind=kind,
            label=label,
            source=source,
            inputs=inputs or [],
            outputs=outputs or [],
            meta=meta or {},
        )

        self.nodes.append(node)
        self.g.add_node(node.id, data=node)

        for inp in node.inputs:
            if inp in self.var_last_node:
                self.g.add_edge(self.var_last_node[inp], node.id, label=inp)

        for out in node.outputs:
            self.var_last_node[out] = node.id

        return node

    # -------------------------
    # runtime hook
    # -------------------------
    def add_runtime_event(self, kind, label=None, inputs=None, outputs=None, payload=None):
        node = self.add_node(
            kind=kind,
            label=label or kind,
            source=f"Runtime: {kind}",
            inputs=inputs,
            outputs=outputs,
            meta={"runtime": True, "payload": payload},
        )
        return node

    def analyze_model_archetype(self):
        """Programmatically evaluate the model type and suggest a workflow."""
        if not self.nodes:
            return "Unknown", "No model structure detected."
        
        causal = self.analyze_causal_structure()
        outcome = causal.get("outcome")
        predictors = causal.get("predictors", [])
        latents = causal.get("latents", [])
        
        # Check for hierarchical structure (plates/subgraphs in the original model)
        is_hierarchical = any("plate" in n.label.lower() for n in self.nodes) or \
                         (len(latents) > 10) # Heuristic for complex models
        
        if is_hierarchical:
            return "Hierarchical/Mixed Model", "Evaluate group-level shrinkage. Compare group-level priors against the pooled estimate to check for partial pooling."
        
        if len(predictors) == 0 and len(latents) > 0:
            return "Purely Generative/Prior Model", "Perform Prior Predictive Checks to ensure the priors are reasonably aligned with domain knowledge."
            
        if len(predictors) > 0:
            if len(predictors) < 5:
                return "Simple Regression Model", "Run Posterior Predictive Checks (PPC) to verify if the model captures the data variance and skewness."
            else:
                return "High-Dimensional Model", "Consider Regularizing Priors (e.g., Horseshoe or Laplace) to prevent overfitting given the number of predictors."
        
        return "General Probabilistic Model", "Explore the joint posterior distribution of parameters to identify correlations."

    def analyze_diagnostics(self):
        """Analyze MCMC summary statistics for convergence and health."""
        if not self.summary_output:
            return "⚠️ Diagnostics Missing: You have fitted a model, but no summary was generated. We suggest running `m.summary()` to check for convergence (R-hat < 1.1) and sufficient Effective Sample Size (ESS)."

        try:
            lines = self.summary_output.splitlines()
            header_line = next((l for l in lines if "mean" in l and "r_hat" in l), None)
            if not header_line:
                return "✅ Summary generated, but R-hat column not found. Convergence cannot be programmatically verified."

            cols = header_line.split()
            # Data rows have one extra leading token (param name) not present in header.
            # All col indices must be offset by +1 when indexing into data row parts.
            r_hat_idx   = cols.index("r_hat")
            ess_bulk_idx = cols.index("ess_bulk") if "ess_bulk" in cols else (cols.index("ess") if "ess" in cols else None)
            ess_tail_idx = cols.index("ess_tail") if "ess_tail" in cols else None
            mcse_mean_idx = cols.index("mcse_mean") if "mcse_mean" in cols else None
            sd_idx       = cols.index("sd") if "sd" in cols else None

            issues = []
            for line in lines:
                parts = line.split()
                # Data rows have len(cols)+1 tokens; skip header, blank, and footer lines
                if len(parts) != len(cols) + 1:
                    continue
                # Skip rows where first token is not a valid parameter name
                try:
                    float(parts[0])
                    continue  # first token is numeric → not a param row
                except ValueError:
                    pass

                param = parts[0]

                # R-hat check (threshold 1.01 strict, warn at 1.05)
                try:
                    r_hat = float(parts[r_hat_idx + 1])
                    if r_hat > 1.1:
                        issues.append(f"Parameter `{param}` did NOT converge (R-hat = {r_hat:.3f} > 1.1)")
                    elif r_hat > 1.01:
                        issues.append(f"Parameter `{param}` borderline convergence (R-hat = {r_hat:.3f}, ideal < 1.01)")
                except (ValueError, IndexError):
                    pass

                # ESS bulk check
                if ess_bulk_idx is not None:
                    try:
                        ess_bulk = float(parts[ess_bulk_idx + 1])
                        if ess_bulk < 400:
                            issues.append(f"Parameter `{param}` low ESS bulk (ESS = {ess_bulk:.0f} < 400)")
                    except (ValueError, IndexError):
                        pass

                # ESS tail check (important for HDI/quantile reliability)
                if ess_tail_idx is not None:
                    try:
                        ess_tail = float(parts[ess_tail_idx + 1])
                        if ess_tail < 400:
                            issues.append(f"Parameter `{param}` low ESS tail (ESS_tail = {ess_tail:.0f} < 400) — tail quantiles unreliable")
                    except (ValueError, IndexError):
                        pass

                # MCSE/SD ratio check: mcse_mean > 10% of SD signals noisy estimates
                if mcse_mean_idx is not None and sd_idx is not None:
                    try:
                        mcse = float(parts[mcse_mean_idx + 1])
                        sd   = float(parts[sd_idx + 1])
                        if sd > 0 and mcse / sd > 0.1:
                            issues.append(f"Parameter `{param}` high MCSE (MCSE/SD = {mcse/sd:.2f} > 0.10) — run more samples")
                    except (ValueError, IndexError):
                        pass

            if not issues:
                return "✅ Convergence Check: All parameters converged (R-hat < 1.01), ESS bulk/tail ≥ 400, MCSE/SD < 0.10."
            else:
                return "⚠️ Diagnostics Warning:\n" + "\n".join([f"- {iss}" for iss in issues])

        except Exception as e:
            return f"Could not parse diagnostics: {e}"

    def analyze_sampler_diagnostics(self):
        """Run programmatic diagnostics via diagWIP (rhat, ess, divergences, BFMI)."""
        if not self.fitted_BF:
            return None

        import numpy as np
        sections = []

        # Resolve the raw NumPyro MCMC sampler from the BF object
        sampler = None
        for attr in ("sampler", "mcmc", "_sampler", "_mcmc"):
            if hasattr(self.fitted_BF, attr):
                sampler = getattr(self.fitted_BF, attr)
                break
        if sampler is None:
            return "_Sampler object not accessible on BF model — skipping advanced diagnostics._"

        try:
            from BayesForge.Diagnostic.Diag2 import diagWIP
            diag = diagWIP(sampler)
            diag.to_az()

            # --- R-hat via ArviZ ---
            try:
                rhat_ds = diag.rhat()
                bad_rhat = []
                for var in rhat_ds.data_vars:
                    for v in np.array(rhat_ds[var]).flatten():
                        if not np.isnan(v) and v > 1.01:
                            bad_rhat.append(f"`{var}`: {v:.4f}")
                if bad_rhat:
                    sections.append("**R-hat (ArviZ)** ⚠️\n" + "\n".join(f"  - {x}" for x in bad_rhat))
                else:
                    sections.append("**R-hat (ArviZ)**: All < 1.01 ✅")
            except Exception as e:
                sections.append(f"**R-hat**: error — {e}")

            # --- ESS bulk via ArviZ ---
            try:
                ess_ds = diag.ess()
                low_ess = []
                for var in ess_ds.data_vars:
                    for v in np.array(ess_ds[var]).flatten():
                        if not np.isnan(v) and v < 400:
                            low_ess.append(f"`{var}`: {v:.0f}")
                if low_ess:
                    sections.append("**ESS bulk (ArviZ)** ⚠️\n" + "\n".join(f"  - {x}" for x in low_ess))
                else:
                    sections.append("**ESS bulk (ArviZ)**: All ≥ 400 ✅")
            except Exception as e:
                sections.append(f"**ESS bulk**: error — {e}")

            # --- ESS tail via ArviZ ---
            try:
                ess_tail_ds = diag.ess(kind="tail")
                low_tail = []
                for var in ess_tail_ds.data_vars:
                    for v in np.array(ess_tail_ds[var]).flatten():
                        if not np.isnan(v) and v < 400:
                            low_tail.append(f"`{var}`: {v:.0f}")
                if low_tail:
                    sections.append("**ESS tail (ArviZ)** ⚠️ — tail quantiles/HDI unreliable\n" + "\n".join(f"  - {x}" for x in low_tail))
                else:
                    sections.append("**ESS tail (ArviZ)**: All ≥ 400 ✅")
            except Exception as e:
                sections.append(f"**ESS tail**: error — {e}")

        except Exception as e:
            sections.append(f"**diagWIP init failed**: {e}")
            sampler = None  # fall through to raw checks

        # --- Divergences (raw extra fields) ---
        if sampler is not None:
            try:
                extra = sampler.get_extra_fields()
                if "diverging" in extra:
                    divs = int(np.sum(extra["diverging"]))
                    total = int(extra["diverging"].size)
                    pct = 100 * divs / max(total, 1)
                    if divs > 0:
                        sections.append(f"**Divergences**: {divs}/{total} ({pct:.1f}%) ⚠️ — reparameterize or tighten priors")
                    else:
                        sections.append(f"**Divergences**: 0/{total} ✅")
                else:
                    sections.append("**Divergences**: not recorded (rerun with `extra_fields=('diverging',)`)")
            except Exception as e:
                sections.append(f"**Divergences**: error — {e}")

            # --- BFMI from energy ---
            try:
                extra = sampler.get_extra_fields()
                if "energy" in extra:
                    energy = np.array(extra["energy"])
                    if energy.ndim == 1:
                        energy = energy[np.newaxis, :]
                    bfmis = [np.var(np.diff(energy[c])) / np.var(energy[c]) for c in range(energy.shape[0])]
                    min_bfmi = float(np.min(bfmis))
                    bfmi_str = ", ".join(f"{v:.3f}" for v in bfmis)
                    if min_bfmi < 0.3:
                        sections.append(f"**BFMI**: [{bfmi_str}] ⚠️ — min {min_bfmi:.3f} < 0.3, sampler geometry poor")
                    else:
                        sections.append(f"**BFMI**: [{bfmi_str}] ✅")
                else:
                    sections.append("**BFMI**: not recorded (rerun with `extra_fields=('energy',)`)")
            except Exception as e:
                sections.append(f"**BFMI**: error — {e}")

            # --- NUTS max tree depth saturation ---
            try:
                extra = sampler.get_extra_fields()
                if "num_steps" in extra:
                    steps = np.array(extra["num_steps"])
                    max_steps = int(np.max(steps))
                    saturated = int(np.sum(steps == max_steps))
                    total = steps.size
                    pct = 100 * saturated / max(total, 1)
                    if pct > 1.0:
                        sections.append(f"**NUTS tree depth**: {saturated}/{total} ({pct:.1f}%) hit max ({max_steps} steps) ⚠️ — increase `max_tree_depth`")
                    else:
                        sections.append(f"**NUTS tree depth**: max depth not saturated ✅")
            except Exception:
                pass

        return "\n\n".join(sections) if sections else None

    def analyze_ppc(self):
        """Summarize PPC samples vs observed data as markdown text."""
        if not self.ppc_samples:
            return None
        if "_error" in self.ppc_samples:
            return f"⚠️ PPC failed: {self.ppc_samples['_error']}"

        import numpy as np

        causal = self.analyze_causal_structure()
        outcome = causal.get("outcome")
        if not outcome:
            return "_Outcome not identified — cannot compare PPC to observed._"

        observed = self.model_args.get(outcome)
        if observed is None:
            return "_Observed data not in model_args — cannot compare._"
        observed = np.array(observed)

        # Find the predictive site: prefer one matching outcome name, else last site
        ppc_key = next(
            (k for k in self.ppc_samples if outcome in k or k == "obs"),
            list(self.ppc_samples.keys())[-1] if self.ppc_samples else None,
        )
        if ppc_key is None:
            return "_No predictive site found in PPC samples._"

        ppc = np.array(self.ppc_samples[ppc_key])  # (draws, N)

        draw_means = np.mean(ppc, axis=-1)
        draw_stds  = np.std(ppc,  axis=-1)
        obs_mean   = float(np.mean(observed))
        obs_std    = float(np.std(observed))

        ppc_lower = np.percentile(ppc, 5.5,  axis=0)
        ppc_upper = np.percentile(ppc, 94.5, axis=0)
        coverage  = float(np.mean((observed >= ppc_lower) & (observed <= ppc_upper))) * 100

        cov_flag = "✅" if coverage >= 80 else ("⚠️" if coverage >= 60 else "❌")
        lines = [
            f"**Site**: `{ppc_key}` | **Draws**: {ppc.shape[0]} | **N**: {ppc.shape[1]}",
            "",
            "| Statistic | Observed | PPC mean | PPC SD |",
            "|-----------|----------|----------|--------|",
            f"| Mean | {obs_mean:.3f} | {float(np.mean(draw_means)):.3f} | {float(np.std(draw_means)):.3f} |",
            f"| Std  | {obs_std:.3f} | {float(np.mean(draw_stds)):.3f} | {float(np.std(draw_stds)):.3f} |",
            f"| Min  | {float(np.min(observed)):.3f} | {float(np.mean(np.min(ppc, axis=-1))):.3f} | — |",
            f"| Max  | {float(np.max(observed)):.3f} | {float(np.mean(np.max(ppc, axis=-1))):.3f} | — |",
            "",
            f"**89% PPC Coverage**: {coverage:.1f}% of observed within interval {cov_flag}",
        ]
        if coverage < 80:
            lines.append("→ Model may be misspecified or underdispersed relative to data.")

        # Bayesian p-value for mean
        bp = float(np.mean(draw_means >= obs_mean))
        bp_flag = "✅" if 0.05 <= bp <= 0.95 else "⚠️"
        lines.append(f"**Bayesian p-value (mean)**: {bp:.3f} {bp_flag}  _(extreme values near 0 or 1 signal misfit)_")

        return "\n".join(lines)

    def save_ppc_plot(self):
        """Save Plotly PPC plot as ppc_plot.html. Returns path or None."""
        if not self.ppc_samples or "_error" in self.ppc_samples:
            return None
        try:
            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            causal = self.analyze_causal_structure()
            outcome = causal.get("outcome")
            observed = self.model_args.get(outcome) if outcome else None
            if observed is None:
                return None
            observed = np.array(observed)

            ppc_key = next(
                (k for k in self.ppc_samples if outcome in k or k == "obs"),
                list(self.ppc_samples.keys())[-1] if self.ppc_samples else None,
            )
            if ppc_key is None:
                return None
            ppc = np.array(self.ppc_samples[ppc_key])

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Density: PPC draws vs Observed", "Bayesian p-value: draw means"],
            )

            # Left: observed histogram + 50 PPC draws
            fig.add_trace(
                go.Histogram(x=observed, name="Observed", histnorm="probability density",
                             marker_color="rgba(30,120,200,0.8)", nbinsx=30, zorder=10),
                row=1, col=1,
            )
            n_show = min(50, ppc.shape[0])
            for i in range(n_show):
                fig.add_trace(
                    go.Histogram(x=ppc[i], histnorm="probability density",
                                 marker_color="rgba(220,60,60,0.04)", nbinsx=30,
                                 showlegend=(i == 0), name="PPC draws"),
                    row=1, col=1,
                )

            # Right: distribution of draw means + observed mean vline
            draw_means = np.mean(ppc, axis=-1)
            obs_mean   = float(np.mean(observed))
            fig.add_trace(
                go.Histogram(x=draw_means, name="PPC draw means",
                             marker_color="rgba(220,60,60,0.6)", nbinsx=40),
                row=1, col=2,
            )
            fig.add_vline(x=obs_mean, line_dash="dash", line_color="steelblue",
                          annotation_text=f"obs mean={obs_mean:.2f}", row=1, col=2)

            fig.update_layout(
                title_text=f"Posterior Predictive Check — {outcome}",
                height=450, barmode="overlay",
            )
            path = "ppc_plot.html"
            fig.write_html(path)
            return path
        except Exception:
            return None

    def generate_report(self):
        report = []
        report.append("# BF Workflow Research Report")
        
        # 1. Data Profiling
        if self.data_profiles:
            report.append("\n## Data Profiling")
            for name, prof in self.data_profiles.items():
                report.append(f"### `{name}`")
                report.append(f"- **Shape**: `{prof.get('shape')}`")
                report.append("- **Columns/Types**:")
                for col, dtype in prof.get("dtypes", {}).items():
                    report.append(f"  - `{col}`: `{dtype}`")
        
        # 2. Causal Workflow DAG
        mermaid_text = self.as_mermaid()
        report.append("\n## Causal Workflow DAG")
        
        raw_mermaid = mermaid_text.replace("```mermaid\n", "").replace("\n```", "")
        with open("causal_workflow.mmd", "w", encoding="utf-8") as f:
            f.write(raw_mermaid)
            
        saved_image = False
        # Try Kroki first (fast and does not require Node/headless browser dependencies)
        try:
            import urllib.request, base64, zlib
            data = raw_mermaid.encode('utf-8')
            compressed = zlib.compress(data, 9)
            b64 = base64.urlsafe_b64encode(compressed).decode('ascii')
            url = f"https://kroki.io/mermaid/png/{b64}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=5) as response:
                with open("causal_workflow.png", "wb") as img_f:
                    img_f.write(response.read())
            report.append("![Causal Workflow DAG](causal_workflow.png)")
            saved_image = True
        except Exception:
            # Fallback to local mermaid-cli if Kroki fails
            import subprocess
            try:
                try:
                    subprocess.run(["mmdc", "-i", "causal_workflow.mmd", "-o", "causal_workflow.png", "-b", "transparent"], check=True, capture_output=True, timeout=10)
                    report.append("![Causal Workflow DAG](causal_workflow.png)")
                    saved_image = True
                except FileNotFoundError:
                    subprocess.run(["npx", "-y", "@mermaid-js/mermaid-cli", "-i", "causal_workflow.mmd", "-o", "causal_workflow.png", "-b", "transparent"], check=True, capture_output=True, timeout=15)
                    report.append("![Causal Workflow DAG](causal_workflow.png)")
                    saved_image = True
            except Exception:
                pass

        if not saved_image:
            report.append(mermaid_text)
                
        # Generate DOT and SVG files before checking for them
        self.save_dot_file()
        

        # 3b. Programmatic Graphs (Model Graph & Causal DAG)
        if self.fitted_BF and self.real_model:
            try:
                import matplotlib.pyplot as plt
                graph = self.fitted_BF.graph(self.real_model, draw=False)
                graph.draw(title="Programmatic Model Graph")
                plt.savefig("programmatic_model_graph.png", dpi=90, bbox_inches="tight")
                plt.close("all")
                report.append("\n## Programmatic Model Graph")
                report.append("![Model Graph](programmatic_model_graph.png)")
            except Exception:
                pass
                
            try:
                dagitty_text = self.generate_dagitty_text()
                if dagitty_text and not dagitty_text.startswith("#"):
                    import re
                    m_dag = re.search(r'dag\s*\{\s*(.*?)\s*\}', dagitty_text)
                    if m_dag:
                        dag_str = m_dag.group(1).replace(";", ",")
                        import matplotlib.pyplot as plt
                        g = self.fitted_BF.dag(dag_str)
                        
                        causal = self.analyze_causal_structure()
                        plot_kwargs = {"title": "Programmatic Causal DAG"}
                        x_var = None
                        y_var = causal.get("outcome") if causal else None
                        if y_var and causal and causal.get("predictors"):
                            x_var = causal["predictors"][0]
                            plot_kwargs["x"] = x_var
                            plot_kwargs["y"] = y_var
                            
                        g.plot(**plot_kwargs)
                        plt.savefig("programmatic_causal_dag.png", dpi=90, bbox_inches="tight")
                        plt.close("all")
            except Exception:
                pass
        
        # 5. Statistical Logic (LaTeX)
        report.append("\n## Statistical Formulation")
        if self.latex_output:
            report.append(self.latex_output)
        else:
            report.append(self.generate_latex())
        
        # 7. Causal Analysis
        causal = self.analyze_causal_structure()
        if causal:
            report.append("\n## Causal Analysis")
            outcome = causal.get("outcome")
            predictors = causal.get("predictors", [])
            latents = causal.get("latents", [])
            
            report.append(f"- **Outcome (Y)**: `{outcome if outcome else 'Not identified'}`")
            report.append(f"- **Predictors (X)**: {', '.join([f'`{p}`' for p in predictors]) if predictors else 'None'}")
            report.append(f"- **Latent Parameters ($\theta$)**: {', '.join([f'`{l}`' for l in latents]) if latents else 'None'}")
            
            estimands = self.identify_estimands()
            if estimands:
                report.append("\n### Suggested Estimands")
                for est in estimands:
                    report.append(f"- {est}")
        
        # 8. Model Archetype & Workflow
        archetype, suggestion = self.analyze_model_archetype()
        report.append("\n## Model Archetype")
        report.append(f"- **Type**: `{archetype}`")
        report.append(f"- **Suggested Next Steps**: {suggestion}")

        # 8b. Recommended BI Workflows
        report.append("\n## Recommended BI Workflows")
        report.append("Based on the model archetype, consider exploring these built-in BI workflows (`BI_mcp list_workflows`):")
        if "Hierarchical" in archetype:
            report.append("- `hierarchical`: For evaluating multilevel group-level shrinkage.")
        elif "Simple Regression" in archetype:
            report.append("- `dgp_estimation`: For parameter recovery via Data Generating Process simulation.")
            report.append("- `posterior`: For in-depth posterior analysis and marginal effects.")
        elif "High-Dimensional" in archetype:
            report.append("- `model_comparison`: To compare this model against simpler variants via LOO/WAIC.")
        elif "Purely Generative" in archetype:
            report.append("- `dgp_estimation`: To validate the prior distributions and simulation.")
        else:
            report.append("- `model_comparison`: To evaluate the fit against alternative model structures.")
            report.append("- `posterior`: For generic posterior predictive insights.")
            
        report.append("- `diagnostics`: If you encounter Divergences, R-hat, or ESS problems.")
        report.append("- `model_improvement`: For iterative improvements if predictive coverage is poor.")

        # 9. Generative Model (Simulation)
        report.append("\n## Generative Model (Simulation)")
        report.append("```python")
        report.append(self.generate_simulation_code())
        report.append("```")
        
        # 10. Diagnostics
        report.append("\n## MCMC Diagnostics Analysis")
        report.append(self.analyze_diagnostics())
        if self.summary_output:
            report.append("\n### Full Summary Table")
            report.append("```text")
            report.append(self.summary_output)
            report.append("```")

        # 10b. Advanced sampler diagnostics (diagWIP)
        adv = self.analyze_sampler_diagnostics()
        if adv:
            report.append("\n### Advanced Sampler Diagnostics")
            report.append(adv)

        # 10c. Posterior Predictive Check
        ppc_text = self.analyze_ppc()
        if ppc_text:
            report.append("\n### Posterior Predictive Check")
            report.append(ppc_text)
            ppc_path = self.save_ppc_plot()
            if ppc_path and os.path.exists(ppc_path):
                report.append(f"\n_Interactive plot: [{ppc_path}]({ppc_path})_")

        # 11. Missing Dependencies
        all_inputs = set()
        all_outputs = set()
        for n in self.nodes:
            all_inputs.update(n.inputs)
            all_outputs.update(n.outputs)
        
        all_data_cols = set()
        for p in self.data_profiles.values():
            all_data_cols.update(p.get("dtypes", {}).keys())
        
        missing = (all_inputs - all_outputs - all_data_cols) - {"m", "BF"}
        if missing:
            report.append("\n## Missing Dependencies")
            for m in sorted(list(missing)):
                report.append(f"* `{m}`")
        
        return "\n".join(report)


    def generate_dagitty_text(self):
        if not self.real_model:
            return None
        try:
            causal = self.analyze_causal_structure()
            if not causal or not causal.get("outcome"):
                return None
            
            outcome = causal["outcome"]
            predictors = causal.get("predictors", [])
            latents = causal.get("latents", [])
            
            edges = []
            # Predictors point to outcome
            for p in predictors:
                edges.append((p, outcome))
            # Latents point to outcome
            for l in latents:
                edges.append((l, outcome))
                
            edge_str = "; ".join(f"{u} -> {v}" for u, v in edges)
            nodes = [outcome] + list(predictors) + list(latents)
            iso_nodes = [n for n in nodes if all(n not in e for e in edges)]
            if iso_nodes:
                edge_str += "; " + "; ".join(iso_nodes)
            return f"dag {{ {edge_str} }}"
        except Exception as e:
            return f"# Could not generate Dagitty text: {e}"

    def generate_mermaid_plate(self):
        """Generates a Mermaid diagram representing the Plate structure found in the NumPyro DOT source."""
        if not self.real_model:
            return "_Model not yet fitted for plate extraction._"
        
        try:
            import numpyro
            import re
            
            model = self.real_model
            args = tuple(self.model_args.values())
            
            # Get DOT source with distributions rendered to see parameters
            g = numpyro.render_model(model, model_args=args, render_distributions=True)
            causal = self.analyze_causal_structure()
            if causal:
                x_vars = causal.get("predictors", [])
                y_var = causal.get("outcome")
                if y_var:
                    for x_var in x_vars:
                        g.node(x_var, label=x_var, fillcolor="white", shape="ellipse", style="filled")
                        g.edge(x_var, y_var)
            dot = g.source
            
            lines = ["```mermaid", "graph TD"]
            
            # Basic parsing of DOT to Mermaid
            # Find clusters (plates)
            plates = re.findall(r'subgraph (cluster_\w+) \{(.*?)\}', dot, re.DOTALL)
            processed_nodes = set()
            
            # Map node IDs to labels if possible
            node_labels = {}
            for line in dot.splitlines():
                m = re.search(r'(\w+)\s+\[label="([^"]+)"', line)
                if m:
                    node_labels[m.group(1)] = m.group(2)

            for plate_id, content in plates:
                plate_label = re.search(r'label="([^"]+)"', content)
                label = plate_label.group(1) if plate_label else plate_id
                lines.append(f"  subgraph {plate_id} [Plate: {label}]")
                
                # Find nodes in this plate
                node_matches = re.findall(r'(\w+)\s+\[', content)
                for node_id in node_matches:
                    lbl = node_labels.get(node_id, node_id)
                    lines.append(f'    {node_id}(("{lbl}"))')
                    processed_nodes.add(node_id)
                lines.append("  end")

            # Add nodes outside plates
            all_nodes = re.findall(r'(\w+)\s+\[', dot)
            for node_id in all_nodes:
                if node_id not in processed_nodes:
                    lbl = node_labels.get(node_id, node_id)
                    lines.append(f'  {node_id}(("{lbl}"))')

            # Add edges
            edges = re.findall(r'(\w+)\s*->\s*(\w+)', dot)
            for u, v in edges:
                lines.append(f"  {u} --> {v}")
            
            lines.append("```")
            return "\n".join(lines)
        except Exception as e:
            return f"_Could not generate Mermaid plate: {e}_"

    def save_dot_file(self):
        """Saves the DOT source to plate_diagram.dot and generates a clean SVG."""
        if not self.real_model:
            return
        try:
            import numpyro
            model = self.real_model
            args = tuple(self.model_args.values())
            g = numpyro.render_model(model, model_args=args, render_distributions=True)
            causal = self.analyze_causal_structure()
            if causal:
                x_vars = causal.get("predictors", [])
                y_var = causal.get("outcome")
                if y_var:
                    for x_var in x_vars:
                        g.node(x_var, label=x_var, fillcolor="white", shape="ellipse", style="filled")
                        g.edge(x_var, y_var)
            dot_source = g.source
            with open("plate_diagram.dot", "w") as f:
                f.write(dot_source)

            # Strategy 1: use 'dot' CLI (graphviz system binary)
            try:
                import subprocess
                subprocess.run(
                    ["dot", "-Tsvg", "-o", "plate_diagram.svg", "plate_diagram.dot"],
                    check=True, capture_output=True, timeout=10
                )
                return
            except Exception:
                pass

            # Strategy 2: Kroki API
            try:
                import urllib.request, base64, zlib
                data = dot_source.encode("utf-8")
                compressed = zlib.compress(data, 9)
                b64 = base64.urlsafe_b64encode(compressed).decode("ascii")
                urllib.request.urlretrieve(f"https://kroki.io/graphviz/svg/{b64}", "plate_diagram.svg")
                return
            except Exception:
                pass

            # Strategy 3: matplotlib manual render from DOT source
            try:
                import re, math, matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                edges = re.findall(r'\t(\w+)\s*->\s*(\w+)', dot_source)
                node_labels_raw = re.findall(r'\t(\w+)\s+\[label=(\w+)', dot_source)
                observed_set = set(re.findall(r'\t(\w+)\s+\[label=\w+\s+fillcolor=grey', dot_source))
                dist_text_match = re.search(r'distribution_description_node\s+\[label=\"([^\"]+)\"', dot_source)

                labels = {n: l for n, l in node_labels_raw if n != "distribution_description_node"}
                draw_edges = [(u, v) for u, v in edges if "distribution_description_node" not in (u, v)]

                latent_nodes = [n for n in labels if n not in observed_set]
                observed_nodes = [n for n in labels if n in observed_set]
                pos = {}
                for i, n in enumerate(latent_nodes):
                    pos[n] = (i - (len(latent_nodes) - 1) / 2, 1.0)
                for i, n in enumerate(observed_nodes):
                    pos[n] = (0.0, -0.5)

                fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
                ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 2)
                ax.set_aspect("equal")

                for u, v in draw_edges:
                    if u in pos and v in pos:
                        x1, y1 = pos[u]; x2, y2 = pos[v]
                        dy = -0.18 if y1 > y2 else 0.18
                        ax.annotate("", xy=(x2, y2 + 0.18), xytext=(x1, y1 - 0.18),
                                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

                for node, (x, y) in pos.items():
                    color = "#c0c0c0" if node in observed_set else "white"
                    lbl = labels.get(node, node)
                    circle = plt.Circle((x, y), 0.18, color=color, ec="black", lw=1.5, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y, lbl, ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

                if dist_text_match:
                    dist_txt = dist_text_match.group(1).replace(r"\l", "\n").strip()
                    ax.text(1.1, 1.0, dist_txt, fontsize=9, va="top",
                            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

                ax.set_title("Probabilistic Plate Diagram", fontsize=13)
                ax.axis("off")
                plt.tight_layout()
                plt.savefig("plate_diagram.svg", format="svg", bbox_inches="tight")
                plt.close("all")
            except Exception:
                pass

        except Exception:
            pass


    def analyze_causal_structure(self):
        # 1. Identify Outcome (Y) - The node receiving 'obs' data
        outcome_var = None
        outcome_node = None
        for n in reversed(self.nodes):
            if n.kind == "latent" and "obs" in n.source:
                # Extract the actual variable name from outputs
                if n.outputs:
                    outcome_var = n.outputs[0]
                outcome_node = n
                break
        
        if not outcome_node:
            return {}

        # 2. Identify Predictors (X) and Latents (theta)
        predictors = set()
        latents = set()
        
        # Get all ancestors of the outcome node in the graph
        ancestors = nx.ancestors(self.g, outcome_node.id)
        
        for node_id in ancestors:
            node = self.g.nodes[node_id]['data']
            if node.kind == "latent":
                # Prior parameters
                name = node.label.split(":")[-1]
                if name not in ["normal", "log_normal", "uniform"]:
                    latents.add(name)
            elif node.kind in ("data_attach", "assign", "call_assign", "model_assign"):
                for inp in node.inputs:
                    predictors.add(inp)
                for out in node.outputs:
                    predictors.add(out)
        
        # Crucial: Find data variables used in the likelihood node itself
        for inp in outcome_node.inputs:
            # Check if the input is a data variable
            if inp in self.data_profiles:
                predictors.add(inp)
            # Also check if it's a variable name that matches a data column
            all_data_cols = set()
            for p in self.data_profiles.values():
                all_data_cols.update(p.get("dtypes", {}).keys())
            if inp in all_data_cols:
                predictors.add(inp)
        
        # Filter predictors to only those that are actual data columns
        all_data_cols = set()
        for p in self.data_profiles.values():
            all_data_cols.update(p.get("dtypes", {}).keys())
        
        final_predictors = predictors.intersection(all_data_cols)
        
        return {
            "outcome": outcome_var,
            "predictors": list(final_predictors),
            "latents": list(latents),
            "outcome_node_id": outcome_node.id
        }

    def identify_estimands(self):
        structure = self.analyze_causal_structure()
        if not structure or not structure.get("outcome"):
            return []
        
        outcome = structure["outcome"]
        predictors = structure["predictors"]
        latents = structure["latents"]
        outcome_node_id = structure["outcome_node_id"]
        
        estimands = []
        
        # For each predictor, find the path to the outcome
        for x in predictors:
            # Find node representing the predictor
            x_node_id = self.var_last_node.get(x)
            if not x_node_id: continue
            
            # Find paths from x_node_id to outcome_node_id
            try:
                paths = nx.all_simple_paths(self.g, source=x_node_id, target=outcome_node_id)
                for path in paths:
                    # Path is a list of node IDs. 
                    # We care about latents in between.
                    mid_nodes = [self.g.nodes[nid]['data'].label.split(":")[-1] 
                                 for nid in path if self.g.nodes[nid]['data'].kind == "latent"]
                    
                    if not mid_nodes:
                        estimands.append(f"Direct effect of `{x}` on `{outcome}`")
                    else:
                        mediators = ", ".join(mid_nodes)
                        estimands.append(f"Effect of `{x}` on `{outcome}` via parameter(s) `{mediators}`")
            except nx.NetworkXNoPath:
                continue
                
        return estimands

    def generate_latex(self):
        if self.latex_output:
            return self.latex_output
            
        # Use native logic but adapt to our captured source to avoid inspect errors
        try:
            from BayesForge.PostModel.to_latex import extract_lines, extract_latex_line_final
            if not self.model_fn:
                return "_No model function found in script._"
            
            source = ast.get_source_segment(self.source, self.model_fn)
            lines = extract_lines(source)
            # Skip the 'def model(...):' line
            lines_clean = [line.lstrip() for line in lines][1:]
            latex_lines = [extract_latex_line_final(line) for line in lines_clean]
            
            latex_str = "$$\n\\begin{aligned}\n"
            latex_str += " \\\\\n".join(filter(None, reversed(latex_lines)))
            latex_str += "\n\\end{aligned}\n$$"
            return latex_str
        except Exception as e:
            return f"_LaTeX generation error: {e}_"

    def generate_simulation_code(self):
        if not self.model_fn:
            return "# No model function found"
        
        # 1. Find the likelihood node (the one with obs=...) to determine what we are simulating
        likelihood_node = None
        for n in self.nodes:
            if n.kind == "latent" and "obs" in n.source:
                likelihood_node = n
                break
        
        # Determine the outcome variable name (e.g., 'height')
        outcome_var = None
        if likelihood_node:
            # Try to find the variable name from outputs or by parsing the source
            if likelihood_node.outputs:
                outcome_var = likelihood_node.outputs[0]
            else:
                # Fallback: search for 'obs=' in the source
                import re
                m = re.search(r'obs\s*=\s*(\w+)', likelihood_node.source)
                if m: outcome_var = m.group(1)

        # 2. Identify the distribution used for the outcome
        dist_name = "normal" # default
        if likelihood_node:
            dist_name = likelihood_node.label.split(":")[-1]

        # 3. Generate header comments
        lines = []
        if outcome_var:
            lines.append(f"# Generate `{outcome_var}` through a {dist_name} distribution based on the likelihood model")
        else:
            lines.append("# Automated Generative Process (Prior Simulation)")

        # 4. Process the source code
        source = ast.get_source_segment(self.source, self.model_fn)
        
        import re
        # Process each line to handle the 'obs' argument
        updated_lines = []
        for line in source.splitlines():
            # Remove 'obs=something' and replace with 'something ='
            # Look for m.dist.something(..., obs=var)
            if "obs=" in line and "m.dist." in line:
                # Find the variable assigned to obs
                obs_match = re.search(r'obs\s*=\s*(\w+)', line)
                if obs_match:
                    var_name = obs_match.group(1)
                    # Remove the obs argument and its comma
                    # This regex handles both 'obs=var,' and ', obs=var'
                    clean_line = re.sub(r',\s*obs\s*=\s*\w+', '', line)
                    clean_line = re.sub(r'obs\s*=\s*\w+,\s*', '', clean_line)
                    # Transform 'm.dist.normal(...)' to 'var = m.dist.normal(...)'
                    # We need to find where the call starts
                    call_start = clean_line.find("m.dist.")
                    if call_start != -1:
                        prefix = clean_line[:call_start].strip()
                        # If it was just a call, we prepend the outcome variable
                        if not prefix or not prefix.endswith("="):
                            clean_line = f"{var_name} = {clean_line[call_start:]}"
                        
                        updated_lines.append(clean_line)
                        continue
            
            # Standard replacement for sample=True in priors
            updated_line = re.sub(r'(m\.dist\.\w+\()', r'\1sample=True, ', line)
            updated_lines.append(updated_line)

        # Remove function declaration and indentation
        final_code = "\n".join(updated_lines)
        if "def " in final_code:
            lines_split = final_code.splitlines()
            if lines_split and lines_split[0].startswith("def "):
                body = lines_split[1:]
                if body:
                    indent = len(body[0]) - len(body[0].lstrip())
                    body_clean = [l[indent:] if len(l) >= indent else l for l in body]
                    final_code = "\n".join(body_clean)

        lines.append(final_code)
        return "\n".join(lines)


    def as_mermaid(self):
        lines = ["```mermaid", "graph TD"]
        # Styling
        lines.append("  classDef data fill:#f9f,stroke:#333,stroke-width:2px;")
        lines.append("  classDef latent fill:#bbf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;")
        lines.append("  classDef outcome fill:#fbb,stroke:#333,stroke-width:2px;")
        
        # Variables we want to show
        vars_nodes: Dict[str, str] = {} # var_name -> node_id
        
        # Helper to recursively resolve inputs through deterministic assignments
        def resolve_inputs(name: str) -> Set[str]:
            if name in vars_nodes:
                return {name}
            for node in self.nodes:
                if node.kind in ("model_assign", "Assign", "latent") and name in node.outputs:
                    if node.kind == "latent":
                        site_name = node.label.split(":")[-1]
                        if site_name in vars_nodes:
                            return {site_name}
                    res = set()
                    for inp in node.inputs:
                        res.update(resolve_inputs(inp))
                    return res
            return set()

        GREEK_LETTERS = {
            "alpha": "α",
            "beta": "β",
            "gamma": "γ",
            "delta": "δ",
            "epsilon": "ε",
            "zeta": "ζ",
            "eta": "η",
            "theta": "θ",
            "iota": "ι",
            "kappa": "κ",
            "lambda": "λ",
            "mu": "μ",
            "nu": "ν",
            "xi": "ξ",
            "omicron": "ο",
            "pi": "π",
            "rho": "ρ",
            "sigma": "σ",
            "tau": "τ",
            "upsilon": "υ",
            "phi": "φ",
            "chi": "χ",
            "psi": "ψ",
            "omega": "ω",
        }

        def replace_greek(text: str) -> str:
            import re
            for g_name, g_char in GREEK_LETTERS.items():
                text = re.sub(r'\b' + g_name + r'\b', g_char, text, flags=re.IGNORECASE)
            return text

        def to_pretty_label(text: str) -> str:
            parts_super = text.split('^')
            base_sub_part = parts_super[0]
            super_part = parts_super[1] if len(parts_super) > 1 else None
            
            parts_sub = base_sub_part.split('_')
            base_part = parts_sub[0]
            sub_part = parts_sub[1] if len(parts_sub) > 1 else None
            
            base_pretty = replace_greek(base_part)
            
            result = base_pretty
            if sub_part:
                sub_pretty = replace_greek(sub_part)
                result += f"<sub>{sub_pretty}</sub>"
            if super_part:
                super_pretty = replace_greek(super_part)
                result += f"<sup>{super_pretty}</sup>"
            return result

        def pretty_formula(expr: str) -> str:
            import re
            pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\^[a-zA-Z0-9_]+)?\b'
            return re.sub(pattern, lambda m: to_pretty_label(m.group(0)), expr)

        # 1. Identify Data Variables
        data_vars = set()
        for prof_name, prof in self.data_profiles.items():
            for col in prof.get("dtypes", {}).keys():
                data_vars.add(col)
                node_id = f"d_{col}"
                # Only add if the variable is actually used in the model
                if any(col in n.inputs or col in n.outputs for n in self.nodes):
                    if node_id not in [l.split('[')[0].strip() for l in lines]:
                        lines.append(f'  {node_id}["{col}"]:::data')
                    vars_nodes[col] = node_id


        # 2. Identify Latents and Outcomes
        latent_vars: Dict[str, Node] = {} # name -> node
        outcome_vars: Dict[str, str] = {} # var_name -> node_id (usually d_var)
        
        # First, find all latents (parameters)
        for n in self.nodes:
            if n.kind == "latent":
                name = n.label.split(":")[-1]
                # If it's a parameter (has a name in the script like a, b, s)
                if ":" in n.label and n.label.startswith("prior:"):
                    if name not in vars_nodes:
                        lines.append(f'  {n.id}(("{to_pretty_label(name)}")):::latent')
                        vars_nodes[name] = n.id
                    # Only set/overwrite if we don't already have a static AST node with dist info
                    existing = latent_vars.get(name)
                    if existing is None or not existing.meta.get("runtime"):
                        if existing is None or n.meta.get("dist"):
                            latent_vars[name] = n

        # Find multiplications between coefficient latents and data predictors
        coef_targets = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                left_names = collect_names(node.left)
                right_names = collect_names(node.right)
                
                resolved_left = set()
                for n_val in left_names:
                    resolved_left.update(resolve_inputs(n_val))
                resolved_right = set()
                for n_val in right_names:
                    resolved_right.update(resolve_inputs(n_val))
                
                for l_val in resolved_left:
                    for r_val in resolved_right:
                        if l_val in latent_vars and r_val in data_vars:
                            coef_targets[l_val] = r_val
                        elif r_val in latent_vars and l_val in data_vars:
                            coef_targets[r_val] = l_val

        # Second, handle likelihood nodes — insert a link-function node between inputs and outcome
        seen_link_edges = set()
        for n in self.nodes:
            if n.kind != "latent" or not n.label.startswith("dist:"):
                continue
            # Only process AST nodes that carry arg_exprs; skip runtime duplicates
            arg_exprs = n.meta.get("arg_exprs")
            if arg_exprs is None:
                continue

            obs_targets = [o for o in n.outputs if o in vars_nodes]
            if not obs_targets:
                continue

            target_id = vars_nodes[obs_targets[0]]
            dist_name = n.label.split(":")[-1].capitalize()

            # Build link-function label: e.g. "Normal(a + b * weight, s)"
            expr_str = ", ".join(arg_exprs)
            link_label = pretty_formula(f"{dist_name}({expr_str})")
            link_id = f"link_{n.id}"
            lines.append(f'  {link_id}["{link_label}"]')

            # Highlight outcome node
            for i, line in enumerate(lines):
                if line.startswith(f"  {target_id}["):
                    lines[i] = line.replace(":::data", ":::outcome")

            # inputs → link node
            resolved_inps = set()
            for inp_name in n.inputs:
                resolved_inps.update(resolve_inputs(inp_name))

            for inp_name in resolved_inps:
                if inp_name in vars_nodes:
                    src_id = vars_nodes[inp_name]
                    # If this input is a latent coefficient and has a predictor target,
                    # draw the edge directly to the predictor instead of the link node!
                    if inp_name in coef_targets:
                        target_var = coef_targets[inp_name]
                        if target_var in vars_nodes:
                            coeff_target_id = vars_nodes[target_var]
                            edge = (src_id, coeff_target_id)
                            if edge not in seen_link_edges:
                                lines.append(f"  {src_id} --> {coeff_target_id}")
                                seen_link_edges.add(edge)
                            continue
                    
                    edge = (src_id, link_id)
                    if edge not in seen_link_edges:
                        lines.append(f"  {src_id} --> {link_id}")
                        seen_link_edges.add(edge)

            # link node → outcome
            lines.append(f"  {link_id} --> {target_id}")

        # 3. Add Edges for Latent Parameter dependencies
        seen_edges = set()
        for name, node in latent_vars.items():
            resolved_inps = set()
            for inp in node.inputs:
                resolved_inps.update(resolve_inputs(inp))
            for inp in resolved_inps:
                if inp in vars_nodes:
                    src_id = vars_nodes[inp]
                    edge = (src_id, node.id)
                    if edge not in seen_edges:
                        lines.append(f"  {src_id} --> {node.id}")
                        seen_edges.add(edge)
                elif inp in latent_vars:
                    # Connect prior to its parameter if it's a dependency
                    src_id = vars_nodes.get(inp)
                    if src_id:
                        edge = (src_id, node.id)
                        if edge not in seen_edges:
                            lines.append(f"  {src_id} --> {node.id}")
                            seen_edges.add(edge)

        # 4. Add Causal Info Card at the bottom of the diagram
        causal = self.analyze_causal_structure()
        if causal and causal.get("outcome") and causal.get("predictors"):
            y_var = causal["outcome"]
            dag_text = self.generate_dagitty_text()
            
            dag_g = None
            if dag_text and not dag_text.startswith("#"):
                import re
                m_dag = re.search(r'dag\s*\{\s*(.*?)\s*\}', dag_text)
                if m_dag:
                    dag_str = m_dag.group(1).replace(";", ",")
                    try:
                        from BayesForge.Causal.dag import DAG
                        dag_g = DAG(dag_str)
                    except Exception:
                        pass

            adj_lines = []
            for pred in causal["predictors"]:
                adj_str = "N/A"
                if dag_g:
                    try:
                        adj = dag_g.adjustment_sets(pred, y_var)
                        if not adj:
                            adj_str = "∅"
                        else:
                            adj_str = ", ".join(f"{{{', '.join(s)}}}" if s else "∅" for s in adj)
                    except Exception:
                        pass
                adj_lines.append(f"Adjustment Sets ({to_pretty_label(pred)} -> {to_pretty_label(y_var)}): {to_pretty_label(adj_str)}")
            
            adj_section = "<br/>".join(adj_lines)
            
            # Format prior distributions for the legend
            priors_list = []
            for name, node in latent_vars.items():
                dist_name = node.meta.get("dist", "").split(".")[-1] if node.meta else ""
                arg_exprs = node.meta.get("arg_exprs", []) if node.meta else []
                if dist_name and arg_exprs:
                    formatted_dist = "".join(part.capitalize() for part in dist_name.split("_"))
                    pretty_args = [pretty_formula(arg) for arg in arg_exprs]
                    priors_list.append(f"{to_pretty_label(name)} ~ {formatted_dist}({', '.join(pretty_args)})")
            
            # Format data and their types
            data_list = []
            for prof_name, prof in self.data_profiles.items():
                for col, dtype in prof.get("dtypes", {}).items():
                    data_list.append(f"{col}: {dtype}")
            data_str = "<br/>".join(data_list)
            data_section = f"<br/><br/>Data:<br/>{data_str}" if data_list else ""

            priors_str = "<br/>".join(priors_list)
            priors_section = f"<br/><br/>Priors:<br/>{priors_str}" if priors_list else ""
            
            lines.append("  classDef info fill:#f0f4f8,stroke:#cbd5e1,stroke-width:1px;")
            lines.append(f'  info_node["Causal Properties<br/>{adj_section}{priors_section}{data_section}"]:::info')

        lines.append("```")
        return "\n".join(lines)

    def build(self):
        for stmt in self.tree.body:
            self._handle(stmt, in_model=False)
        return self

    def _handle(self, stmt, in_model):
        if isinstance(stmt, ast.Assign):
            self._assign(stmt, in_model)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._call(stmt.value, in_model)
        elif isinstance(stmt, ast.FunctionDef):
            self._model_def(stmt)

    def _assign(self, stmt, in_model):
        if not isinstance(stmt.value, ast.Call):
            out = [target_name(t) for t in stmt.targets]
            inp = list(collect_names(stmt.value) - {"m"})
            self.add_node("assign", "Assign", unparse(stmt), inp, out)
            return

        call = stmt.value
        path = call_path(call.func)
        out = [target_name(t) for t in stmt.targets]

        if path == "BF":
            self.add_node("setup", "BF()", unparse(stmt), outputs=out)
            return

        if path.startswith("m.load."):
            self.add_node("data_load", path, unparse(stmt), outputs=out)
            return

        self.add_node("call_assign", path, unparse(stmt), outputs=out)

    def _call(self, call, in_model):
        path = call_path(call.func)

        if path == "m.data":
            node = self.add_node(
                "data_attach",
                "m.data",
                unparse(call),
                inputs=["data_path"],
                outputs=["m.df"],
            )

            self.var_last_node["m.df"] = node.id
            return

        if path == "m.scale":
            self.add_node("scale", "m.scale", unparse(call), inputs=["m.df"], outputs=["m.df"])
            return

        if path == "m.fit":
            model = call.args[0].id if call.args and isinstance(call.args[0], ast.Name) else None

            node = self.add_node(
                "fit",
                "fit",
                unparse(call),
                inputs=["m.df", model] if model else ["m.df"],
                outputs=["posterior"],
            )

            self.var_last_node["posterior"] = node.id
            return

        if path == "m.summary":
            self.add_node("summary", "summary", unparse(call), inputs=["posterior"])
            return

        self.add_node("call", path, unparse(call))

    def _model_def(self, fn):
        self.model_fn = fn
        self.add_node(
            "model_def",
            f"model:{fn.name}",
            ast.get_source_segment(self.source, fn),
            outputs=[fn.name],
        )

        for stmt in fn.body:
            if isinstance(stmt, ast.Assign):
                self._model_assign(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._model_expr(stmt.value)

    def _model_expr(self, call):
        path = call_path(call.func)
        
        # Extract TRUE dependencies from arguments
        inp = set()
        obs_var = None
        for arg in call.args:
            inp.update(collect_names(arg))
        for kw in call.keywords:
            if kw.arg == "name":
                continue
            if kw.arg == "obs":
                names = collect_names(kw.value)
                if names: obs_var = list(names)[0]
            else:
                inp.update(collect_names(kw.value))
        inp -= {"m"}

        if path.startswith("m.dist."):
            arg_exprs = [unparse(a) for a in call.args]
            self.add_node(
                "latent",
                f"dist:{path.split('.')[-1]}",
                unparse(call),
                inputs=list(inp),
                outputs=[obs_var] if obs_var else [],
                meta={"dist": path, "arg_exprs": arg_exprs, "obs_var": obs_var},
            )
            return

        self.add_node("model_expr", path, unparse(call), inputs=list(inp))

    def _model_assign(self, stmt):
        call = stmt.value
        out = [target_name(t) for t in stmt.targets]
        if not isinstance(call, ast.Call):
            inp = list(collect_names(call) - {"m"})
            self.add_node("model_assign", "Assign", unparse(stmt), inputs=inp, outputs=out)
            return
            
        path = call_path(call.func)

        # Extract TRUE dependencies from arguments
        inp = set()
        obs_var = None
        for arg in call.args:
            inp.update(collect_names(arg))
        for kw in call.keywords:
            if kw.arg == "name":
                continue
            if kw.arg == "obs":
                names = collect_names(kw.value)
                if names: obs_var = list(names)[0]
            else:
                inp.update(collect_names(kw.value))
        inp -= {"m"}

        if path.startswith("m.dist."):
            latent = None
            for kw in call.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    latent = kw.value.value

            outputs = list(out)
            if latent and latent not in outputs:
                outputs.append(latent)
            if obs_var and obs_var not in outputs:
                outputs.append(obs_var)

            arg_exprs = [unparse(a) for a in call.args]
            self.add_node(
                "latent",
                f"prior:{latent}" if latent else f"dist:{path.split('.')[-1]}",
                unparse(stmt),
                inputs=list(inp),
                outputs=outputs,
                meta={"dist": path, "arg_exprs": arg_exprs, "obs_var": obs_var},
            )
            return

        self.add_node("model_assign", path, unparse(stmt), inputs=list(inp), outputs=out)


# =========================================================
# TRACED BF (RUNTIME LAYER)
# =========================================================

class TracedBI:
    def __init__(self, real, builder=None):
        super().__setattr__("_m", real)
        super().__setattr__("builder", builder)
        super().__setattr__("state", {})
        super().__setattr__("last_model", None)

    def __setattr__(self, name, value):
        if name in ("_m", "builder", "state", "last_model"):
            super().__setattr__(name, value)
        else:
            setattr(self._m, name, value)

    def __getattr__(self, name):
        attr = getattr(self._m, name)

        if name == "dist":
            return TracedDist(attr, self)

        return attr

    def data(self, *args, **kwargs):
        res = self._m.data(*args, **kwargs)

        if self.builder:
            inp = []
            if args:
                if isinstance(args[0], str):
                    inp = ["data_path"] # Map runtime path arg to AST data_path
                else:
                    inp = ["data"]
            
            # Capture data profiling
            if hasattr(res, "shape"):
                self.builder.data_profiles["m.df"] = {
                    "shape": res.shape,
                    "dtypes": {k: str(v) for k, v in res.dtypes.items()} if hasattr(res, "dtypes") else {}
                }

            self.builder.add_runtime_event(
                kind="data_attach",
                label="m.data",
                inputs=inp,
                outputs=["m.df"],
                payload={"args": args, "kwargs": kwargs}
            )

        return res

    def scale(self, *args, **kwargs):
        res = self._m.scale(*args, **kwargs)

        if self.builder:
            self.builder.add_runtime_event(
                kind="scale",
                label="m.scale",
                inputs=["m.df"],
                outputs=["m.df"],
                payload={"args": args, "kwargs": kwargs}
            )

        return res

    def fit(self, model, *args, **kwargs):
        res = self._m.fit(model, *args, **kwargs)

        if self.builder:
            try:
                # Try all possible names
                methods = ["latex", "to_latex"]
                for mname in methods:
                    if hasattr(self._m, mname):
                        res_latex = getattr(self._m, mname)()
                        if res_latex:
                            self.builder.latex_output = res_latex
                            break
            except Exception as e:
                print(f"LaTeX error: {e}")

            if hasattr(self._m, "df"):
                df = self._m.df
                self.builder.data_profiles["m.df"] = {
                    "shape": df.shape,
                    "dtypes": {k: str(v) for k, v in df.dtypes.items()} if hasattr(df, "dtypes") else {}
                }

            model_name = getattr(model, "__name__", "model")
            self.builder.real_model = model
            self.builder.model_args = self._m.data_on_model if hasattr(self._m, "data_on_model") else {}
            self.builder.fitted_BF = self._m

            self.builder.add_runtime_event(
                kind="fit",
                label="m.fit",
                inputs=["m.df", model_name],
                outputs=["posterior"],
                payload={"args": args, "kwargs": kwargs}
            )

            try:
                self.run_ppc()
            except Exception:
                pass

        return res

    def run_ppc(self, rng_seed=42):
        """Auto-run PPC after fitting. Temporarily disables builder to suppress trace logging."""
        import jax
        import jax.numpy as jnp
        from numpyro.infer import Predictive

        if not self.builder or not self.builder.real_model or not self.builder.model_args:
            return

        # Resolve raw NumPyro MCMC sampler from BayesForge object
        sampler = None
        for attr in ("sampler", "mcmc", "_sampler", "_mcmc"):
            if hasattr(self._m, attr):
                sampler = getattr(self._m, attr)
                break
        if sampler is None:
            return

        posterior_samples = sampler.get_samples(group_by_chain=False)

        # Build pred kwargs: covariates only, outcome set to None so NumPyro samples it
        causal = self.builder.analyze_causal_structure()
        outcome = causal.get("outcome")
        pred_kwargs = {
            k: jnp.array(v) if hasattr(v, "__len__") else v
            for k, v in self.builder.model_args.items()
        }
        if outcome and outcome in pred_kwargs:
            pred_kwargs[outcome] = None

        # Disable builder so TracedDist doesn't log PPC calls as graph nodes
        saved_builder = self.builder
        self.builder = None
        try:
            predictive = Predictive(self.builder if False else saved_builder.real_model,
                                    posterior_samples=posterior_samples)
            ppc = predictive(jax.random.PRNGKey(rng_seed), **pred_kwargs)
            saved_builder.ppc_samples = ppc
        except Exception as e:
            saved_builder.ppc_samples = {"_error": str(e)}
        finally:
            self.builder = saved_builder

    def summary(self, *args, **kwargs):
        res = self._m.summary(*args, **kwargs)

        if self.builder:
            self.builder.summary_output = str(res)
            self.builder.add_runtime_event(
                kind="summary",
                label="m.summary",
                inputs=["posterior"],
                payload={"args": args, "kwargs": kwargs}
            )

        return res


class TracedDist:
    def __init__(self, real, traced):
        self._d = real
        self.traced = traced

    def __getattr__(self, name):
        fn = getattr(self._d, name)

        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)

            if self.traced.builder:
                name_val = kwargs.get("name")
                self.traced.builder.add_runtime_event(
                    kind="latent",
                    label=f"prior:{name_val}" if name_val else f"dist:{name}",
                    inputs=[], 
                    outputs=[name_val] if name_val else [],
                    payload={"type": name, "args": str(args), "kwargs": str(kwargs)},
                )

            return res

        return wrapper

#%%
# =========================================================
# EXAMPLE
# =========================================================

script = """
from BayesForge import BayesForge

data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=';')
m.scale(['weight', 'age'])

def model(weight, height, age):
    alpha = m.dist.normal(178, 20, name='alpha')
    beta_weight = m.dist.log_normal(0, 1, name='beta_weight')
    beta_age = m.dist.log_normal(0, 1, name='beta_age')    
    sigma = m.dist.uniform(0, 50, name='sigma')
    
    m.dist.normal(alpha + beta_weight * weight + beta_age*age, sigma, name='height', obs=height)

m.fit(model, num_warmup=500, num_samples=500, num_chains=1)
m.summary()
"""

from BayesForge import bf

builder = ScriptGraphBuilder(script)
builder.build() # Static analysis first

real_m = bf()
m = TracedBI(real_m, builder)

# Use m for runtime trace
env = {
    "m": m,
    "BF": bf,
}

exec(script, env)

report = builder.generate_report()
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
print(report)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(report)