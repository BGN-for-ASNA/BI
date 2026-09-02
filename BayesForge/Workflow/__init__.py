"""Multi-fit workflow orchestration for BF -- ``m.workflow``.

See :class:`BayesForge.Workflow.core.Workflow` for the full method list
(recover, sbc, contrast, poststratify, decide, annotated_summary, advise,
parallel_report, plot_recovery, plot_sbc, plot_annotated_summary).
"""
from .core import Workflow
from .results import RecoveryResult, SBCResult, ContrastResult

__all__ = ["Workflow", "RecoveryResult", "SBCResult", "ContrastResult"]
