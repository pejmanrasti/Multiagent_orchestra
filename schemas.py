"""
schemas.py — the structured messages that flow between the three agents and the loop.

    Strategist ──StrategyProposal──► Coder ──train.py──► (run + deterministic eval)
        ▲                                                         │
        └────────────── ErrorAnalysis ◄── Error Analysis Agent ◄──┘ (validation report)

Nothing here carries test-set numbers: test performance lives only in the final
report, produced after the last agent call.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class StrategyProposal(BaseModel):
    """Model Strategist output: WHAT to try next and WHY (never code)."""
    experiment_name: str = Field(..., description="Short unique name, e.g. 'resnet18_ft_aug'.")
    architecture: str = Field(..., description="Model architecture / family to use.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict,
        description="Model hyperparameters, e.g. {'C': 10} or {'dropout': 0.3}.")
    training_config: Dict[str, Any] = Field(default_factory=dict,
        description="Optimiser, learning rate, epochs, augmentation, regularisation, ...")
    changes_from_previous: List[str] = Field(default_factory=list,
        description="Concrete differences from the current best experiment.")
    addresses_findings: List[str] = Field(default_factory=list,
        description="Which error-analysis findings this experiment responds to.")
    rationale: str = Field(..., description="Why this follows from the evidence.")
    expected_effect: str = Field("", description="Predicted effect on validation metrics.")
    coder_instructions: str = Field(...,
        description="Precise implementation instructions for the Coder agent.")
    stop: bool = Field(False, description="True if further experiments are not worthwhile.")

    def fingerprint(self) -> str:
        """Identity of a configuration, used to detect repeated experiments."""
        import json
        norm = lambda d: json.dumps(d, sort_keys=True, default=str).lower()
        return f"{self.architecture.strip().lower()}|{norm(self.hyperparameters)}|" \
               f"{norm(self.training_config)}"


class ErrorAnalysis(BaseModel):
    """Error Analysis Agent output: an interpretation of deterministic validation metrics."""
    fit_diagnosis: str = Field(...,
        description="One of: overfitting, underfitting, good_fit, unclear.")
    observations: List[str] = Field(default_factory=list,
        description="Facts directly visible in the metrics (cite the numbers).")
    hypotheses: List[str] = Field(default_factory=list,
        description="Possible CAUSES of the observations, explicitly tentative.")
    class_weaknesses: List[str] = Field(default_factory=list,
        description="Classes with poor precision/recall/F1 and how they fail.")
    systematic_errors: List[str] = Field(default_factory=list,
        description="Recurring confusions or error patterns.")
    significance: str = Field("",
        description="Whether the change vs the current best is statistically meaningful.")
    recommendations: List[str] = Field(default_factory=list,
        description="Actionable suggestions for the Model Strategist, most important first.")
    summary: str = Field("", description="Two-sentence summary.")

    @field_validator("fit_diagnosis")
    @classmethod
    def _norm_fit(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        for key in ("overfitting", "underfitting", "good_fit"):
            if v.startswith(key[:6]):
                return key
        return "unclear"


class ExperimentRecord(BaseModel):
    """Everything recorded about ONE iteration (validation data only)."""
    iteration: int
    proposal: StrategyProposal
    status: str = "pending"            # ok | codegen_failed | train_failed | eval_failed
    error: str = ""
    seed: int = 42
    train_metrics: Dict[str, Any] = Field(default_factory=dict)   # summary on the train split
    val_metrics: Dict[str, Any] = Field(default_factory=dict)     # full deterministic report
    selection_score: Optional[float] = None
    comparison_to_best: Optional[Dict[str, Any]] = None
    analysis: Optional[ErrorAnalysis] = None
    is_best: bool = False
    artifacts_dir: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.selection_score is not None


class PipelineResult(BaseModel):
    """What run() returns: the selected model and its single test evaluation."""
    run_dir: str
    selection_metric: str
    best: Optional[ExperimentRecord] = None
    test_metrics: Optional[Dict[str, Any]] = None
    history: List[ExperimentRecord] = Field(default_factory=list)
    stop_reason: str = ""
