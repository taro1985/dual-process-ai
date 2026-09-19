#!/usr/bin/env python3
"""
Type-Safe Data Models for DPAI (Dual-Process AI)

Provides Pydantic models with dict-like backward compatibility:
- RoutingDecision: System 1 classification result
- ProcessResult: Dual-process pipeline execution result
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class DictCompatibleModel(BaseModel):
    """BaseModel with dict-like index and .get() access for full backward compatibility."""

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RoutingDecision(DictCompatibleModel):
    """Result of System 1 reflexive classification."""
    source: str = Field(..., description="Classifier source (Jev API, Keyword, Habituation, etc.)")
    action: str = Field(..., description="Action name or routing target")
    confidence: float = Field(..., description="Calibrated confidence score (0.0 - 1.0)")
    urgency: int = Field(default=1, description="Urgency level (0-5)")
    latency_ms: float = Field(default=0.0, description="Classification latency in milliseconds")
    is_habituated: bool = Field(default=False, description="Whether this decision was served from learned S1 reflex")
    habit_response: Optional[str] = Field(default=None, description="Compiled habit response if habituated")


class ProcessResult(DictCompatibleModel):
    """Complete execution result of DualProcessRouter.process()."""
    input: str = Field(..., description="Original user input query")
    decision: RoutingDecision = Field(..., description="System 1 classification decision")
    confidence: float = Field(..., description="Confidence score")
    threshold: float = Field(..., description="Effective confidence threshold used")
    risk_level: str = Field(..., description="Evaluated action risk level")
    system2_engaged: bool = Field(..., description="Whether System 2 deep reasoning was engaged")
    is_habituated: bool = Field(default=False, description="Whether answered via S1 habituated reflex")
    is_fallback: bool = Field(default=False, description="Whether served via local/offline fallback due to LLM limit")
    pruned_tools: List[str] = Field(default_factory=list, description="Names of pruned MCP tools")
    principles: List[str] = Field(default_factory=list, description="Actionable rules from recalled principles")
    total_latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    output: str = Field(..., description="Final text output or tool execution result")
