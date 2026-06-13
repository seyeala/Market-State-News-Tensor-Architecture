"""Event ontology and rulebook placeholders.

The ontology constrains ex-ante :class:`EventObject` instances to a small,
explicit set of context types, event types, and affected metrics.  This makes
future event tensorization deterministic: every accepted event has a known
semantic bucket and a known feature target.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from market_state_news_tensor.schemas.events import EventObject

ALLOWED_CONTEXT_TYPES: frozenset[str] = frozenset(
    {
        "geopolitical",
        "macro",
        "earnings",
        "fundamental",
        "legal/regulatory",
        "product",
        "market-belief",
        "opinion",
    }
)

VALID_EVENT_TYPES_BY_CONTEXT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "geopolitical": frozenset(
            {
                "conflict_escalation",
                "conflict_deescalation",
                "sanctions",
                "trade_policy",
                "election_result",
            }
        ),
        "macro": frozenset(
            {
                "inflation_print",
                "jobs_report",
                "central_bank_decision",
                "gdp_release",
                "rates_outlook",
            }
        ),
        "earnings": frozenset(
            {
                "earnings_beat",
                "earnings_miss",
                "revenue_beat",
                "revenue_miss",
                "margin_change",
                "guidance_update",
            }
        ),
        "fundamental": frozenset(
            {
                "guidance_raise",
                "guidance_cut",
                "analyst_upgrade",
                "analyst_downgrade",
                "capital_allocation",
                "management_change",
            }
        ),
        "legal/regulatory": frozenset(
            {
                "lawsuit_filed",
                "lawsuit_resolved",
                "regulatory_approval",
                "regulatory_investigation",
                "compliance_action",
            }
        ),
        "product": frozenset(
            {
                "product_launch",
                "product_delay",
                "product_recall",
                "partnership",
                "supply_constraint",
            }
        ),
        "market-belief": frozenset(
            {
                "positioning_shift",
                "sentiment_shift",
                "narrative_change",
                "crowding_signal",
                "liquidity_signal",
            }
        ),
        "opinion": frozenset(
            {
                "bullish_commentary",
                "bearish_commentary",
                "neutral_analysis",
                "rumor",
                "expert_interview",
            }
        ),
    }
)

VALID_AFFECTED_METRICS_BY_CONTEXT_EVENT: Mapping[str, Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "geopolitical": MappingProxyType(
            {
                "conflict_escalation": frozenset({"risk_premium", "volatility", "supply", "demand"}),
                "conflict_deescalation": frozenset({"risk_premium", "volatility", "supply", "demand"}),
                "sanctions": frozenset({"revenue", "cost", "supply", "liquidity"}),
                "trade_policy": frozenset({"revenue", "cost", "supply", "demand"}),
                "election_result": frozenset({"risk_premium", "regulatory_risk", "demand", "cost"}),
            }
        ),
        "macro": MappingProxyType(
            {
                "inflation_print": frozenset({"rates", "valuation_multiple", "demand", "margin"}),
                "jobs_report": frozenset({"rates", "demand", "wages", "margin"}),
                "central_bank_decision": frozenset({"rates", "liquidity", "valuation_multiple", "volatility"}),
                "gdp_release": frozenset({"growth", "demand", "revenue", "risk_premium"}),
                "rates_outlook": frozenset({"rates", "valuation_multiple", "liquidity", "volatility"}),
            }
        ),
        "earnings": MappingProxyType(
            {
                "earnings_beat": frozenset({"eps", "margin", "profitability", "cash_flow"}),
                "earnings_miss": frozenset({"eps", "margin", "profitability", "cash_flow"}),
                "revenue_beat": frozenset({"revenue", "growth", "demand"}),
                "revenue_miss": frozenset({"revenue", "growth", "demand"}),
                "margin_change": frozenset({"margin", "cost", "profitability"}),
                "guidance_update": frozenset({"growth", "revenue", "eps", "margin"}),
            }
        ),
        "fundamental": MappingProxyType(
            {
                "guidance_raise": frozenset({"growth", "revenue", "eps", "margin", "cash_flow"}),
                "guidance_cut": frozenset({"growth", "revenue", "eps", "margin", "cash_flow"}),
                "analyst_upgrade": frozenset({"valuation_multiple", "growth", "risk_premium"}),
                "analyst_downgrade": frozenset({"valuation_multiple", "growth", "risk_premium"}),
                "capital_allocation": frozenset({"cash_flow", "leverage", "share_count", "dividend"}),
                "management_change": frozenset({"execution_risk", "growth", "margin", "risk_premium"}),
            }
        ),
        "legal/regulatory": MappingProxyType(
            {
                "lawsuit_filed": frozenset({"legal_risk", "cost", "cash_flow", "risk_premium"}),
                "lawsuit_resolved": frozenset({"legal_risk", "cost", "cash_flow", "risk_premium"}),
                "regulatory_approval": frozenset({"revenue", "growth", "regulatory_risk"}),
                "regulatory_investigation": frozenset({"regulatory_risk", "cost", "revenue", "risk_premium"}),
                "compliance_action": frozenset({"regulatory_risk", "cost", "margin"}),
            }
        ),
        "product": MappingProxyType(
            {
                "product_launch": frozenset({"revenue", "growth", "demand", "market_share"}),
                "product_delay": frozenset({"revenue", "growth", "execution_risk", "market_share"}),
                "product_recall": frozenset({"cost", "revenue", "brand_risk", "legal_risk"}),
                "partnership": frozenset({"revenue", "growth", "distribution", "market_share"}),
                "supply_constraint": frozenset({"supply", "revenue", "margin", "cost"}),
            }
        ),
        "market-belief": MappingProxyType(
            {
                "positioning_shift": frozenset({"sentiment", "liquidity", "volatility", "momentum"}),
                "sentiment_shift": frozenset({"sentiment", "momentum", "valuation_multiple", "risk_premium"}),
                "narrative_change": frozenset({"sentiment", "growth", "valuation_multiple", "risk_premium"}),
                "crowding_signal": frozenset({"liquidity", "volatility", "reversal_risk", "momentum"}),
                "liquidity_signal": frozenset({"liquidity", "volatility", "spread", "market_depth"}),
            }
        ),
        "opinion": MappingProxyType(
            {
                "bullish_commentary": frozenset({"sentiment", "valuation_multiple", "momentum"}),
                "bearish_commentary": frozenset({"sentiment", "valuation_multiple", "momentum"}),
                "neutral_analysis": frozenset({"sentiment", "uncertainty"}),
                "rumor": frozenset({"sentiment", "uncertainty", "volatility"}),
                "expert_interview": frozenset({"sentiment", "growth", "risk_premium"}),
            }
        ),
    }
)


@dataclass(frozen=True)
class DecayRule:
    """Placeholder decay configuration for future state-tensor updates."""

    half_life_hours: float
    floor_weight: float = 0.0


@dataclass(frozen=True)
class ScoringRule:
    """Placeholder scoring weights for future event impact calculations."""

    magnitude_weight: float = 1.0
    uncertainty_weight: float = -1.0
    source_quality_weight: float = 1.0
    novelty_weight: float = 1.0


@dataclass(frozen=True)
class ContradictionRule:
    """Placeholder contradiction behavior for future reliability gates."""

    lookback_hours: float
    max_direction_gap: float
    action: str


DECAY_RULES_BY_CONTEXT: Mapping[str, DecayRule] = MappingProxyType(
    {
        "geopolitical": DecayRule(half_life_hours=72.0),
        "macro": DecayRule(half_life_hours=24.0),
        "earnings": DecayRule(half_life_hours=48.0),
        "fundamental": DecayRule(half_life_hours=168.0),
        "legal/regulatory": DecayRule(half_life_hours=336.0),
        "product": DecayRule(half_life_hours=168.0),
        "market-belief": DecayRule(half_life_hours=12.0),
        "opinion": DecayRule(half_life_hours=6.0),
    }
)

SCORING_RULES_BY_CONTEXT: Mapping[str, ScoringRule] = MappingProxyType(
    {context: ScoringRule() for context in ALLOWED_CONTEXT_TYPES}
)

CONTRADICTION_RULES_BY_CONTEXT: Mapping[str, ContradictionRule] = MappingProxyType(
    {
        context: ContradictionRule(
            lookback_hours=DECAY_RULES_BY_CONTEXT[context].half_life_hours,
            max_direction_gap=1.0,
            action="increase_uncertainty",
        )
        for context in ALLOWED_CONTEXT_TYPES
    }
)


def validate_event_against_ontology(event: EventObject) -> None:
    """Validate an event's context, type, and metrics against the ontology.

    Raises:
        ValueError: If the event has an unknown context type, an event type not
            mapped for that context, or an affected metric not mapped for the
            context/event-type pair.
    """

    if event.context_type not in ALLOWED_CONTEXT_TYPES:
        raise ValueError(f"unknown context_type: {event.context_type!r}")

    valid_event_types = VALID_EVENT_TYPES_BY_CONTEXT[event.context_type]
    if event.event_type not in valid_event_types:
        raise ValueError(
            f"unknown event_type {event.event_type!r} for context_type {event.context_type!r}; "
            f"valid event types: {sorted(valid_event_types)}"
        )

    valid_metrics = VALID_AFFECTED_METRICS_BY_CONTEXT_EVENT[event.context_type][event.event_type]
    invalid_metrics = sorted(set(event.affected_metrics) - valid_metrics)
    if invalid_metrics:
        raise ValueError(
            f"invalid affected_metrics for context_type {event.context_type!r} "
            f"and event_type {event.event_type!r}: {invalid_metrics}; "
            f"valid metrics: {sorted(valid_metrics)}"
        )
