# src/scoring/controller_agent.py

from typing import Dict, Any, List


def controller_decide(cluster: Dict[str, Any], critic_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controller Agent: Make final decision based on critic's evaluation.
    
    Policy:
    - HIGH confidence → promote to active
    - MEDIUM confidence → keep as candidate
    - LOW confidence → demote, wait for future evidence
    
    Args:
        cluster: Cluster dict
        critic_report: Evaluation from critic_agent
    
    Returns:
        Dict with final_action and decision_trace
    """
    
    confidence = critic_report.get("confidence", "low")
    recommended_action = critic_report.get("recommended_action", "demote_wait")
    flags = critic_report.get("flags", [])
    metrics = critic_report.get("metrics", {})
    
    # Decision logic
    if confidence == "high":
        final_action = "promote"
        decision_trace = _generate_trace_high(metrics, flags)
    
    elif confidence == "medium":
        final_action = "keep_candidate"
        decision_trace = _generate_trace_medium(metrics, flags)
    
    else:  # low
        final_action = "demote_wait"
        decision_trace = _generate_trace_low(metrics, flags)
    
    return {
        "final_action": final_action,
        "decision_trace": decision_trace,
        "confidence": confidence,
        "flags": flags
    }


def _generate_trace_high(metrics: Dict[str, Any], flags: List[str]) -> str:
    """Generate decision trace for high confidence clusters."""
    signal_count = metrics.get("signal_count", 0)
    coherence = metrics.get("coherence", 0.0)
    sources = metrics.get("source_diversity", 0)
    
    return (
        f"High confidence → Promoted to active "
        f"({signal_count} signals, {sources} sources, coherence {coherence:.2f})"
    )


def _generate_trace_medium(metrics: Dict[str, Any], flags: List[str]) -> str:
    """Generate decision trace for medium confidence clusters."""
    signal_count = metrics.get("signal_count", 0)
    
    if "insufficient evidence" in flags:
        return f"Medium confidence → Kept as candidate (only {signal_count} signals, waiting for more)"
    
    return f"Medium confidence → Kept as candidate (tracking for future promotion)"


def _generate_trace_low(metrics: Dict[str, Any], flags: List[str]) -> str:
    """Generate decision trace for low confidence clusters."""
    primary_issue = _identify_primary_issue(flags, metrics)
    
    return f"Low confidence → Demoted to wait state ({primary_issue})"


def _identify_primary_issue(flags: List[str], metrics: Dict[str, Any]) -> str:
    """Identify the primary issue causing low confidence."""
    
    if "very low coherence" in flags or "weak coherence" in flags or "low coherence" in flags:
        coherence = metrics.get("coherence", 0.0)
        return f"coherence {coherence:.2f} too low"
    
    if "single source" in flags:
        return "single source only"
    
    if "insufficient evidence" in flags:
        signal_count = metrics.get("signal_count", 0)
        return f"only {signal_count} signals"
    
    return "waiting for future evidence"
