# src/scoring/critic_agent.py

from typing import Dict, Any, List


def evaluate_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critic Agent: Evaluate cluster quality based on evidence metrics.
    
    Evaluates:
    - Signal count (evidence breadth)
    - Source diversity (cross-validation)
    - Semantic coherence (cluster tightness)
    
    Args:
        cluster: Cluster dict with signals, embeddings, signal_count
    
    Returns:
        Dict with confidence level, flags, and recommended action
    """
    
    # Extract metrics
    signal_count = cluster.get("signal_count", len(cluster.get("signals", [])))
    
    # Source diversity
    signals = cluster.get("signals", [])
    unique_sources = len(set(s.get("source", "unknown") for s in signals))
    
    # Semantic coherence (from grounding agent or compute on-the-fly)
    coherence = cluster.get("coherence", 0.0)
    
    # If coherence not pre-computed, estimate from embeddings
    if coherence == 0.0 and cluster.get("embeddings"):
        from src.scoring.grounding_agent import compute_cluster_grounding
        grounding = compute_cluster_grounding(cluster)
        coherence = grounding.get("coherence", 0.0)
    
    # Evaluation flags
    flags = []
    
    # Coherence flags (relaxed thresholds)
    if coherence < 0.30:
        flags.append("very low coherence")
    elif coherence < 0.40:
        flags.append("weak coherence")
    elif coherence >= 0.70:
        flags.append("high coherence")
    
    # Source diversity flags
    if unique_sources == 1:
        flags.append("single source")
    elif unique_sources >= 3:
        flags.append("multi-source validated")
    
    # Signal count flags
    if signal_count < 3:
        flags.append("insufficient evidence")
    elif signal_count >= 10:
        flags.append("strong evidence")
    
    # Confidence classification
    confidence = _classify_confidence(signal_count, coherence, unique_sources)
    
    # Recommended action based on confidence
    recommended_action = _recommend_action(confidence, signal_count)
    
    return {
        "confidence": confidence,
        "flags": flags,
        "recommended_action": recommended_action,
        "metrics": {
            "signal_count": signal_count,
            "source_diversity": unique_sources,
            "coherence": coherence
        }
    }


def _classify_confidence(
    signal_count: int, 
    coherence: float, 
    source_diversity: int
) -> str:
    """
    Classify cluster confidence level (relaxed for early weak signals).
    
    HIGH: count ≥10, coherence ≥0.50, sources ≥2
    MEDIUM: count ≥3, coherence ≥0.40
    LOW: count <3 OR coherence <0.30 (rare, only extreme cases)
    """
    
    # HIGH confidence criteria (meaningful emerging trend)
    if signal_count >= 10 and coherence >= 0.50 and source_diversity >= 2:
        return "high"
    
    # LOW confidence criteria (only extreme cases)
    # Single source is NOT an auto-disqualifier if signal count is high
    if signal_count < 3:
        return "low"
    if coherence < 0.30:
        return "low"
    
    # MEDIUM confidence (default - let clusters evolve)
    # Includes: 3-9 signals, coherence ≥0.30, any source diversity
    return "medium"


def _recommend_action(confidence: str, signal_count: int) -> str:
    """
    Recommend action based on confidence level.
    
    promote: High confidence, ready for active display
    keep_candidate: Medium confidence, keep tracking
    demote_wait: Low confidence, wait for future evidence
    """
    
    if confidence == "high":
        return "promote"
    elif confidence == "medium":
        # Only promote medium if signal count is at least 3
        if signal_count >= 3:
            return "keep_candidate"
        else:
            return "demote_wait"
    else:  # low
        return "demote_wait"
