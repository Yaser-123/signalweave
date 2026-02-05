# test_critic_controller.py

"""Test Critic Agent + Controller Agent autonomous evaluation system"""

from src.scoring.critic_agent import evaluate_cluster
from src.scoring.controller_agent import controller_decide
from datetime import datetime, timedelta
import numpy as np


def create_test_cluster(signal_count, source_count, coherence_level="high"):
    """Helper to create test clusters with controlled parameters"""
    np.random.seed(42)
    
    # Create embeddings with controlled coherence
    embeddings = []
    
    if coherence_level == "very_low":
        # For very low coherence, create completely random vectors
        for i in range(signal_count):
            np.random.seed(42 + i * 100)  # Different seed for each
            embedding = np.random.randn(384)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
    else:
        # For other levels, use base vector + noise approach
        base_vector = np.random.randn(384)
        
        # Adjust noise based on coherence level
        noise_levels = {
            "high": 0.05,     # Very similar → high coherence (0.95+)
            "medium": 0.30,   # Moderately similar → medium coherence (0.50-0.70)
            "low": 1.20,      # Quite different → low coherence (0.30-0.40)
        }
        noise_factor = noise_levels.get(coherence_level, 0.30)
        
        for i in range(signal_count):
            noise = np.random.randn(384) * noise_factor
            embedding = base_vector + noise
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
    
    # Compute centroid
    centroid = np.mean(np.array(embeddings), axis=0).tolist()
    
    # Create signals with controlled source diversity
    now = datetime.utcnow()
    signals = []
    sources = [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://semianalysis.substack.com/feed",
        "https://www.datacenterdynamics.com/rss/"
    ]
    
    for i in range(signal_count):
        timestamp = now - timedelta(days=i*2)
        signal = {
            "signal_id": f"test_signal_{i}",
            "text": f"Test signal about AI trend {i}",
            "timestamp": timestamp.isoformat(),
            "source": sources[i % source_count],  # Limit to source_count sources
            "domain": "emerging_technology",
            "subdomain": "ai"
        }
        signals.append(signal)
    
    cluster = {
        "cluster_id": f"test_cluster_{signal_count}_{source_count}_{coherence_level}",
        "signals": signals,
        "embeddings": embeddings,
        "centroid": centroid,
        "signal_count": signal_count,
        "growth_ratio": 1.0,
        "created_at": (now - timedelta(days=30)).isoformat()
    }
    
    return cluster


def test_high_confidence_cluster():
    """Test cluster that should get HIGH confidence → PROMOTE"""
    print("\n" + "=" * 70)
    print("TEST 1: High Confidence Cluster (Relaxed Thresholds)")
    print("=" * 70)
    
    # Create: 10 signals, 2 sources, medium coherence (0.50+)
    cluster = create_test_cluster(signal_count=10, source_count=2, coherence_level="medium")
    
    # Critic evaluation
    critic_report = evaluate_cluster(cluster)
    
    print(f"\nCluster Setup:")
    print(f"  - Signals: {cluster['signal_count']}")
    print(f"  - Sources: 2 (arxiv, semianalysis)")
    print(f"  - Coherence: {critic_report['metrics']['coherence']:.2f} (moderate)")
    
    print(f"\n🧪 Critic Report:")
    print(f"  - Confidence: {critic_report['confidence']}")
    print(f"  - Flags: {critic_report['flags']}")
    print(f"  - Recommended: {critic_report['recommended_action']}")
    
    # Controller decision
    controller_decision = controller_decide(cluster, critic_report)
    
    print(f"\n🤖 Controller Decision:")
    print(f"  - Action: {controller_decision['final_action']}")
    print(f"  - Trace: {controller_decision['decision_trace']}")
    
    # Assertions
    assert critic_report['confidence'] == "high", f"Expected high, got {critic_report['confidence']}"
    assert critic_report['recommended_action'] == "promote", "Should recommend promote"
    assert controller_decision['final_action'] == "promote", "Should promote to active"
    assert "strong evidence" in critic_report['flags'], "Should flag strong evidence (10+ signals)"
    
    print("\n✅ HIGH CONFIDENCE TEST PASSED")


def test_medium_confidence_cluster():
    """Test cluster that should get MEDIUM confidence → KEEP CANDIDATE"""
    print("\n" + "=" * 70)
    print("TEST 2: Medium Confidence Cluster (Early Weak Signal)")
    print("=" * 70)
    
    # Create: 5 signals, 2 sources, low coherence (0.40+)
    cluster = create_test_cluster(signal_count=5, source_count=2, coherence_level="low")
    
    # Critic evaluation
    critic_report = evaluate_cluster(cluster)
    
    print(f"\nCluster Setup:")
    print(f"  - Signals: {cluster['signal_count']}")
    print(f"  - Sources: 2")
    print(f"  - Coherence: {critic_report['metrics']['coherence']:.2f} (acceptable)")
    
    print(f"\n🧪 Critic Report:")
    print(f"  - Confidence: {critic_report['confidence']}")
    print(f"  - Flags: {critic_report['flags']}")
    print(f"  - Recommended: {critic_report['recommended_action']}")
    
    # Controller decision
    controller_decision = controller_decide(cluster, critic_report)
    
    print(f"\n🤖 Controller Decision:")
    print(f"  - Action: {controller_decision['final_action']}")
    print(f"  - Trace: {controller_decision['decision_trace']}")
    
    # Assertions
    assert critic_report['confidence'] == "medium", f"Expected medium, got {critic_report['confidence']}"
    assert controller_decision['final_action'] == "keep_candidate", "Should keep as candidate"
    
    print("\n✅ MEDIUM CONFIDENCE TEST PASSED")


def test_low_confidence_low_coherence():
    """Test cluster with LOW confidence due to very low coherence → DEMOTE"""
    print("\n" + "=" * 70)
    print("TEST 3: Low Confidence (Very Low Coherence <0.30)")
    print("=" * 70)
    
    # Create: 5 signals, 2 sources, but VERY LOW coherence
    cluster = create_test_cluster(signal_count=5, source_count=2, coherence_level="very_low")
    
    # Manually inject very low coherence for testing (since generating truly random coherence is unreliable)
    # This simulates a cluster where signals are semantically unrelated
    from src.scoring.grounding_agent import compute_cluster_grounding
    grounding = compute_cluster_grounding(cluster)
    # Override with test value
    cluster["coherence"] = 0.25  # Below 0.30 threshold
    
    # Critic evaluation
    from src.scoring.critic_agent import evaluate_cluster
    critic_report = evaluate_cluster(cluster)
    
    print(f"\nCluster Setup:")
    print(f"  - Signals: {cluster['signal_count']}")
    print(f"  - Sources: 2")
    print(f"  - Coherence: {cluster['coherence']:.2f} (very low - manually set for testing)")
    
    print(f"\n🧪 Critic Report:")
    print(f"  - Confidence: {critic_report['confidence']}")
    print(f"  - Flags: {critic_report['flags']}")
    print(f"  - Recommended: {critic_report['recommended_action']}")
    
    # Controller decision
    from src.scoring.controller_agent import controller_decide
    controller_decision = controller_decide(cluster, critic_report)
    
    print(f"\n🤖 Controller Decision:")
    print(f"  - Action: {controller_decision['final_action']}")
    print(f"  - Trace: {controller_decision['decision_trace']}")
    
    # Assertions
    assert critic_report['confidence'] == "low", f"Expected low, got {critic_report['confidence']}"
    assert "very low coherence" in critic_report['flags'], "Should flag very low coherence"
    assert controller_decision['final_action'] == "demote_wait", "Should demote"
    assert "coherence" in controller_decision['decision_trace'].lower(), "Trace should mention coherence"
    
    print("\n✅ LOW CONFIDENCE (COHERENCE) TEST PASSED")


def test_medium_confidence_single_source():
    """Test cluster with single source but high signal count → MEDIUM (Keep evolving)"""
    print("\n" + "=" * 70)
    print("TEST 4: Medium Confidence (Single Source, High Count)")
    print("=" * 70)
    
    # Create: 8 signals, 1 source only, medium coherence
    cluster = create_test_cluster(signal_count=8, source_count=1, coherence_level="medium")
    
    # Critic evaluation
    critic_report = evaluate_cluster(cluster)
    
    print(f"\nCluster Setup:")
    print(f"  - Signals: {cluster['signal_count']}")
    print(f"  - Sources: 1 (single source)")
    print(f"  - Coherence: {critic_report['metrics']['coherence']:.2f}")
    
    print(f"\n🧪 Critic Report:")
    print(f"  - Confidence: {critic_report['confidence']}")
    print(f"  - Flags: {critic_report['flags']}")
    print(f"  - Recommended: {critic_report['recommended_action']}")
    
    # Controller decision
    controller_decision = controller_decide(cluster, critic_report)
    
    print(f"\n🤖 Controller Decision:")
    print(f"  - Action: {controller_decision['final_action']}")
    print(f"  - Trace: {controller_decision['decision_trace']}")
    
    # Assertions - single source NO LONGER auto-demotes if signal count is sufficient
    assert critic_report['confidence'] == "medium", f"Expected medium, got {critic_report['confidence']}"
    assert "single source" in critic_report['flags'], "Should flag single source"
    assert controller_decision['final_action'] == "keep_candidate", "Should keep as candidate (not demote)"
    
    print("\n✅ MEDIUM CONFIDENCE (SINGLE SOURCE) TEST PASSED")


def test_low_confidence_insufficient_evidence():
    """Test cluster with LOW confidence due to insufficient signals → DEMOTE"""
    print("\n" + "=" * 70)
    print("TEST 5: Low Confidence (Insufficient Evidence)")
    print("=" * 70)
    
    # Create: only 2 signals, 2 sources, high coherence
    cluster = create_test_cluster(signal_count=2, source_count=2, coherence_level="high")
    
    # Critic evaluation
    critic_report = evaluate_cluster(cluster)
    
    print(f"\nCluster Setup:")
    print(f"  - Signals: {cluster['signal_count']} (insufficient)")
    print(f"  - Sources: 2")
    print(f"  - Coherence: {critic_report['metrics']['coherence']:.2f}")
    
    print(f"\n🧪 Critic Report:")
    print(f"  - Confidence: {critic_report['confidence']}")
    print(f"  - Flags: {critic_report['flags']}")
    print(f"  - Recommended: {critic_report['recommended_action']}")
    
    # Controller decision
    controller_decision = controller_decide(cluster, critic_report)
    
    print(f"\n🤖 Controller Decision:")
    print(f"  - Action: {controller_decision['final_action']}")
    print(f"  - Trace: {controller_decision['decision_trace']}")
    
    # Assertions
    assert critic_report['confidence'] == "low", f"Expected low, got {critic_report['confidence']}"
    assert "insufficient evidence" in critic_report['flags'], "Should flag insufficient evidence"
    assert controller_decision['final_action'] == "demote_wait", "Should demote"
    
    print("\n✅ LOW CONFIDENCE (INSUFFICIENT EVIDENCE) TEST PASSED")


def run_all_tests():
    """Run all critic + controller tests"""
    print("\n" + "=" * 70)
    print("CRITIC + CONTROLLER AGENT TEST SUITE (RELAXED THRESHOLDS)")
    print("=" * 70)
    
    test_high_confidence_cluster()
    test_medium_confidence_cluster()
    test_low_confidence_low_coherence()
    test_medium_confidence_single_source()
    test_low_confidence_insufficient_evidence()
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary (Relaxed Thresholds for Early Weak Signals):")
    print("✅ HIGH confidence (10+ signals, 0.50+ coherence, 2+ sources) → promote to active")
    print("✅ MEDIUM confidence (3-9 signals, 0.30+ coherence) → keep as candidate")
    print("✅ MEDIUM confidence (single source, high count) → keep evolving")
    print("✅ LOW confidence (<3 signals) → demote, wait for evidence")
    print("✅ LOW confidence (<0.30 coherence) → demote, wait for evidence")
    print("\nThe Critic + Controller system now tolerates early-stage weak signals!")


if __name__ == "__main__":
    run_all_tests()
