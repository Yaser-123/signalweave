# Relaxed Critic Agent Thresholds

## Overview

The Critic Agent evaluation thresholds have been **relaxed** to support **early-stage weak signal detection** during the hackathon phase. This prevents premature demotion of emerging trends that need time to accumulate evidence.

---

## Updated Evaluation Rules

### 🟢 HIGH Confidence (Promote to Active)

Clusters are promoted to active display when they meet ALL criteria:

- **Signal count:** ≥ 10 signals
- **Coherence:** ≥ 0.50 (moderate semantic similarity)
- **Source diversity:** ≥ 2 independent sources

**Rationale:** Meaningful emerging trends with sufficient cross-validation.

---

### 🟡 MEDIUM Confidence (Keep as Candidate)

Clusters remain in candidate memory when:

- **Signal count:** ≥ 3 signals
- **Coherence:** ≥ 0.30 (acceptable semantic similarity)
- **Source diversity:** Any (even single-source clusters can evolve)

**Rationale:** Early weak signals that need time to mature. Single-source clusters are NOT automatically demoted if signal count is sufficient.

---

### 🔴 LOW Confidence (Demote + Wait)

Clusters are demoted ONLY in extreme cases:

- **Signal count:** < 3 signals (insufficient evidence)
- **OR Coherence:** < 0.30 (very low semantic similarity - likely noise)

**Rationale:** Rare demotion for clear noise/spam. Most clusters survive and accumulate over time.

---

## Flag System Updates

### Coherence Flags

| Coherence Range | Flag | Action |
|----------------|------|--------|
| ≥ 0.70 | "high coherence" | Positive indicator |
| 0.40 - 0.69 | "weak coherence" | Acceptable for early signals |
| 0.30 - 0.39 | "weak coherence" | Monitor but keep |
| < 0.30 | "very low coherence" | Demote (likely noise) |

### Source Diversity Flags

- **Single source:** Flagged but NOT auto-demoted if signal count is high
- **Multi-source (≥3):** Flagged as "multi-source validated"

### Signal Count Flags

- **< 3 signals:** "insufficient evidence" → Demote
- **≥ 10 signals:** "strong evidence" → Promotes if coherence ≥ 0.50

---

## Impact on Pipeline

### Before (Strict Thresholds)

- HIGH required: 5 signals, 0.70 coherence, 2 sources
- Single-source clusters → Auto-demoted
- Coherence < 0.55 → Auto-demoted
- **Result:** Early weak signals killed prematurely

### After (Relaxed Thresholds)

- HIGH requires: 10 signals, 0.50 coherence, 2 sources
- Single-source clusters → Kept in candidate pool
- Coherence < 0.30 → Demoted (only extreme noise)
- **Result:** Early weak signals survive and accumulate over time

---

## Testing Validation

All 5 test cases pass:

1. ✅ **High confidence (10 signals, 0.96 coherence)** → Promote
2. ✅ **Medium confidence (5 signals, 0.71 coherence)** → Keep candidate
3. ✅ **Low confidence (5 signals, 0.25 coherence)** → Demote
4. ✅ **Medium confidence (8 signals, single source)** → Keep evolving
5. ✅ **Low confidence (2 signals)** → Demote

---

## Controller Logic (Unchanged)

The Controller Agent's decision policy remains the same:

```python
if confidence == "high":
    return "promote"  # To active display
elif confidence == "medium":
    return "keep_candidate"  # Let it evolve
else:  # low
    return "demote_wait"  # Wait for future evidence
```

---

## Design Philosophy

🔥 **Goal:** Early-stage weak signals should survive and accumulate over time instead of being killed too early.

- **Tolerance over precision:** Accept moderate coherence (0.50+) for signals with high counts
- **Time as evidence:** Let clusters evolve in candidate memory instead of aggressive pruning
- **Cross-validation optional:** Single-source clusters can promote if signal count is high enough
- **Noise reduction preserved:** Only demote extremely incoherent (<0.30) or tiny (<3 signals) clusters

---

## Production Considerations

When moving to production, consider:

1. **Adaptive thresholds** based on historical performance
2. **Domain-specific coherence baselines** (AI vs. climate signals may differ)
3. **User feedback loop** to calibrate confidence classification
4. **Numeric confidence scores** (0.0-1.0) instead of categorical for fine-tuning

---

## Files Modified

- `src/scoring/critic_agent.py` - Updated `_classify_confidence()` and flag thresholds
- `src/scoring/controller_agent.py` - Updated `_identify_primary_issue()` to handle new flags
- `test_critic_controller.py` - Updated test cases to match relaxed thresholds

---

**Last Updated:** 2026-02-05  
**Status:** ✅ All tests passing
