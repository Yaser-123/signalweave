# src/ingestion/mock_ingestor.py

from datetime import datetime, timedelta
from typing import List

from .signal import Signal


def load_mock_signals() -> List[Signal]:
    now = datetime.utcnow()

    signals = [
        Signal(
            signal_id="sig_001",
            text="Several research blogs report unexpected emergent behaviors in multi-agent systems without explicit coordination training.",
            timestamp=now - timedelta(days=5),
            source="research_blog",
            domain="emerging_technology",
            subdomain="ai",
            metadata={"confidence_hint": 0.6}
        ),
        Signal(
            signal_id="sig_002",
            text="Multiple startups mention rising difficulty in procuring high-end GPUs for large-scale model training.",
            timestamp=now - timedelta(days=15),
            source="tech_news",
            domain="emerging_technology",
            subdomain="compute",
            metadata={"confidence_hint": 0.7}
        ),
        Signal(
            signal_id="sig_003",
            text="Policy drafts in several regions discuss incentives for low-energy AI inference hardware.",
            timestamp=now - timedelta(days=30),
            source="policy_update",
            domain="emerging_technology",
            subdomain="energy",
            metadata={"confidence_hint": 0.5}
        ),
        Signal(
            signal_id="sig_004",
            text="Academic papers highlight new quantum computing algorithms that could disrupt classical encryption methods.",
            timestamp=now - timedelta(days=10),
            source="academic_journal",
            domain="emerging_technology",
            subdomain="compute",
            metadata={"confidence_hint": 0.8}
        )
    ]

    return signals