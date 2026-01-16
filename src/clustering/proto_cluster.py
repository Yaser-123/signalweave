# src/clustering/proto_cluster.py

from typing import Dict, Any
from datetime import datetime
import uuid


def create_proto_cluster(
    contextualized_output: Dict[str, Any]
) -> Dict[str, Any]:
    all_signals = [
        contextualized_output["signal"],
        *contextualized_output.get("similar_signals", [])
    ]

    return {
        "cluster_id": str(uuid.uuid4()),
        "signals": all_signals,
        "signal_count": len(all_signals),
        "created_at": datetime.utcnow().isoformat()
    }