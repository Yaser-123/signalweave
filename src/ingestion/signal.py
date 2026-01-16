# src/ingestion/signal.py

from datetime import datetime
from typing import Dict, Any


class Signal:
    def __init__(
        self,
        signal_id: str,
        text: str,
        timestamp: datetime,
        source: str,
        domain: str,
        subdomain: str,
        metadata: Dict[str, Any] = None
    ):
        self.signal_id = signal_id
        self.text = text
        self.timestamp = timestamp
        self.source = source
        self.domain = domain
        self.subdomain = subdomain
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        return cls(
            signal_id=data["signal_id"],
            text=data["text"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            domain=data["domain"],
            subdomain=data["subdomain"],
            metadata=data.get("metadata", {})
        )