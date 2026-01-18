# src/memory/candidate_store.py

import json
import os
from typing import List, Dict, Any

CANDIDATE_STORE_FILE = "candidate_clusters.json"


def load_candidates() -> List[Dict[str, Any]]:
    if not os.path.exists(CANDIDATE_STORE_FILE):
        return []
    with open(CANDIDATE_STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidates(candidates: List[Dict[str, Any]]):
    with open(CANDIDATE_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)