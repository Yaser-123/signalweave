# src/clustering/cluster_evolution.py

from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import uuid


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    return np.mean(np.array(embeddings), axis=0).tolist()


def evolve_clusters(
    existing_candidates: List[Dict[str, Any]],
    new_batch_clusters: List[Dict[str, Any]],
    embedding_model,
    similarity_threshold: float = 0.70
) -> List[Dict[str, Any]]:
    """
    existing_candidates: stored candidate clusters from previous runs
    new_batch_clusters: proto-clusters formed in current run
    """

    # Prepare existing candidates with centroids
    for c in existing_candidates:
        if "centroid" not in c:
            texts = [s["text"] for s in c["signals"]]
            embeddings = [embedding_model.embed(t) for t in texts]
            c["embeddings"] = embeddings
            c["centroid"] = compute_centroid(embeddings)

    for new_cluster in new_batch_clusters:
        texts = [s["text"] for s in new_cluster["signals"]]
        new_embeddings = [embedding_model.embed(t) for t in texts]
        new_centroid = compute_centroid(new_embeddings)

        merged = False

        for candidate in existing_candidates:
            sim = cosine_similarity(new_centroid, candidate["centroid"])

            if sim >= similarity_threshold:
                # Merge - but avoid duplicate signals
                existing_signal_ids = {s["signal_id"] for s in candidate["signals"]}
                
                # Only add new signals that aren't already in the cluster
                new_signals_to_add = [
                    s for s in new_cluster["signals"] 
                    if s["signal_id"] not in existing_signal_ids
                ]
                
                if new_signals_to_add:
                    # Compute embeddings only for truly new signals
                    new_signal_embeddings = [
                        embedding_model.embed(s["text"]) 
                        for s in new_signals_to_add
                    ]
                    
                    candidate["signals"].extend(new_signals_to_add)
                    candidate["embeddings"].extend(new_signal_embeddings)
                    candidate["centroid"] = compute_centroid(candidate["embeddings"])
                    candidate["signal_count"] = len(candidate["signals"])
                
                merged = True
                break

        if not merged:
            # create new candidate
            existing_candidates.append({
                "cluster_id": str(uuid.uuid4()),
                "signals": new_cluster["signals"],
                "embeddings": new_embeddings,
                "centroid": new_centroid,
                "signal_count": len(new_cluster["signals"]),
                "created_at": datetime.utcnow().isoformat()
            })

    return existing_candidates